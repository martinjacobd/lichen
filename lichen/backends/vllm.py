"""Read the same typed decisions from a vLLM endpoint instead of a local GGUF.

Lichen's method is engine-agnostic: build a prompt that ends where the label goes, read the
softmax over the label tokens of one next-token distribution. Only the way the logits arrive is
llama.cpp-specific. This module gets them over HTTP from vLLM; `method` does the rest, so
Method, variants, rotations, fibers, readings, shrink, disagreement, runoff and confidence are
the same code that serves a GGUF, not a second copy of it.

Two reasons it is worth having:

  * The local server answers one request at a time (see README's "Limits"). vLLM does
    continuous batching, and with --enable-prefix-caching gives for free the shared-prefix
    reuse that --batch hand-rolls. For a sweep over thousands of decisions that serialisation
    is the whole cost.
  * vLLM cannot read GGUF at all (its V1 engine lists GGUF as removed), so this is the only way
    to put the same question to an FP8 or AWQ checkpoint.

Each request names its label tokens in `allowed_token_ids`, and the endpoint must run with
--logprobs-mode processed_logprobs. The log-probabilities then come after the mask, so they are
the softmax over the label tokens alone, and the top-N list holds every label and nothing else.
With a raw mode the top-N is taken over the whole vocabulary, and a confident model ranks its
unlikely labels below other tokens: on gemma-4-26B-A4B, 3 of 6 labels were missing from the top
64. The greedy request applies no temperature, so the processed values are the masked logits
less one constant, which cancels in the softmax over the labels -- provided no penalty moves
them, so the request pins the penalties rather than inherit the server's defaults.

The server needs --max-logprobs >= (options x fibers), at most 62; its default of 20 is too low
for a wide choice. `_probs` still refuses a reply that lacks a label, which would otherwise read
as probability zero. (62 is len(method.LABELS); a raw-mode server fails that check on most wide
choices, which is how a missing --logprobs-mode shows up.)

vLLM is not deterministic across batches: the same request gave label probabilities up to 0.09
apart on gemma-4-26B-A4B, enough to move a borderline answer. VLLM_BATCH_INVARIANT=1 in the
server's environment made two full JevBench runs agree on all 231 items, for ~23% latency.
A launch that serves this backend reproducibly:

    VLLM_BATCH_INVARIANT=1 vllm serve MODEL --enable-prefix-caching \\
      --logprobs-mode processed_logprobs --max-logprobs 64 --max-model-len 32768

`usage.input_tokens` is the full prompt as vLLM counts it. The llama.cpp backend counts only the
tokens it evaluated after its cached prefix, so the two are not comparable as costs.

--thinking DOES NOT USE THE CHAT ENDPOINT, and both reasons were found the hard way against a
live Qwen3.8-27B (2026-09-26):

  * The trace cannot be read back from it. With enable_thinking=True the template OPENS <think>
    in the prompt, so the model's output holds </think> but never <think>; vLLM's qwen3
    reasoning parser finds no opener, keeps only the text after </think>, and DROPS the
    reasoning. 416 generated tokens arrived as reasoning_content='' and content='\n\nyes' --
    and "yes" is a non-empty string, so a guard on emptiness passes it through as the trace.
  * The label cannot be read at the right position. Splicing the trace in as a partial assistant
    turn (continue_final_message) loses the trailing blank line, because the template trims
    assistant content: the prompt ends at `</think>`, so the next token is the \n\n the model
    always emits there (p ~ 1.0) and the labels sit ~11 nats down. A softmax over them still
    returns a confident-looking distribution, which is the silent part.

So the reasoning stage and the thinking label read both go through /tokenize + /v1/completions,
which applies no reasoning parser and takes a token-id prompt. The splice is then the token-level
twin of llamacpp's splice_trace: the non-thinking render already ends `<think>\n\n</think>\n\n`,
so putting the trace inside that empty block preserves the label position exactly. Verified: the
completions path returns the SAME distribution as the chat path on the same prompt (Yes -0.181,
No -1.806), so nothing about the published no-thinking numbers moves. Without --thinking the chat
endpoint is still used, unchanged -- one round trip, as the one-forward-pass method intends.

Not ported: --recheck, whose second round would want the endpoint's own answer written back,
and --embedding. Asked for, they raise rather than be ignored. Nor are the prompts that
llamacpp.render builds per model -- Granite Guardian's criteria, Qwen3Guard's plain ChatML, and
closing a reasoning block a template leaves open (LFM2.5) -- so those models are refused rather
than read with a prompt they were never given (`check_model`). Nothing here needs vLLM
installed: the engine is reached over HTTP, and a `pip install lichen` is the whole dependency.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib import error, request

import numpy

from ..method import Method, chat_messages, label_constraint, reasoning_messages


THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


class NoReasoningBlock(ValueError):
    """The model's chat template has no reasoning block to put a trace into.

    Same name and same meaning as the llamacpp backend's: refused rather than guessed at,
    because injecting <think> into a model never trained on it produces confident nonsense.
    """


class Endpoint:
    """A vLLM OpenAI-compatible endpoint, in place of a model on this machine."""

    threaded = True  # vLLM batches for itself; requests may be served concurrently

    def __init__(self, endpoint: str, model: str, method: Method, top_logprobs: int = 64,
                 workers: int = 8, served: str | None = None, timeout: float = 600.0):
        for unsupported in ("recheck", "embedding"):
            if getattr(method, unsupported):
                raise SystemExit(f"--{unsupported} is not ported to the vLLM backend")
        self.endpoint = endpoint.rstrip("/")
        self.name = model           # the name a reply carries
        self.served = served or model  # the name this endpoint knows it by
        self.where = f"vLLM {self.served} at {self.endpoint}"
        self.top_logprobs = top_logprobs
        self.workers = workers
        self.timeout = timeout
        self._tokens: dict[str, int] = {}
        self._model_checked = False
        self.mask = True  # False reads the unmasked top-N: a diagnostic, see `_read`
        self._ids: dict[str, int] = {}   # <think>, </think>, newlines: for the token-level splice

    def close(self) -> None:
        pass  # the weights are the endpoint's

    def _post(self, path: str, body: dict) -> dict:
        req = request.Request(f"{self.endpoint}{path}", data=json.dumps(body).encode(),
                              headers={"Content-Type": "application/json"})
        try:
            with request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except error.HTTPError as exc:
            raise RuntimeError(f"{path} -> HTTP {exc.code}: {exc.read()[:400].decode(errors='replace')}") from exc

    def _get(self, path: str) -> dict:
        try:
            with request.urlopen(f"{self.endpoint}{path}", timeout=self.timeout) as r:
                return json.load(r)
        except error.HTTPError as exc:
            raise RuntimeError(f"{path} -> HTTP {exc.code}: {exc.read()[:400].decode(errors='replace')}") from exc

    def check_model(self, messages: list[dict], method: Method) -> None:
        """Refuse a model whose lichen prompt only the llama.cpp backend knows how to build.

        llamacpp.render gives Granite Guardian and Qwen3Guard their own prompts, and closes a
        reasoning block the template leaves open. Here the server's template renders the plain
        messages, so those models would be read at the wrong place or with the wrong question --
        and a distribution comes back either way. `--model` is a served name, which need not say
        what it serves, so the checkpoint path the server reports is checked too; and the open
        block is found in the rendered prompt itself, whatever the model is called. Once per
        endpoint: the answer cannot change while it is up.
        """
        if self._model_checked:
            return
        served = self._get("/v1/models").get("data", [])
        root = next((m.get("root") or "" for m in served if m.get("id") == self.served), "")
        names = f"{self.name} {self.served} {root}".lower()
        for family in ("granite-guardian", "qwen3guard"):
            if family in names:
                raise RuntimeError(f"{family} needs a prompt only the llama.cpp backend builds; "
                                   f"serve it from a GGUF rather than --vllm-endpoint")
        strs = self._post("/tokenize", {
            "model": self.served, "messages": messages, "add_generation_prompt": True,
            "return_token_strs": True, "chat_template_kwargs": {"enable_thinking": False},
        }).get("token_strs") or []
        # --thinking splices its trace into that block and closes it, so only a plain read
        # is left reading reasoning.
        if not method.thinking and strs and strs[-1] == THINK_OPEN:
            raise RuntimeError(
                "this model's chat template opens a reasoning block with thinking off, so the "
                "next token is reasoning, not a label. The llama.cpp backend closes it; this one "
                "does not -- serve it from a GGUF rather than --vllm-endpoint")
        self._model_checked = True

    def _id(self, piece: str) -> int:
        """The single token id of `piece`, cached. Raises if it is not one token."""
        if piece not in self._ids:
            ids = self._post("/tokenize", {"model": self.served, "prompt": piece,
                                           "add_special_tokens": False}).get("tokens", [])
            if len(ids) != 1:
                raise RuntimeError(f"{piece!r} is {len(ids)} tokens on this tokenizer, not one")
            self._ids[piece] = ids[0]
        return self._ids[piece]

    def _render(self, messages: list[dict], thinking: bool) -> list[int]:
        """The prompt the server would build from `messages`, as token ids.

        /tokenize applies the server's own chat template, so the template stays the server's
        business (the reason lichen can point at any endpoint). Ids rather than text because
        /tokenize's token_strs are byte-BPE forms -- 'Ġone', 'Ċ' -- which do not concatenate
        back into the prompt, and a detokenize round trip is both an extra call and a chance to
        alter what the model sees.
        """
        return self._post("/tokenize", {
            "model": self.served, "messages": messages, "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": thinking},
        })["tokens"]

    def _fragment(self, text: str) -> list[int]:
        """Token ids for a plain string, with no template and no special tokens."""
        return self._post("/tokenize", {"model": self.served, "prompt": text,
                                        "add_special_tokens": False})["tokens"]

    def splice_trace(self, ids: list[int], trace_ids: list[int],
                     labels: list[str] | None = None) -> list[int]:
        """`ids` with the trace inside its reasoning block, still ending where the label goes.

        The token-level twin of llamacpp.splice_trace, and it protects the same invariant: the
        prompt must end exactly at the answer position. With enable_thinking=False the Qwen3.8
        template renders `<think>\n\n</think>\n\n` -- an empty block, then the blank line the
        label follows -- so replacing that block's interior leaves everything after `</think>`
        untouched. A template that leaves the block open instead gets it closed here.
        """
        opener, closer, nl = self._id(THINK_OPEN), self._id(THINK_CLOSE), self._id("\n")
        try:
            at = len(ids) - 1 - ids[::-1].index(opener)
        except ValueError:
            raise NoReasoningBlock(
                "this model's chat template opens no reasoning block, so --thinking has nowhere "
                "to put the trace. Run it without --thinking, or use a thinking model.") from None
        body = list(trace_ids)
        if labels:
            # The trace reasons about option content and never names a label, so the block ends by
            # saying which tokens the answer may be -- see method.label_constraint for what this
            # is worth (captured mass 0.465 -> 0.929, and 24/50 argmaxes on a label -> 50/50).
            body += self._fragment("\n\n" + label_constraint(labels))
        head = ids[:at + 1] + [nl] + body + [nl]
        tail = ids[at + 1:]
        if closer in tail:
            return head + tail[tail.index(closer):]
        # Block left open by the template: close it and supply the blank line ourselves.
        return head + [closer, self._id("\n\n")]

    def check_labels(self, labels: list[str]) -> None:
        """lichen's invariant: every label is exactly one token, and no two share one.

        Same check, same reason -- a label that splits cannot be read from one next-token
        distribution, so it is an error rather than a quietly wrong answer.
        """
        unknown = [l for l in labels if l not in self._tokens]
        for label in unknown:
            ids = self._post("/tokenize", {"model": self.served, "prompt": label,
                                           "add_special_tokens": False}).get("tokens", [])
            if len(ids) != 1:
                raise ValueError(f"label {label!r} is {len(ids)} tokens, not one: {ids}")
            self._tokens[label] = ids[0]
        ids = [self._tokens[l] for l in labels]
        if len(set(ids)) != len(ids):
            raise ValueError(f"labels share a token: {labels} -> {ids}")

    def reason(self, case: dict, method: Method) -> str:
        """The reasoning trace for one case, from the letterless prompt.

        Greedy, and thinking forced ON for this call regardless of the endpoint's default -- the
        point of the stage is to obtain a trace.

        Read through /v1/completions, not /v1/chat/completions, because no reasoning parser may
        stand between the model and the trace: the template opens <think> in the PROMPT, so the
        output carries only the closing tag, and vLLM's qwen3 parser responds by discarding the
        reasoning entirely (see the module docstring). Raw output has no such opinion.
        """
        ids = self._render(reasoning_messages(case, method), thinking=True)
        closer = self._id(THINK_CLOSE)
        if closer in ids:
            # A template that closes the block even with thinking on leaves the model nowhere to
            # reason; cut back to just inside it so generation lands in the block.
            ids = ids[:ids.index(closer)]
        elif self._id(THINK_OPEN) not in ids:
            raise NoReasoningBlock(
                "this model's chat template opens no reasoning block even with "
                "enable_thinking=True, so there is nothing for --thinking to fill.")
        d = self._post("/v1/completions", {
            "model": self.served, "prompt": ids, "max_tokens": method.thinking,
            "temperature": 0.0, "stop": [THINK_CLOSE],
            # Pinned for the reason `_read` pins them: a greedy trace is otherwise whatever the
            # server's sampling defaults make it (Qwen's no-thinking set carries presence 1.5).
            "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        })
        choice = d["choices"][0]
        trace = (choice.get("text") or "").strip()
        if not trace:
            raise RuntimeError(
                "the endpoint returned no reasoning for --thinking. Either the model does not "
                "think, or max_tokens was too small to produce any; raise --thinking.")
        if choice.get("finish_reason") == "length":
            # Truncated mid-thought. Usable, but the trace is a fragment and the caller should
            # know: it is the difference between a considered answer and an interrupted one.
            print(f"warning: reasoning hit the --thinking budget of {method.thinking} tokens; "
                  f"the trace for {case.get('id', '?')} is truncated", file=sys.stderr)
        return trace

    def label_probs(self, asked: list[dict], method: Method,
                    trace: str | None = None) -> tuple[list[numpy.ndarray], int]:
        """The label probabilities of each variant, and the prompt tokens the endpoint charged.

        The whole point of the port: hand every rotation or fibered prompt over at once and let
        vLLM's scheduler batch them, rather than walking them one at a time.

        `trace` is one reasoning trace shared by every variant, so the only difference between
        them stays the letter assignment that rotations and fibers exist to average over. It is
        tokenized ONCE here rather than per variant, for the same reason.
        """
        rendered = [chat_messages(v, method) for v in asked]
        self.check_model(rendered[0][0], method)
        trace_ids = self._fragment(trace.strip()) if trace else None

        def one(r):
            return self._probs(r[0], r[1], method.temperature, trace_ids)

        if len(rendered) == 1:
            read = [one(rendered[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(rendered))) as pool:
                read = list(pool.map(one, rendered))
        return [p for p, _ in read], sum(tokens for _, tokens in read)

    def _probs(self, messages: list[dict], labels: list[str], temperature: float,
               trace_ids: list[int] | None = None) -> tuple[numpy.ndarray, int]:
        """One prompt: the softmax over its label tokens, and the tokens it cost.

        The count is returned rather than added to the endpoint, which several threads and
        several requests share.
        """
        self.check_labels(labels)
        ids = [self._tokens[l] for l in labels]
        if len(ids) > self.top_logprobs:
            raise ValueError(f"{len(ids)} labels; --top-logprobs is {self.top_logprobs}")
        if trace_ids is None:
            top, tokens = self._chat_top(messages, ids)
        else:
            top, tokens = self._spliced_top(messages, trace_ids, labels, ids)
        missing = [l for l, i in zip(labels, ids) if i not in top]
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {len(labels)} labels absent from the reply ({missing[:6]}). "
                + ("Is the endpoint running with --logprobs-mode processed_logprobs?" if self.mask
                   else f"Unmasked, only the top {self.top_logprobs} come back; a missing label "
                        f"would otherwise read as probability zero."))
        x = numpy.asarray([top[i] for i in ids], dtype=numpy.float64) / temperature
        return numpy.exp(x - numpy.logaddexp.reduce(x)), tokens

    def _read(self, ids: list[int]) -> dict:
        """The request fields that make a one-token read return the label distribution.

        Masked (the default): `allowed_token_ids` restricts the next token to the labels, and with
        --logprobs-mode processed_logprobs the log-probs are taken after the mask, so every label
        comes back and nothing else does. Unmasked (`self.mask = False`): the top
        `--top-logprobs` of the whole vocabulary, which is what a diagnostic needs -- the mask
        makes the label mass 1 by construction, so a read at the wrong position (the failure
        bench/validate_thinking.py exists to catch) would look exactly as healthy as a right one.
        Either way ids come back as "token_id:<id>", so labels are matched by id, not by string.
        """
        read = {"max_tokens": 1, "temperature": 0.0, "return_tokens_as_token_ids": True,
                # processed_logprobs puts every logits processor before the read, and vLLM fills
                # any parameter a request omits from the model's generation_config (or the
                # server's --override-generation-config). Greedy resets top-k, top-p and min-p,
                # but not the penalties: a repetition_penalty != 1 would scale the labels the
                # prompt already contains, which is not a constant and does not cancel. Pinned,
                # so the read is the masked logits whatever the server's sampling defaults.
                "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}
        if self.mask:
            read["allowed_token_ids"] = ids
        return read

    @staticmethod
    def _by_id(top: dict[str, float]) -> dict[int, float]:
        return {int(t.rpartition(":")[2]): lp for t, lp in top.items()}

    def _chat_top(self, messages: list[dict], ids: list[int]) -> tuple[dict[int, float], int]:
        """Log-probabilities at the label position by token id, via the chat endpoint.

        The no-thinking path: one round trip, which is what makes the method as fast as it
        claims.
        """
        d = self._post("/v1/chat/completions", {
            "model": self.served, "messages": messages, **self._read(ids),
            "logprobs": True, "top_logprobs": len(ids) if self.mask else self.top_logprobs,
            # Lichen's chat_prompt renders with enable_thinking=False. It is load-bearing: with
            # thinking on, the first token is <think> and every decision is garbage -- silently,
            # since a distribution still comes back. A chat template defaults it to true, so this
            # override per request is what makes the method work at all.
            "chat_template_kwargs": {"enable_thinking": False},
        })
        content = (d["choices"][0].get("logprobs") or {}).get("content") or []
        if not content:
            raise RuntimeError("no logprobs in reply; is the server built with logprobs support?")
        return (self._by_id({e["token"]: e["logprob"] for e in content[0].get("top_logprobs", [])}),
                int((d.get("usage") or {}).get("prompt_tokens") or 0))

    def _spliced_top(self, messages: list[dict], trace_ids: list[int],
                     labels: list[str], ids: list[int]) -> tuple[dict[int, float], int]:
        """Log-probabilities at the label position by token id, with the trace in the think block.

        Two round trips instead of one (render, then read), which --thinking has already paid for
        many times over in its decoding loop. /v1/completions is what makes it exact: it takes
        the spliced token ids verbatim, so the prompt the model sees is the one built here rather
        than one a chat template has re-rendered and trimmed.
        """
        prompt = self.splice_trace(self._render(messages, thinking=False), trace_ids, labels)
        d = self._post("/v1/completions", {
            "model": self.served, "prompt": prompt, **self._read(ids),
            "logprobs": len(ids) if self.mask else self.top_logprobs,
        })
        top = ((d["choices"][0].get("logprobs") or {}).get("top_logprobs") or [])
        if not top:
            raise RuntimeError("no logprobs in reply; is the server built with logprobs support?")
        return self._by_id(top[0]), int((d.get("usage") or {}).get("prompt_tokens") or 0)
