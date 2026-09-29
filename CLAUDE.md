# Working in ~/src/lichen

A **fork** of [Mushroom-Systems/lichen](https://github.com/Mushroom-Systems/lichen) — Jeff's
drop-in replacement for Jev, TypeSafe's "System One" typed-decision model. Jeff is a friend;
this started as a favour (he asked for replication before publicising) and became a fork.

**Read `README.md` and `docs/RESULTS.md` first — they are his and they are good.** This file only
records what is *ours* and what is not obvious from the code.

## What lichen actually does

One forward pass, no decoding loop. Build a prompt that ends exactly where the answer letter
goes, then read the softmax over the label tokens (`A`, `B`, …) of that single next-token
distribution. Everything else — option rotation, "fibers", `shrink`, calibration — is arithmetic
on that probability vector. `label_probabilities` raises rather than guess if a label is not
exactly one token, or if two labels share one.

## Fork layout — branches

- `main` tracks `origin/main` (upstream). Don't commit here.
- `vllm-backend` — **broken, kept only as history.** Its call-time `import llama_cpp` binds
  locally, so four `Evaluator` methods read an undefined global and every `--batch` request 500s
  with `NameError`. `--batch` is the Docker default. Superseded by `backend-split`.
- `backend-split` — six commits over upstream, pushed to `fork`. The last three answer Jeff's
  review (below): his label-mask patch (`cd3ea3a`, his authorship), and two of ours.
- `thinking` — `--thinking N` lets the model reason before the label is read. Rebased onto
  `backend-split` 2026-09-29; validated live, see the `--thinking` section below.

`origin` = upstream (read-only in practice), `fork` = github.com/martinjacobd/lichen.

## What we changed, and why

1. **`lichen/backends/`** — `llamacpp.py`, `vllm.py`, `embed.py`, with one dispatch in
   `backends/__init__.py:backend_from()`. The seam is `label_probs(variants, method)`: the whole
   set of a question's variants at once, so `--batch`'s Evaluator and vLLM's scheduler each get
   to read them together. `method.answer()` shapes the reply for both, which is what keeps
   `/v1/systemone` byte-compatible — the earlier vLLM code had its own copy of the fibered sum,
   shrink and confidence, kept in step by hand.
   **Invariant worth preserving:** `grep -rn llama_cpp lichen/` must hit only `backends/`.
2. **`llama-cpp-python` is the `llamacpp` extra**, not a hard dependency. A plain install is a
   working vLLM install; a GGUF without the extra exits with a message naming the fix. There is
   deliberately no `vllm` extra — that backend reaches its engine over HTTP, so it would be empty.
3. **`--vllm-endpoint`** serves the same method from a vLLM endpoint. Undocumented in `README.md`
   on purpose: that is Jeff's call, not ours.

### Two bugs we fixed that are HIS, and one that has a documentation consequence

- **`last_logits` aliased llama.cpp's own logits buffer** (`912ed45`, pre-existing on upstream
  `main`). The next `eval` overwrites it, and `runoff`'s unbatched branch is the one caller
  holding two at once — so **`--runoff` without `--batch` could only flatten the top two options
  to a tie**, and swap them, since `argmax` then takes the earlier key. **If the `--runoff`
  numbers in `docs/RESULTS.md` (+3 hard for E4B, −4 for 26B-A4B) were measured without `--batch`,
  they measured a tie-breaker.** Unresolved; worth asking him. This commit is self-contained and
  is the one thing here worth offering upstream regardless of the rest.
- **The vLLM path's `usage.input_tokens` was racy** (ours) — a shared counter on the backend with
  a thread pool per request. Four concurrent copies of one request reported
  `[2202, 1467, 1712, 1957]` where 1467 is correct.

## Things that will bite

- **The vLLM endpoint MUST run `--logprobs-mode processed_logprobs`** (this reversed on
  2026-09-29 — it used to be "raw, never processed"). Each read sends `allowed_token_ids` = the
  label tokens, and only a processed mode takes the log-probs after that mask. On a raw server
  the masked read refuses most wide choices (197 of 270 prompts), so the mistake is loud.
  gpum: `LOGPROBS_MODE=processed_logprobs ./gpum up qwen3.8-27b-fp8` (the knob is ours, added
  2026-09-29; empty = vLLM's default, raw).
- **processed mode puts the server's sampling defaults into the read.** vLLM fills whatever a
  request omits from `generation_config` / `--override-generation-config`; greedy resets
  top-k/top-p/min-p but not the penalties, and a `repetition_penalty != 1` would scale label
  tokens the prompt contains. `_read` and `reason()` pin all three penalties. Keep it that way.
- **The mask blinds `captured`** (label mass is 1 by construction), so a read at the wrong
  position looks healthy. Diagnostics set `Endpoint.mask = False` — `bench/diag.py`. Never
  validate a read-position change with the mask on.
- **`--max-logprobs >= options × fibers`, at most 62** (= `len(LABELS)`); vLLM's default is 20.
- **vLLM is not batch-invariant.** The same request moves label probabilities by up to ~0.03
  (Qwen3.8, ours) / 0.09 (gemma, Jeff's) — enough to flip a borderline item — and a ~500-token
  greedy trace diverges outright (39 of 47 differ between two runs). `VLLM_BATCH_INVARIANT=1`
  fixed it for Jeff (231/231 identical, +23% latency) but **cannot run Qwen3.8 on vLLM 0.29**:
  it forces Triton attention (no FP8 KV below SM89 — fixable with `KV_CACHE_DTYPE=auto`) and
  then refuses outright: "batch_invariant mode is not supported for GDN_ATTN". gpum's compose
  has the knob anyway (default 0). **What works instead: `MAX_NUM_SEQS=1`.** Read noise went to
  exactly 0 on 270 short prompts (masked == unmasked to 4 dp), for ~15% wall time on
  mask_check. Not perfect on long prompts: a full JevBench rerun gave identical predictions on
  231/231 but probs differing by <= 0.03 on 8 long_policy items — probably chunked prefill
  (MAX_NUM_BATCHED_TOKENS 8192) meeting a different prefix-cache state; unconfirmed.
- **With `MAX_NUM_SEQS=16` the Qwen preset crash-loops at the default 262K context** (KV cache
  4.48 GiB < 4.64 needed; `restart: unless-stopped` hides it as "did not reach serving"). Add
  `MAX_MODEL_LEN=32768`.
- **Guardian, Qwen3Guard and open-`<think>` templates are refused on vLLM** (`check_model`):
  their prompts exist only in `llamacpp.render`.
- **`enable_thinking=False` is load-bearing.** With thinking on, the first token is `<think>` and
  every decision is garbage — silently, because a distribution still comes back.
- **`Path("qwen3.8-27b-fp8").stem` is `"qwen3"`.** With `--vllm-endpoint`, `--model` is a
  served-model-name and must not be stemmed.
- **8192 `max-model-len` is not enough** for JevBench's hard tier under `--repeat 2`: a
  3,746-token state doubles past it and vLLM returns 400. 32768 is comfortable.
- **`--batch` is silently ignored with `--vllm-endpoint`** (vLLM batches natively), and
  `ContextOverflow → 422` is llama.cpp-only; vLLM's equivalent 400 surfaces as a 500.
- `--runoff` works on both backends now. `--recheck` and `--embedding` are llama.cpp-only.

## Testing — there is none in the repo, and that is a real gap

Byte-compat evidence lives in **`~/src/lichen-harness/`**, not here: `fakelib/` (a stub
`llama_cpp`), `drive.py` (the llama.cpp flag matrix), `drive_vllm.py`, `conc.py` (the concurrency
check), `models/` (a CPU Qwen2.5-0.5B GGUF). 11 of 12 flag sets are byte-identical to
`origin/main` on a real CPU model; the twelfth is `runoff`, which differs because it is fixed.

**Caveat that matters: the stub harness did NOT catch the logits-aliasing bug** — it allocated a
fresh array per call, so the aliasing never bit and all 12 sets "passed" before and after the
fix. Only the real library exposed it. Stub-only results prove less than they look like they do.

Recreate the real-library venv (it was not moved; venvs bake absolute paths):

    uv venv ~/src/lichen-harness/venv-real
    uv pip install --python ~/src/lichen-harness/venv-real llama-cpp-python==0.3.35 numpy jinja2

Rebuild upstream baselines for comparison:

    git worktree add ../lichen-upstream origin/main --detach

## Replication findings (2026-09-24), and what Jeff did with them

Ours are in `~/sysadmin_things/lichen-replication.md`; the shareable write-up is at
<https://claude.ai/artifact/C9AoJtaMKnxJHVQf7fS1BZ>. Headline numbers, JevBench public hard tier:

| | |
|---|---|
| gemma-4-26B-A4B, no prompt techniques | 82/111 (his own run: 83) |
| Jev 1.13.0 | 81/111 |
| Qwen3.8-27B-FP8, plain → full stack | 76 → 88 (**p = 0.004**) |
| our vLLM backend vs his llama.cpp Q8_0 | 88/111 vs 88/111, agreeing on 109 of 111 |

He took the lot in `0ff9797` — the no-tricks baseline, the exact McNemar p = 0.09 against Jev,
the cross-GPU caveat, the deployments-not-models latency note — and cites our run. So the
critique is settled; don't relitigate it.

**The one finding of ours he has not got:** the prompt techniques are worth far more on
Qwen3.8-27B than on the gemma he ships — confirmed on all 231 v1.4 items 2026-09-29, plain 190 →
image config 206 (19 gained, 3 lost, p = 0.001; hard 74 → 88), against gemma's 83 → 88 hard.

**RETRACTED 2026-09-29: "ship Qwen3.8-27B".** Measured on the full v1.4 public set with the
image config (masked read, `MAX_NUM_SEQS=1`, `bench/jevbench_v14_qwen.sh`,
`results/jevbench-v14/`), Qwen3.8-27B-FP8 scores 206/231, hard 88/111. The repo's gemma-4-26B-A4B
QAT run scores 207/231, hard 88/111 — paired, 9 items each way on hard, **p = 1.0**. The techniques
lift Qwen *up to* gemma, not past it, and gemma is the smaller, cheaper model. Against Jev 1.13.0
Qwen is +7 hard (14 vs 7 discordant, p = 0.19), not significant. The earlier "+12, p = 0.004"
was plain vs full-stack *within* Qwen, which never justified a cross-model recommendation.

## Jeff's review of `backend-split` (2026-09-29)

In `~/Downloads/backend-split-feedback.md` + `…-vllm-label-mask.patch`; gemma-4-26B-A4B on vLLM
0.30, JevBench v1.4. What we did with each point:

1. Labels outside the top-64 → 500. **Taken** (his patch, `cd3ea3a`), then verified on Qwen3.8 /
   vLLM 0.29 (`bench/mask_check.py`, `results/mask-*`): masked argmax == unmasked on every
   comparable prompt, differences within vLLM's own noise. Our additions: pinned penalties
   (`14454dc`), the unmasked diagnostic path.
2. Labels matched by string. **Taken** (same patch; ids via `return_tokens_as_token_ids`).
3. Nondeterminism / `VLLM_BATCH_INVARIANT=1`. **Documented** in `vllm.py`'s docstring.
4. `input_tokens` differs by backend. **Documented**, not changed.
5. Guardian / Qwen3Guard / LFM prompts llama.cpp-only. **Refused** on vLLM (`1e313c7`), not ported.
6. Questions in one request run serially. **Not done** — only multi-question requests benefit.

Still to ask him: whether the `--runoff` numbers in `docs/RESULTS.md` were measured without
`--batch` (see the aliasing bug above).

## Also worth knowing

- `~/sysadmin_things/gpu-manager/jev/jev_vllm.py` is the **pre-fork ancestor** of
  `lichen/backends/vllm.py` — it loads lichen's `method` with a stubbed `llama_cpp`. Now
  redundant; `dream/triage` still points at it, so it cannot just be deleted.
- Benchmark harness: `~/sysadmin_things/gpu-manager/dream/triage` drives `/v1/systemone`, and
  jevbench's `typesafe` adapter works against it unchanged. `--key-env ""` for a local endpoint.
- Upstream's apt 404 was a transient Ubuntu mirror inconsistency, since resolved. His
  `Acquire::Retries=5` cannot fix a 404; his README's archive-only stanza is the part that works.

## `--thinking`: validated live 2026-09-26 — correct now, and not worth turning on

Run against Qwen3.8-27B-FP8 on leviathan (vLLM 0.29.0) over the 50 JevBench hard cases.
Harness `bench/validate_thinking.py`, cue sweep `bench/thinking_cue_sweep.py`, data in
`results/thinking-*.json`. Three silent bugs found and fixed (`f53cdac`, `d7fc692`); the
accuracy question is answered.

**Final state of the read** (48 of 50 — `cat_dock` and `cat_mesh` have 19 labels each and lose
some to the server's top-64; see the open item below):

| | plain | thinking |
|---|---|---|
| argmax on a label | 48/48 | 48/48 |
| captured mass (mean / min) | 0.996 / 0.985 | 0.967 / 0.760 |
| confidence (mean) | 0.874 | 0.870 |
| accuracy vs gold | 45/48 (93.8%) | 45/48 (93.8%) |

Discordant pairs 2 and 2; **exact McNemar p = 1.0000**. So reasoning-before-label buys nothing
here for ~30x the latency (~5–8s a case against ~0.2s). Four discordant pairs is very little
power, so that is *no evidence of benefit*, not evidence of no benefit — and at 93.8% there are
only three errors left to fix, so a ceiling effect is the obvious confound. A harder tier or a
weaker model is where this would be worth re-asking.

**`captured` is the metric to judge any change by** — the label probability BEFORE renormalising,
i.e. what the read actually sees. Every one of the three bugs returned a confident-looking
renormalised distribution while measuring nothing; in the worst case `confidence` *rose* to 0.916
as the read degraded. `bench/validate_thinking.py` reports it per case and per type and classifies
each read `label` / `label-case` / `other`. Do not judge a thinking change by accuracy or
confidence alone.

### Still open

1. ~~The top-64 truncation~~ — **closed 2026-09-29** by the label mask (Jeff found the same
   failure independently). `cat_dock`, `cat_sdcard`, `cat_mesh` now read, argmax agreeing with
   the labels the unmasked read could see; `results/mask-qwen38-27b-hard-thinking.json`. It did
   NOT make `captured` unnecessary — it made it unmeasurable with the mask on; see "Things that
   will bite".
2. **The llamacpp half of the bridge has never been run.** `splice_trace` there takes `labels` and
   appends the same `label_constraint`, and it is by construction the same edit, but no GGUF has
   exercised it. It needs the harness venv, which does not currently exist, and a thinking GGUF
   that llama-cpp-python's bundled llama.cpp actually supports — `Mellum2-12B-A2.5B-Thinking-Q8_0`
   and the Ornith MTP file are both on `/mnt/local/models/`, but both are new architectures (the
   Mellum2 preset needs build 11176), so arch support is the thing to check first, before
   attributing any failure to the splice:

       uv venv ~/src/lichen-harness/venv-real
       uv pip install --python ~/src/lichen-harness/venv-real llama-cpp-python==0.3.35 numpy jinja2

   Check the same two things the vLLM run had to: that the trace is non-empty and is not just the
   model's one-word answer, and that the argmax at the label position is a label.
3. **`shrink` and `confidence` still have constants fit on no-thinking distributions.** Less
   urgent than it looked: with the bridge in place thinking's confidence (0.870) sits essentially
   on top of plain's (0.874), so there is no large peak-sharpening to correct for. Worth
   remeasuring only if thinking is ever adopted.

A deliberate non-goal: sampling k traces and averaging the label vectors (a principled
marginalisation over reasoning, and strictly better than majority voting since it uses the whole
distribution). `combine()` is already the right machinery. Given the p = 1.0000 above, it needs a
tier where single-trace thinking shows a benefit first, or it is just a more expensive way to get
the same answer.
