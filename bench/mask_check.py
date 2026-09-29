#!/usr/bin/env python3
"""Does the masked label read (allowed_token_ids) read the same distribution as the unmasked one?

The vLLM backend reads each prompt through a mask on the label tokens, with the endpoint on
--logprobs-mode processed_logprobs. That can only be right if the processed log-probs are the
masked logits less a constant -- so this reads every prompt both ways, plus the unmasked read a
second time, whose difference is vLLM's own run-to-run noise and the yardstick for the rest.

    mask_check.py ENDPOINT MODE --out results/mask-....json          # plain: plain, permute, fibers 3
    mask_check.py ENDPOINT MODE --thinking 1024 --out ...            # the spliced read, one trace per case

MODE is only a tag for the file (raw_logprobs / processed_logprobs): the server decides it. On a
raw-mode server the masked read should REFUSE most wide choices -- that is the check that a
missing --logprobs-mode fails loudly.
"""
import argparse, json, pathlib, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))

from lichen.backends.vllm import Endpoint
from lichen.method import SYSTEM, GUARD, Method, chat_messages, variants
import cases_hard

ap = argparse.ArgumentParser()
ap.add_argument("endpoint")
ap.add_argument("mode")
ap.add_argument("--model", default="qwen3.8-27b-fp8")
ap.add_argument("--thinking", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

if a.thinking:
    METHODS = {"thinking": Method(SYSTEM + GUARD, thinking=a.thinking)}
else:
    METHODS = {"plain": Method(SYSTEM + GUARD), "permute": Method(SYSTEM + GUARD, permute=True),
               "fibers3": Method(SYSTEM + GUARD, fibers=3)}
first = next(iter(METHODS.values()))
masked = Endpoint(a.endpoint, a.model, first, top_logprobs=64, served=a.model)
plain = Endpoint(a.endpoint, a.model, first, top_logprobs=64, served=a.model)
plain.mask = False


def top(ep, messages, labels, trace_ids):
    ep.check_labels(labels)
    ids = [ep._tokens[l] for l in labels]
    if trace_ids is None:
        return ep._chat_top(messages, ids)[0]
    return ep._spliced_top(messages, trace_ids, labels, ids)[0]


def run(job):
    case, mname, vi, messages, labels, trace_ids = job
    masked.check_labels(labels)
    row = dict(case=case, method=mname, variant=vi, labels=labels,
               ids=[masked._tokens[l] for l in labels],
               unmasked=[top(plain, messages, labels, trace_ids) for _ in range(2)])
    try:
        row["masked"] = masked._probs(messages, labels, 1.0, trace_ids)[0].tolist()
    except Exception as exc:
        row["masked_error"] = str(exc)[:300]
    return row


def jobs_for(case):
    trace_ids = None
    if a.thinking:
        trace_ids = masked._fragment(masked.reason(case, METHODS["thinking"]).strip())
    out = []
    for mname, m in METHODS.items():
        if mname == "fibers3" and case["question"]["type"] != "choice":
            continue
        for vi, v in enumerate(variants(case, m)):
            messages, labels, _ = chat_messages(v, m)
            out.append((case["id"], mname, vi, messages, labels, trace_ids))
    return out


with ThreadPoolExecutor(8) as pool:
    jobs = [j for js in pool.map(jobs_for, cases_hard.CASES) for j in js]
    rows = list(pool.map(run, jobs))
pathlib.Path(a.out).write_text(json.dumps({"mode": a.mode, "model": a.model, "thinking": a.thinking,
                                           "rows": rows}))
print(f"{a.mode}: {len(rows)} prompts, {sum('masked' in r for r in rows)} read masked, "
      f"{sum('masked_error' in r for r in rows)} refused")
