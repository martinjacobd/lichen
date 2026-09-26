#!/usr/bin/env python3
"""Validate lichen's --thinking against a live endpoint, and dump paired distributions.

Step 1 of CLAUDE.md's list: is the trace real, and does the label read land where it should?
Both failures are silent -- a plausible distribution comes back either way -- so the POSITION is
checked over the unrestricted top-k rather than trusted via the label softmax.

Three outcomes are distinguished, because they mean very different things:
  label        the argmax is one of the labels lichen reads. Healthy.
  label-case   the argmax is another CASING/spacing of a label ('yes' for 'Yes'). The read still
               points the right way, but some of the distribution's mass sits on tokens lichen
               never looks at, so the calibration moves even though the answer does not.
  other        the argmax is something else entirely (whitespace, prose). Broken.

`captured` is the metric that matters for step 2: the probability mass the read actually sees,
BEFORE renormalising over the labels. A read that captures 0.95 plain and 0.55 with thinking is
measuring a different object in the two conditions, whatever the renormalised numbers say.
"""
import argparse, collections, json, math, pathlib, sys, time
from math import comb

ROOT = pathlib.Path("/home/jacobm/src/lichen")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))

from lichen.backends.vllm import Endpoint
from lichen.method import SYSTEM, GUARD, Method, chat_messages, confidence
import cases_hard, numpy

ap = argparse.ArgumentParser()
ap.add_argument("endpoint")
ap.add_argument("--model", default="qwen3.8-27b-fp8")
ap.add_argument("--cases", type=int, default=0, help="0 = all")
ap.add_argument("--thinking", type=int, default=1024)
ap.add_argument("--out")
a = ap.parse_args()

m_plain = Method(SYSTEM + GUARD, thinking=0)
m_think = Method(SYSTEM + GUARD, thinking=a.thinking)
ep = Endpoint(a.endpoint, a.model, m_think, top_logprobs=64, workers=4, served=a.model)

def gold_key(case):
    """The case's gold answer as the KEY the read reports, so the two are comparable."""
    g = case["gold"]
    if case["question"]["type"] == "noul":
        return "yes" if g else "no"
    return str(g)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on the discordant pairs: b plain-only right, c thinking-only right.

    Exact rather than chi-squared because the discordant counts here are small, which is exactly
    where the asymptotic test misbehaves. The items are identical in the two conditions, so the
    paired test is the one with the power; comparing two proportions would throw most of it away.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def classify(argmax, labels):
    if argmax in labels:
        return "label"
    if argmax.strip().lower() in {l.strip().lower() for l in labels}:
        return "label-case"
    return "other"

def read(top, labels):
    """Renormalised label probabilities, and the raw mass the read captured."""
    lp = numpy.asarray([top[l] for l in labels], dtype=numpy.float64)
    captured = float(sum(math.exp(v) for v in lp))
    return numpy.exp(lp - numpy.logaddexp.reduce(lp)), captured

CASES = cases_hard.CASES[:a.cases] if a.cases else cases_hard.CASES
rows = []
for case in CASES:
    t0 = time.time()
    msgs, labels, keys = chat_messages(case, m_think)
    try:
        trace = ep.reason(case, m_think)
    except Exception as exc:
        print(f"  {case['id']:<22} reason FAILED: {exc}", flush=True); continue
    top_p, _ = ep._chat_top(msgs)
    top_t, _ = ep._spliced_top(msgs, ep._fragment(trace.strip()), labels)
    miss = [l for l in labels if l not in top_p or l not in top_t]
    if miss:
        print(f"  {case['id']:<22} labels missing from top-64: {miss}", flush=True); continue
    pp, cap_p = read(top_p, labels)
    pt, cap_t = read(top_t, labels)
    am_p = max(top_p, key=top_p.get); am_t = max(top_t, key=top_t.get)
    gk = gold_key(case)
    rows.append(dict(
        id=case["id"], type=case["question"]["type"], labels=labels, keys=keys,
        gold=gk, right_plain=keys[int(pp.argmax())] == gk,
        right_think=keys[int(pt.argmax())] == gk, trace_words=len(trace.split()),
        argmax_plain=am_p, argmax_think=am_t,
        land_plain=classify(am_p, labels), land_think=classify(am_t, labels),
        captured_plain=round(cap_p, 4), captured_think=round(cap_t, 4),
        plain={k: round(float(v), 6) for k, v in zip(keys, pp)},
        think={k: round(float(v), 6) for k, v in zip(keys, pt)},
        pick_plain=keys[int(pp.argmax())], pick_think=keys[int(pt.argmax())],
        conf_plain=round(confidence(pp), 4), conf_think=round(confidence(pt), 4),
        seconds=round(time.time() - t0, 1)))
    r = rows[-1]
    mark = {(True, True): "both", (True, False): "plain only", (False, True): "think only",
            (False, False): "neither"}[(r["right_plain"], r["right_think"])]
    print(f"  {r['id']:<22} {r['trace_words']:>4}w  "
          f"plain {am_p!r:<7}{r['land_plain']:<11} cap {cap_p:.3f}  |  "
          f"think {am_t!r:<7}{r['land_think']:<11} cap {cap_t:.3f}  "
          f"right: {mark}", flush=True)

n = len(rows)
print(f"\n== {n} case(s) read")
for cond in ("plain", "think"):
    tally = {}
    for r in rows:
        tally[r[f"land_{cond}"]] = tally.get(f"{r[f'land_{cond}']}", 0) + 1 if False else tally.get(r[f"land_{cond}"], 0) + 1
    caps = [r[f"captured_{cond}"] for r in rows]
    confs = [r[f"conf_{cond}"] for r in rows]
    print(f"   {cond:<6} landing {tally}   captured mean {sum(caps)/max(1,n):.3f} "
          f"min {min(caps, default=0):.3f}   confidence mean {sum(confs)/max(1,n):.3f}")
agree = sum(1 for r in rows if r["pick_plain"] == r["pick_think"])
print(f"   the two conditions pick the same answer on {agree}/{n}")

ok_p = sum(r["right_plain"] for r in rows)
ok_t = sum(r["right_think"] for r in rows)
b = sum(1 for r in rows if r["right_plain"] and not r["right_think"])
c = sum(1 for r in rows if r["right_think"] and not r["right_plain"])
print(f"\n== accuracy against gold")
print(f"   plain     {ok_p}/{n}  ({100 * ok_p / n:.1f}%)")
print(f"   thinking  {ok_t}/{n}  ({100 * ok_t / n:.1f}%)")
print(f"   discordant pairs: plain-only right {b}, thinking-only right {c}")
print(f"   exact McNemar p = {mcnemar_exact(b, c):.4f}"
      f"   <- the paired test; the items are identical in both conditions")
low = [r["id"] for r in rows if r["captured_think"] < 0.9]
print(f"   cases whose thinking read captured < 0.9: {len(low)}"
      + (f" {low}" if low else ""))
if a.out:
    pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"   wrote {a.out}")
