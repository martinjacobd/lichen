#!/usr/bin/env python3
"""Which post-trace bridge brings the label read back onto the labels?

The invariant lichen protects is only that the prompt ENDS where the label goes, so anything
before that point is available: text appended inside the reasoning block, and text appended
after it. This measures candidates for both against captured mass -- the label probability
BEFORE renormalising, i.e. what the read actually sees. Anything much below ~0.9 is reading
noise however confident the renormalised number looks.

The trace depends on the case, not the cue, so reason() runs ONCE per case and every candidate
reuses it. That also makes the comparison exactly paired.
"""
import argparse, collections, json, math, pathlib, sys, time

ROOT = pathlib.Path("/home/jacobm/src/lichen")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
from lichen.backends.vllm import Endpoint
from lichen.method import SYSTEM, GUARD, Method, chat_messages, confidence
import cases_hard, numpy
from diag import by_string, unmasked

ap = argparse.ArgumentParser()
ap.add_argument("endpoint")
ap.add_argument("--model", default="qwen3.8-27b-fp8")
ap.add_argument("--cases", type=int, default=0)
ap.add_argument("--thinking", type=int, default=1024)
ap.add_argument("--out")
a = ap.parse_args()

m = Method(SYSTEM + GUARD, thinking=a.thinking)
m_plain = Method(SYSTEM + GUARD, thinking=0)
ep = unmasked(Endpoint(a.endpoint, a.model, m, top_logprobs=64, workers=4, served=a.model))

def mapping(labels, keys):
    return "The labels are: " + ", ".join(f"{l} = {k}" for l, k in zip(labels, keys)) + "."

# in_block goes inside <think>…</think>; after goes between </think>\n\n and the read position.
# Round 2. Round 1 said: the reminder (C) rescues the noul cases that drift into prose, the
# mapping (D) rescues the choice cases that answer with the option's content word, and they win on
# DIFFERENT types -- so try them together. "Answer: " with a trailing space was not a semantic
# failure in round 1 but a tokenization one: the space becomes its own token, so the model emits
# " A" while the read looks for "A". A newline instead puts the label at a line start, where it
# tokenizes bare.
CANDIDATES = {
    "D_mapping":       dict(in_block=["map"],           after=""),
    "F_map_remind":    dict(in_block=["map", "remind"], after=""),
    "G_answer_nl":     dict(in_block=[],                after="Answer:\n"),
    "H_all":           dict(in_block=["map", "remind"], after="Answer:\n"),
    "J_alphabet":      dict(in_block=["alpha"],         after=""),
}
REMIND = "I must now reply with only the label, and nothing else."


def alphabet(labels):
    """Name the label alphabet without giving the mapping -- separates 'which tokens are legal'
    from 'which option each one stands for'."""
    return "I must reply with exactly one of: " + ", ".join(labels) + "."

def spliced(messages, trace, labels, keys, spec):
    ids = ep._render(messages, thinking=False)
    opener, closer = ep._id("<think>"), ep._id("</think>")
    nl, nl2 = ep._id("\n"), ep._id("\n\n")
    at = len(ids) - 1 - ids[::-1].index(opener)
    body = ep._fragment(trace.strip())
    parts = {"remind": REMIND, "map": mapping(labels, keys),
             "alpha": alphabet(labels)}
    for comp in spec["in_block"]:
        body = body + ep._fragment("\n\n" + parts[comp])
    tail = ids[at + 1:]
    tail = tail[tail.index(closer):] if closer in tail else [closer, nl2]
    out = ids[:at + 1] + [nl] + body + [nl] + tail
    if spec["after"]:
        out = out + ep._fragment(spec["after"])
    return out

def read(top, labels):
    missing = [l for l in labels if l not in top]
    if missing:
        return None, 0.0, missing
    lp = numpy.asarray([top[l] for l in labels], dtype=numpy.float64)
    cap = float(sum(math.exp(v) for v in lp))
    return numpy.exp(lp - numpy.logaddexp.reduce(lp)), cap, []

def top_of(ids):
    d = ep._post("/v1/completions", {"model": a.model, "prompt": ids, "max_tokens": 1,
                                    "temperature": 0.0, "logprobs": ep.top_logprobs})
    return dict(((d["choices"][0].get("logprobs") or {}).get("top_logprobs") or [{}])[0])

CASES = cases_hard.CASES[:a.cases] if a.cases else cases_hard.CASES
rows = []
for case in CASES:
    msgs, labels, keys = chat_messages(case, m)
    t0 = time.time()
    try:
        trace = ep.reason(case, m)
    except Exception as exc:
        print(f"  {case['id']:<22} reason FAILED: {exc}", flush=True); continue
    ep.check_labels(labels)
    top_p = by_string(ep, ep._chat_top(msgs, [ep._tokens[l] for l in labels])[0])
    pp, cap_p, miss_p = read(top_p, labels)
    row = dict(id=case["id"], type=case["question"]["type"], labels=labels, keys=keys,
               trace_words=len(trace.split()), captured_plain=round(cap_p, 4),
               pick_plain=keys[int(pp.argmax())] if pp is not None else None,
               conf_plain=round(confidence(pp), 4) if pp is not None else None, cand={})
    for name, spec in CANDIDATES.items():
        top = top_of(spliced(msgs, trace, labels, keys, spec))
        pt, cap, miss = read(top, labels)
        am = max(top, key=top.get)
        row["cand"][name] = dict(
            captured=round(cap, 4), argmax=am, missing=miss,
            landing=("label" if am in labels else
                     "label-case" if am.strip().lower() in {l.strip().lower() for l in labels}
                     else "other"),
            pick=keys[int(pt.argmax())] if pt is not None else None,
            conf=round(confidence(pt), 4) if pt is not None else None)
    row["seconds"] = round(time.time() - t0, 1)
    rows.append(row)
    caps = "  ".join(f"{n.split('_')[0]}={row['cand'][n]['captured']:.3f}" for n in CANDIDATES)
    print(f"  {row['id']:<22}{row['type']:<7} plain={cap_p:.3f} | {caps}", flush=True)

print(f"\n== {len(rows)} cases\n")
hdr = f"{'candidate':<16}{'captured mean':>14}{'min':>8}{'>=0.9':>7}{'on-label':>10}{'agrees w/ plain':>17}{'conf mean':>11}"
print(hdr); print("-" * len(hdr))
print(f"{'(plain, no think)':<16}"
      f"{sum(r['captured_plain'] for r in rows)/len(rows):>14.3f}"
      f"{min(r['captured_plain'] for r in rows):>8.3f}"
      f"{sum(1 for r in rows if r['captured_plain'] >= 0.9):>7}"
      f"{len(rows):>10}{'-':>17}"
      f"{sum(r['conf_plain'] for r in rows)/len(rows):>11.3f}")
for name in CANDIDATES:
    cs = [r["cand"][name] for r in rows]
    caps = [c["captured"] for c in cs]
    onlab = sum(1 for c in cs if c["landing"] == "label")
    agree = sum(1 for r in rows if r["cand"][name]["pick"] == r["pick_plain"])
    confs = [c["conf"] for c in cs if c["conf"] is not None]
    print(f"{name:<16}{sum(caps)/len(caps):>14.3f}{min(caps):>8.3f}"
          f"{sum(1 for c in caps if c >= 0.9):>7}{onlab:>10}{agree:>17}"
          f"{sum(confs)/max(1,len(confs)):>11.3f}")

print("\nby question type (captured mean):")
by = collections.defaultdict(list)
for r in rows: by[r["type"]].append(r)
print(f"  {'type':<8}{'n':>3}{'plain':>8}" + "".join(f"{n.split('_')[0]:>8}" for n in CANDIDATES))
for t, rs in sorted(by.items()):
    line = f"  {t:<8}{len(rs):>3}{sum(r['captured_plain'] for r in rs)/len(rs):>8.3f}"
    for n in CANDIDATES:
        line += f"{sum(r['cand'][n]['captured'] for r in rs)/len(rs):>8.3f}"
    print(line)
if a.out:
    pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {a.out}")
