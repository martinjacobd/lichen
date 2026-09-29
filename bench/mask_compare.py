#!/usr/bin/env python3
"""Summarise mask_check.py output: masked vs unmasked, against the unmasked read's own noise.

    mask_compare.py results/mask-*-processed.json [results/mask-*-raw.json]
"""
import json, sys
import numpy as np


def load(p):
    return {(r["case"], r["method"], r["variant"]): r for r in json.load(open(p))["rows"]}


def renorm(top, ids):
    """The unmasked read over the labels, or None if one fell outside the top-N."""
    if not all(str(i) in top for i in ids):
        return None
    x = np.array([top[str(i)] for i in ids])
    return np.exp(x - np.logaddexp.reduce(x))


def dmax(a, b):
    return None if a is None or b is None else float(np.abs(a - b).max())


pro = load(sys.argv[1])
raw = load(sys.argv[2]) if len(sys.argv) > 2 else None
stats = {"noise (unmasked twice)": [], "masked vs unmasked": []}
if raw:
    stats["unmasked, raw vs processed"] = []
agree = comparable = 0
blind = []
for k, p in pro.items():
    u = [renorm(t, p["ids"]) for t in p["unmasked"]]
    m = np.array(p["masked"])
    stats["noise (unmasked twice)"].append(dmax(u[0], u[1]))
    stats["masked vs unmasked"].append(dmax(m, u[0]))
    if raw:
        stats["unmasked, raw vs processed"].append(dmax(renorm(raw[k]["unmasked"][0], p["ids"]), u[0]))
    if u[0] is not None:
        comparable += 1
        agree += int(u[0].argmax() == m.argmax())
    else:
        top = p["unmasked"][0]
        seen = [(top[str(i)], j) for j, i in enumerate(p["ids"]) if str(i) in top]
        lost = [j for j, i in enumerate(p["ids"]) if str(i) not in top]
        blind.append((k, len(p["ids"]), len(lost), max(seen)[1] == m.argmax(), float(m[lost].sum())))
print(f"{len(pro)} prompts, {comparable} with every label in the unmasked top-N")
for name, v in stats.items():
    v = [x for x in v if x is not None]
    print(f"  {name:<28} max |dp| {max(v):.4f}  p95 {np.percentile(v, 95):.4f}  median {np.median(v):.5f}")
print(f"  argmax masked == unmasked   {agree}/{comparable}")
for k, n, lost, same, mass in blind:
    print(f"  only the mask reads {k}: {lost} of {n} labels outside the top-N; argmax agrees: {same}; "
          f"masked mass on them {mass:.1e}")
if raw:
    refused = sum("masked_error" in r for r in raw.values())
    print(f"  raw-mode server: the masked read refused {refused}/{len(raw)}")
