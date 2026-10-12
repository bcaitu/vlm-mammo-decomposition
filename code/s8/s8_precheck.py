#!/usr/bin/env python3
"""s8: pre-analysis integrity check (NO metrics are computed): every arm and seed present, files loadable, finite scores, row coverage of the test list, equal coverage of VC/V2x and V.
usage (cwd ~/cbmammo, PYTHONPATH as for the analysis): python3 analysis_embed/s8_precheck.py ; exit code 0 = pass"""
import json, os, sys
import numpy as np, torch
te = json.load(open("runs_lc/sets/test_embed.json")); tset = set(te); ok = True; cov = {}
def chk(cond, msg):
    global ok
    print(("OK   " if cond else "FAIL ") + msg); ok = ok and bool(cond)
SPEC = [("V", "runs_lc/V/f1_s{s}/ext.npz", "ext"), ("VC", "runs_s8/VC/f1_s{s}/ext.npz", "ext"), ("V2x", "runs_s8/V2x/f1_s{s}/ext.npz", "ext"), ("VP", "runs_lc/V/f1_s{s}/probe.npz", "probe"),
        ("VCP", "runs_s8/VC/f1_s{s}/probe.npz", "probe"), ("A_tf", "runs_lc/A/tf_f1_s{s}.npz", "tf"), ("D", "runs_lc/D/f1_s{s}/dump_test_embed.pt", "d")]
for arm, tmpl, kind in SPEC:
    for s in (0, 1, 2):
        p = tmpl.format(s=s)
        if not os.path.exists(p): chk(False, f"{arm} s{s}: missing {p}"); continue
        try:
            if kind == "d":
                d = torch.load(p, map_location="cpu"); ids = list(d["sample_id"]); fin = all(bool(torch.isfinite(d["p_" + f]).all()) for f in ("mass", "calc", "asymmetry"))
            else:
                z = np.load(p, allow_pickle=True)
                if kind == "ext": ids = [str(x) for x in z["sample_id"]]; lp = z["logp_no"]; fin = bool(np.isfinite(lp).all()) and lp.ndim == 2 and lp.shape[1] == 4 and lp.shape[0] == len(ids)
                elif kind == "probe": ids = [str(x) for x in z["test_sid"]]; fin = all(bool(np.isfinite(z["best_" + f]).all()) and len(z["best_" + f]) == len(ids) for f in ("mass", "calc", "asymmetry"))
                else: ids = [str(x) for x in z["test_sid"]]; fin = all(bool(np.isfinite(z["score_" + f]).all()) and len(z["score_" + f]) == len(ids) for f in ("mass", "calc", "asymmetry"))
            c = len(tset & set(ids)); cov[(arm, s)] = c; chk(c > 0 and fin, f"{arm} s{s}: rows {len(ids)}, test ids covered {c}/{len(te)}, finite and consistent {fin}")
        except Exception as e: chk(False, f"{arm} s{s}: load error {repr(e)[:100]}")
chk(os.path.exists("runs_lc/A/probe_f1_s0.npz"), "A_probe present")
for s in (0, 1, 2):
    for arm in ("VC", "V2x"):
        if (arm, s) in cov and ("V", s) in cov: chk(cov[(arm, s)] == cov[("V", s)], f"{arm} s{s} covers the same number of test rows as V s{s} ({cov[(arm, s)]} vs {cov[('V', s)]})")
print("PRECHECK", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
