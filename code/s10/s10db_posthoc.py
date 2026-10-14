#!/usr/bin/env python3
"""s10db POST HOC (paper 4). Written after the frozen s10db analysis had been run and read. Repeats the whole-pool contrasts C1-C3 and the descriptive D_b2x - D_bV in the strata of examination type
(all rows / screening only / diagnostic only) that the post hoc part of s10an introduced (prereg/s10an_posthoc_ru.md). Saved scores only; aggregate tables only.
usage: PYTHONPATH=$PWD:$PWD/analysis_embed ~/envs/cbmammo/bin/python analysis_embed/s10db_posthoc.py --out runs_s10db/analysis_posthoc"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s10_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
X = Ctx(boot=a.boot); EM = {}
for l in open("manifests/embed.jsonl", encoding="utf-8"):
    r = json.loads(l); EM[r["sample_id"]] = r
et = np.array([((EM[s].get("embed_meta") or {}).get("exam_type") or "missing") for s in X.te])
ST = {"all rows": np.ones(X.n, bool), "screening only": et == "screening", "diagnostic only": et == "diagnostic"}
rows = []; rc = []
for sn, m in ST.items():
    T = {arm: X.arm_auc(arm, "f1", SEEDS, m) for arm in ("V", "V2x", "D", "D_bV", "D_b2x")}
    for arm in T:
        for f in FIND:
            v = mean_boot(T[arm][f]); lo, hi = ci(v); rows.append(dict(stratum=sn, arm=arm, finding=f, auc=float(v[0]), lo=lo, hi=hi))
    for name, x, y, rule in (("C1 V - D_bV", "V", "D_bV", verdict_gap), ("C2 D_bV - D", "D_bV", "D", verdict_gain), ("C3 V2x - D_b2x", "V2x", "D_b2x", verdict_gap), ("desc K3 V - D", "V", "D", verdict_gap), ("desc D_b2x - D_bV", "D_b2x", "D_bV", verdict_gain)):
        for f in FIND:
            d = mean_boot(T[x][f]) - mean_boot(T[y][f]); lo, hi = ci(d); rc.append(dict(stratum=sn, contrast=name, finding=f, delta=float(d[0]), lo=lo, hi=hi, verdict=rule(d)))
pd.DataFrame(rows).to_csv(os.path.join(a.out, "dbph_auc_by_stratum.csv"), index=False); pd.DataFrame(rc).to_csv(os.path.join(a.out, "dbph_contrasts_by_stratum.csv"), index=False); print("done")
