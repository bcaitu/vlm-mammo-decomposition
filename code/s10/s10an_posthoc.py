#!/usr/bin/env python3
"""s10an POST HOC (paper 4). Written AFTER the frozen blocks B1-B5 had been run and read: B5 showed that the test negatives come almost only from screening examinations while 30-49% of the positives come from diagnostic examinations, and B2 showed that the contrasts depend on which examinations are kept.
This script is therefore post hoc and exploratory; it is not part of the frozen protocol (prereg/s10an_posthoc_ru.md). Saved scores only, no training; aggregate tables only.
PH1 AUC of the examination-type indicator itself (diagnostic = 1) for each finding; PH2 AUC per arm (whole pool) in all / screening-only / diagnostic-only / earliest-examination rows; PH3 contrasts in the same strata; PH4 composition of the earliest-examination subset by examination type.
usage: PYTHONPATH=$PWD:$PWD/analysis_embed ~/envs/cbmammo/bin/python analysis_embed/s10an_posthoc.py --out runs_s10an/analysis_posthoc"""
import argparse, csv, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s10_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
X = Ctx(boot=a.boot)
EM = {}
for l in open("manifests/embed.jsonl", encoding="utf-8"):
    r = json.loads(l); EM[r["sample_id"]] = r
et = np.array([((EM[s].get("embed_meta") or {}).get("exam_type") or "missing") for s in X.te])
CLIN = {}
with open("/raid/datasets/vlm_mammogram_data/data/EMBED/tables/EMBED_OpenData_clinical_reduced.csv", encoding="utf-8", newline="") as fh:
    for r in csv.DictReader(fh):
        k = (r["empi_anon"], r["acc_anon"])
        if k not in CLIN: CLIN[k] = r["study_date_anon"]
        elif not CLIN[k] and r["study_date_anon"]: CLIN[k] = r["study_date_anon"]
def parse(s): _, p, st, side = s.split(":"); return p, st
pr = [parse(s) for s in X.te]; by_p = {}
for p, s in pr: by_p.setdefault(p, set()).add(s)
first = {p: sorted(ex, key=lambda s: ((CLIN.get((p, s)) or "9999-99-99"), s))[0] for p, ex in by_p.items()}      # same rule as B2
earliest = np.array([first[p] == s for p, s in pr])
ST = {"all rows": np.ones(X.n, bool), "screening only": et == "screening", "diagnostic only": et == "diagnostic", "earliest examination": earliest}
ARMS = ["V", "VP", "VC", "VCP", "V2x", "V_ctl", "D", "A_probe", "A_tf"]
# ---- PH1: exam-type indicator as a score (exact AUC with ties counted 1/2: 0.5 + (p1 - p0)/2)
rows = []; ind = (et == "diagnostic").astype(np.float32)
for f in FIND:
    k = X.Y[f] >= 0; pos = (X.Y[f] == 1) & k; neg = (X.Y[f] == 0) & k
    p1 = (X.Wall[:, pos] * ind[pos]).sum(1) / X.Wall[:, pos].sum(1); p0 = (X.Wall[:, neg] * ind[neg]).sum(1) / X.Wall[:, neg].sum(1); d = 0.5 + (p1 - p0) / 2
    lo, hi = ci(d); rows.append(dict(finding=f, share_diagnostic_among_positives=float(p1[0]), share_diagnostic_among_negatives=float(p0[0]), auc_exam_type_indicator=float(d[0]), lo=lo, hi=hi))
pd.DataFrame(rows).to_csv(os.path.join(a.out, "ph1_examtype_indicator.csv"), index=False)
# ---- PH2, PH3
CON = [("K1 VP - A_tf", "VP", "A_tf", lambda d: cls(d, 0.03)), ("K1b VP - A_probe", "VP", "A_probe", lambda d: cls(d, 0.03)), ("K2 V - VP", "V", "VP", lambda d: cls(d, 0.03)), ("K3 V - D", "V", "D", lambda d: cls(d, 0.05)),
       ("S1 VC - V", "VC", "V", verdict_gain), ("S2 VC - D", "VC", "D", verdict_gap), ("S3 V2x - V", "V2x", "V", verdict_gain), ("S4 VCP - VP", "VCP", "VP", verdict_gain), ("S5 VC - VCP", "VC", "VCP", verdict_gap), ("desc V - V_ctl", "V", "V_ctl", lambda d: cls(d, 0.03))]
r2 = []; r3 = []
for sn, m in ST.items():
    T = {arm: X.arm_auc(arm, "f1", SEEDS, m) for arm in ARMS}
    for arm in ARMS:
        for f in FIND:
            if f not in T[arm]: continue
            v = mean_boot(T[arm][f]); lo, hi = ci(v); k = m & (X.Y[f] >= 0)
            r2.append(dict(stratum=sn, arm=arm, finding=f, auc=float(v[0]), lo=lo, hi=hi, n_pos=int((X.Y[f][k] == 1).sum()), n_neg=int((X.Y[f][k] == 0).sum())))
    for name, a_, b_, rule in CON:
        for f in FIND:
            if f in T[a_] and f in T[b_]:
                d = mean_boot(T[a_][f]) - mean_boot(T[b_][f]); lo, hi = ci(d); r3.append(dict(stratum=sn, contrast=name, finding=f, delta=float(d[0]), lo=lo, hi=hi, verdict=rule(d)))
    print("stratum done", sn, flush=True)
pd.DataFrame(r2).to_csv(os.path.join(a.out, "ph2_auc_by_stratum.csv"), index=False); pd.DataFrame(r3).to_csv(os.path.join(a.out, "ph3_contrasts_by_stratum.csv"), index=False)
# ---- PH4
r4 = []
for fl, mk in (("earliest examination", earliest), ("later examinations", ~earliest)):
    for e in ("screening", "diagnostic"):
        for f in FIND:
            k = mk & (et == e) & (X.Y[f] >= 0); r4.append(dict(subset=fl, exam_type=e, finding=f, breasts=int(k.sum()), present=int((X.Y[f][k] == 1).sum()), absent=int((X.Y[f][k] == 0).sum())))
pd.DataFrame(r4).to_csv(os.path.join(a.out, "ph4_earliest_by_examtype.csv"), index=False); print("done")
