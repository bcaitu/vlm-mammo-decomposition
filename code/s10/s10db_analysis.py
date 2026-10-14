#!/usr/bin/env python3
"""s10db analysis (paper 4): D trained for the number of updates of the VLM arm (D_bV; D_b2x at the whole pool for V2x) against V, V2x and the original D.
AUC per finding on the EMBED test set with a known label; 2,000 patient-level bootstrap resamples (seed 20261007, identical to s6lc and s8); interval = percentile interval of the seed mean.
Confirmatory contrasts at the whole pool (calcifications and asymmetry; mass descriptive): C1 = V - D_bV (gap rule, margin 0.03), C2 = D_bV - D (gain rule, margin 0.03), C3 = V2x - D_b2x (gap rule). Reading table: prereg/s10db_prereg_ru.md section 6.
usage: PYTHONPATH=$PWD:$PWD/analysis_embed ~/envs/cbmammo/bin/python analysis_embed/s10db_analysis.py --out runs_s10db/analysis [--boot 2000] [--check-only]"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s10_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--check-only", action="store_true"); ap.add_argument("--dry-run", action="store_true", help="code test on stand-in files (small bootstrap, reproduction check not enforced); outputs are not read and are deleted"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); X = Ctx(boot=a.boot)
A = {}; rows = []
def load(arm, sz):
    r = X.arm_auc(arm, sz)
    for f, v in r.items(): A[(arm, f, sz)] = v
    return r
for arm in ("V", "D"):
    for sz in SZ: load(arm, sz)
load("V2x", "f1")
# ---- reproduction check against the frozen s6lc analysis (K3 V - D at every size) and s8 (V2x - D at the whole pool)
ref = pd.read_csv("runs_lc/analysis/lc_contrasts.csv"); err = []
for sz in SZ:
    for f in FIND:
        r = ref[(ref["contrast"] == "K3 V - D") & (ref["finding"] == f) & (ref["size"] == sz)]
        d = mean_boot(A[("V", f, sz)]) - mean_boot(A[("D", f, sz)])
        if len(r) == 1: err.append(max(abs(float(r["delta"].iloc[0]) - d[0]), abs(float(r["lo"].iloc[0]) - ci(d)[0]), abs(float(r["hi"].iloc[0]) - ci(d)[1])))
print("reproduction check: max abs error", max(err), "over", len(err), "contrasts"); open(os.path.join(a.out, "s10db_check.txt"), "w").write(f"max_abs_error {max(err)} n {len(err)}\n")
if (len(err) != 18 or max(err) > 1e-5) and not a.dry_run: print("REPRODUCTION FAILED"); sys.exit(3)
if a.check_only: sys.exit(0)
for sz in SZ: load("D_bV", sz)
load("D_b2x", "f1")
need = [("D_bV", s) for s in SZ] + [("D_b2x", "f1")]
for arm, sz in need:
    for f in FIND:
        if (arm, f, sz) not in A or len(A[(arm, f, sz)]) != 3: print("MISSING or incomplete", arm, f, sz); sys.exit(4)
# ---- tables
for (arm, f, sz), V in A.items():
    for sd in range(V.shape[0]): rows.append(dict(arm=arm, finding=f, size=sz, seed_index=sd, auc=float(V[sd, 0])))
pd.DataFrame(rows).to_csv(os.path.join(a.out, "s10db_auc_by_run.csv"), index=False)
cur = []
for (arm, f, sz), V in A.items():
    m = mean_boot(V); cur.append(dict(arm=arm, finding=f, size=sz, n_seeds=len(V), auc=float(m[0]), lo=ci(m)[0], hi=ci(m)[1], sd_seeds=float(V[:, 0].std(ddof=1)) if len(V) > 1 else float("nan")))
pd.DataFrame(cur).to_csv(os.path.join(a.out, "s10db_curve.csv"), index=False)
def D_(x, y, f, sz): return mean_boot(A[(x, f, sz)]) - mean_boot(A[(y, f, sz)])
con = []
def add(name, f, d, rule, kind, extra=""):
    lo, hi = ci(d); con.append(dict(contrast=name, finding=f, kind=kind, delta=float(d[0]), lo=lo, hi=hi, verdict=rule(d), cls003=cls(d, 0.03), note=extra))
for f in FIND:
    kind = "confirmatory" if f != "mass" else "descriptive"
    add("C1 V - D_bV (gap at matched updates, whole pool)", f, D_("V", "D_bV", f, "f1"), verdict_gap, kind)
    add("C2 D_bV - D (cost of the update budget of V to D, whole pool)", f, D_("D_bV", "D", f, "f1"), verdict_gain, kind)
    add("C3 V2x - D_b2x (gap at the updates of V2x, whole pool)", f, D_("V2x", "D_b2x", f, "f1"), verdict_gap, kind)
    add("desc K3 V - D (original, whole pool)", f, D_("V", "D", f, "f1"), verdict_gap, "descriptive")
    add("desc D_b2x - D_bV (value of doubling D's updates)", f, D_("D_b2x", "D_bV", f, "f1"), verdict_gain, "descriptive")
    add("desc D_b2x - D", f, D_("D_b2x", "D", f, "f1"), verdict_gain, "descriptive")
    for sz in SZ:
        add(f"desc V - D_bV at {sz}", f, D_("V", "D_bV", f, sz), verdict_gap, "descriptive")
        add(f"desc D_bV - D at {sz}", f, D_("D_bV", "D", f, sz), verdict_gain, "descriptive")
    for arm in ("V", "D_bV", "D"):
        sl = ((mean_boot(A[(arm, f, "f1")]) - mean_boot(A[(arm, f, "f2")])) + (mean_boot(A[(arm, f, "f2")]) - mean_boot(A[(arm, f, "f4")]))) / 2
        lo, hi = ci(sl); con.append(dict(contrast=f"desc gain per doubling over the last two doublings, {arm}", finding=f, kind="descriptive", delta=float(sl[0]), lo=lo, hi=hi, verdict="", cls003="", note="(AUC f1 - AUC f4)/2"))
C = pd.DataFrame(con); C.to_csv(os.path.join(a.out, "s10db_contrasts.csv"), index=False)
def reading(c1, c2):
    if c1.startswith("not below") and c2.startswith("reduces"): return "the gap is explained by the longer training of D: at the updates of V, D reaches V"
    if c1.startswith("below") and c2.startswith("equivalent"): return "the gap persists at matched updates and is not explained by the longer training of D"
    if c1.startswith("below") and c2.startswith("reduces"): return "the gap shrinks but persists at matched updates"
    return "unclear or other: described as is, no conclusion from the table"
rd = []
for f in FIND:
    c1 = C[(C["finding"] == f) & C["contrast"].str.startswith("C1")].iloc[0]; c2 = C[(C["finding"] == f) & C["contrast"].str.startswith("C2")].iloc[0]; c3 = C[(C["finding"] == f) & C["contrast"].str.startswith("C3")].iloc[0]
    k3 = C[(C["finding"] == f) & C["contrast"].str.startswith("desc K3")].iloc[0]
    rd.append(dict(finding=f, C1=c1["verdict"], C2=c2["verdict"], C3=c3["verdict"], reading_C1_C2=reading(c1["verdict"], c2["verdict"]), share_of_K3_due_to_D_training_length=float(c2["delta"] / k3["delta"]) if k3["delta"] != 0 else float("nan")))
pd.DataFrame(rd).to_csv(os.path.join(a.out, "s10db_reading.csv"), index=False)
print(C[C["contrast"].str.startswith(("C1", "C2", "C3", "desc K3"))].round(3).to_string(index=False)); print(pd.DataFrame(rd).to_string(index=False))
