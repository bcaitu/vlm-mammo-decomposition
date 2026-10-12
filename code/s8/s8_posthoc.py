#!/usr/bin/env python3
"""s8 (paper 4), POST HOC (not pre-registered; written and run after the main analysis). Same data loading, bootstrap resamples (seed 20261007, 2000 patient-level resamples)
and seed-mean AUC definition as s8_analysis.py (its source is executed up to the definition of `diff`, writing its duplicate outputs to --out for a determinism check).
Extra contrasts: P1 VP - D, P2 VCP - D (probe on the VLM states vs the stage-1 head), P3 V - V_ctl (step doubling 2,295 -> 4,591, s6lc), P4 V2x - V (S3, repeated),
P5 V2x - V_ctl (two doublings), P6 (V2x - V) - (V - V_ctl) (second doubling minus first doubling; schedules differ between groups: V_ctl has its own 2,295-step schedule).
Intervals: percentile intervals of the seed-mean difference over the patient bootstrap. Verdict column: only whether the interval excludes 0.
usage: PYTHONPATH=$PWD:$PWD/analysis_embed python3 analysis_embed/s8_posthoc.py --out runs_s8/analysis_posthoc [--boot 2000]"""
import sys, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s8_analysis.py")).read()
head = src[:src.index("def diff(")]
exec(compile(head, "s8_analysis.py[prefix]", "exec"))   # defines a, AUC, FIND, SEEDS, logp, boot_auc, ...
import numpy as np, pandas as pd, json
# extra arm: V_ctl (s6lc control, 2,295 steps)
for f in FIND:
    lst = []
    for sd in SEEDS:
        sc = logp(f"runs_lc/V/ctl_s{sd}/test.npz")
        if sc is None: continue
        b = boot_auc(sc[f], f)
        if b is not None: lst.append(b)
    if lst: AUC[("V_ctl", f)] = np.array(lst)
def m(arm, f): return AUC[(arm, f)].mean(0)
def row(name, f, d):
    lo, hi = float(np.percentile(d[1:], 2.5)), float(np.percentile(d[1:], 97.5))
    return dict(contrast=name, finding=f, delta=float(d[0]), lo=lo, hi=hi, excludes_zero=bool(lo > 0 or hi < 0))
rows = []
for f in FIND:
    rows.append(row("P1 VP - D", f, m("VP", f) - m("D", f)))
    rows.append(row("P2 VCP - D", f, m("VCP", f) - m("D", f)))
    rows.append(row("P3 V - V_ctl (first doubling, 2295 -> 4591)", f, m("V", f) - m("V_ctl", f)))
    rows.append(row("P4 V2x - V (second doubling, 4591 -> 9182)", f, m("V2x", f) - m("V", f)))
    rows.append(row("P5 V2x - V_ctl (two doublings)", f, m("V2x", f) - m("V_ctl", f)))
    rows.append(row("P6 (V2x - V) - (V - V_ctl) (second minus first doubling)", f, (m("V2x", f) - m("V", f)) - (m("V", f) - m("V_ctl", f))))
P = pd.DataFrame(rows); P.to_csv(f"{a.out}/s8_posthoc.csv", index=False)
vc = pd.DataFrame([dict(arm="V_ctl", finding=f, n_seeds=len(AUC[("V_ctl", f)]), auc=float(m("V_ctl", f)[0]), lo=float(np.percentile(m("V_ctl", f)[1:], 2.5)), hi=float(np.percentile(m("V_ctl", f)[1:], 97.5))) for f in FIND if ("V_ctl", f) in AUC])
vc.to_csv(f"{a.out}/s8_posthoc_vctl_auc.csv", index=False)
print(P.round(3).to_string(index=False)); print(vc.round(3).to_string(index=False))
