#!/usr/bin/env python3
"""s8 (paper 4), POST HOC 2 (not pre-registered; written and run after the main analysis and post hoc 1). Same data loading, bootstrap resamples (seed 20261007, 2000 patient-level
resamples) and seed-mean AUC definition as s8_analysis.py (its source is executed up to the definition of `diff`, writing its duplicate outputs to --out for a determinism check).
Contrasts: arm - D for arm in A_probe, A_tf, VP, VCP, V_ctl, V, V2x, VC and all three findings; used only for the summary figure 'distance to the specialist head'.
Overlap with earlier outputs (checked by hand after the run): V - D = s6lc K3; VC - D = S2; V2x - D and VP - D, VCP - D = s8 descriptive / post hoc 1.
usage: PYTHONPATH=$PWD:$PWD/analysis_embed python3 analysis_embed/s8_posthoc2.py --out runs_s8/analysis_posthoc2 [--boot 2000]"""
import sys, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s8_analysis.py")).read()
head = src[:src.index("def diff(")]
exec(compile(head, "s8_analysis.py[prefix]", "exec"))   # defines a, AUC, FIND, SEEDS, logp, boot_auc, ...
import numpy as np, pandas as pd
for f in FIND:
    lst = []
    for sd in SEEDS:
        sc = logp(f"runs_lc/V/ctl_s{sd}/test.npz")
        if sc is None: continue
        b = boot_auc(sc[f], f)
        if b is not None: lst.append(b)
    if lst: AUC[("V_ctl", f)] = np.array(lst)
def m(arm, f): return AUC[(arm, f)].mean(0)
rows = []
for f in FIND:
    for arm in ["A_probe", "A_tf", "VP", "VCP", "V_ctl", "V", "V2x", "VC"]:
        d = m(arm, f) - m("D", f)
        rows.append(dict(arm=arm, finding=f, auc=float(m(arm, f)[0]), delta_vs_D=float(d[0]), lo=float(np.percentile(d[1:], 2.5)), hi=float(np.percentile(d[1:], 97.5))))
P = pd.DataFrame(rows); P.to_csv(f"{a.out}/s8_posthoc2_gap_to_D.csv", index=False)
print(P.round(3).to_string(index=False))
