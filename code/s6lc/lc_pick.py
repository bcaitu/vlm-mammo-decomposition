#!/usr/bin/env python3
"""s6lc: choose the LoRA checkpoint on the validation sample (VS): mean AUC over mass, calc, asymmetry of the model's own score (-log P(' No')); ties go to the later checkpoint.
Reads <run>/vs_ck<step>.npz (lc_score.py --no_feats on val_sample.json), writes <run>/selected.json and the symlink <run>/selected -> ckpt_<step>.
usage: lc_pick.py --run runs_lc/V/f16_s0 --meta runs_lc/sets/meta.json"""
import argparse, glob, json, os, re
import numpy as np
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--meta", required=True); a = ap.parse_args()
meta = json.load(open(a.meta)); res = {}
for f in sorted(glob.glob(os.path.join(a.run, "vs_ck*.npz")), key=lambda p: int(re.search(r"ck(\d+)", p).group(1))):
    z = np.load(f); sid = [str(x) for x in z["sample_id"]]; au = []
    for j in range(3):
        y = np.array([meta[s]["y"][j] for s in sid]); k = y >= 0; au.append(float(roc_auc_score(y[k], -z["logp_no"][k, j])))
    res[int(re.search(r"ck(\d+)", f).group(1))] = dict(auc=au, mean=float(np.mean(au)))
assert res, "no vs_ck files"
best = max(res, key=lambda s: (round(res[s]["mean"], 6), s)); json.dump(dict(selected=best, per_ckpt=res), open(os.path.join(a.run, "selected.json"), "w"), indent=1)
lnk = os.path.join(a.run, "selected")
if os.path.islink(lnk): os.remove(lnk)
os.symlink(f"ckpt_{best}", lnk); print("selected", best, {s: round(v["mean"], 4) for s, v in res.items()})
