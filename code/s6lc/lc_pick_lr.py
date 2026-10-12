#!/usr/bin/env python3
"""s6lc: learning rates chosen on the validation sample (VS) from the pilot runs (size f16 = about 4,600 rows, seed 0), before the pre-registration is frozen.
V (LoRA VLM): lr in {5e-5, 1e-4, 2e-4}; criterion = VS mean AUC (mass, calc, asymmetry) of the best of the three checkpoints; ties -> 1e-4.
D (stage-1 head): lr in {1e-4, 3e-4, 1e-3}; criterion = VS mean AUC of P(present) from dump_val.pt; ties -> 3e-4.
Writes runs_lc/lr_V.txt, runs_lc/lr_D.txt, runs_lc/pilot_choice.json and the symlinks runs_lc/V/f16_s0 and runs_lc/D/f16_s0 -> the chosen pilot run (reused as the main run of that size and seed).
usage: PYTHONPATH=$PWD:$PWD/analysis_embed python3 analysis_embed/lc_pick_lr.py"""
import json, os
import numpy as np, torch
from sklearn.metrics import roc_auc_score
meta = json.load(open("runs_lc/sets/meta.json")); vs = json.load(open("runs_lc/sets/val_sample.json")); FIND = ["mass", "calc", "asymmetry"]
v = {}
for lr in ["5e-5", "1e-4", "2e-4"]:
    d = json.load(open(f"runs_lc/V/f16_s0_lr{lr}/selected.json")); v[lr] = dict(mean=d["per_ckpt"][str(d["selected"])]["mean"], ckpt=d["selected"], per_ckpt={k: x["mean"] for k, x in d["per_ckpt"].items()})
lr_v = max(v, key=lambda k: (round(v[k]["mean"], 6), k == "1e-4"))
dd = {}
for dlr in ["1e-4", "3e-4", "1e-3"]:
    d = torch.load(f"runs_lc/D/f16_s0_lr{dlr}/dump_val.pt", map_location="cpu"); ix = {s: i for i, s in enumerate(d["sample_id"])}; au = []
    for j, f in enumerate(FIND):
        y = np.array([meta[s]["y"][j] for s in vs]); k = y >= 0; sc = np.array([float(d["p_" + f][ix[s], 1]) for s in vs]); au.append(float(roc_auc_score(y[k], sc[k])))
    dd[dlr] = dict(mean=float(np.mean(au)), per_finding=au)
lr_d = max(dd, key=lambda k: (round(dd[k]["mean"], 6), k == "3e-4"))
json.dump(dict(V=v, V_star=lr_v, D=dd, D_star=lr_d), open("runs_lc/pilot_choice.json", "w"), indent=1)
open("runs_lc/lr_V.txt", "w").write(lr_v); open("runs_lc/lr_D.txt", "w").write(lr_d)
for base, star in (("runs_lc/V/f16_s0", lr_v), ("runs_lc/D/f16_s0", lr_d)):
    if os.path.islink(base): os.remove(base)
    os.symlink(f"f16_s0_lr{star}", base)
print("V:", {k: round(x["mean"], 4) for k, x in v.items()}, "->", lr_v); print("D:", {k: round(x["mean"], 4) for k, x in dd.items()}, "->", lr_d)
