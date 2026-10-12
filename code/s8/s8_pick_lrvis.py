#!/usr/bin/env python3
"""s8: learning rate of the vision-tower LoRA chosen on the validation sample (VS) from the two pilot runs (size f16, seed 0, 2,295 steps), before the main VC runs start.
lr_vis in {2e-5, 1e-4}; criterion = VS mean AUC (mass, calc, asymmetry) of the best of the three checkpoints (selected.json from lc_pick.py); if the difference is below 0.005 -> 2e-5.
Writes runs_s8/lr_vis.txt and runs_s8/pilot_choice.json. usage: python3 analysis_embed/s8_pick_lrvis.py"""
import json
v = {}
for lr in ["2e-5", "1e-4"]:
    d = json.load(open(f"runs_s8/pilot/lrvis{lr}/selected.json")); v[lr] = dict(mean=d["per_ckpt"][str(d["selected"])]["mean"], ckpt=d["selected"], per_ckpt={k: x["mean"] for k, x in d["per_ckpt"].items()}, per_finding={k: x["auc"] for k, x in d["per_ckpt"].items()})
star = "1e-4" if v["1e-4"]["mean"] - v["2e-5"]["mean"] >= 0.005 else "2e-5"
json.dump(dict(pilot=v, lr_vis=star, rule="1e-4 if mean(1e-4) - mean(2e-5) >= 0.005 else 2e-5"), open("runs_s8/pilot_choice.json", "w"), indent=1)
open("runs_s8/lr_vis.txt", "w").write(star); print({k: round(x["mean"], 4) for k, x in v.items()}, "->", star)
