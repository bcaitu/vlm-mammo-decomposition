#!/usr/bin/env python3
"""Reads aggregate metadata (steps, selected checkpoints, epochs) of the learning-curve, step-budget and specialist-head runs on the compute server and prints data/lc_training_budget.csv. Server layout: ~/cbmammo/runs_lc, runs_vlm, runs_s8. Descriptive, written after the results; it reads logs only.
usage: python3 extract_training_budget.py > lc_training_budget.csv"""
import json, os, csv, sys, glob
os.chdir(os.path.expanduser("~/cbmammo"))
BS = 16
rows = []
summ = {}
for r in csv.DictReader(open("runs_lc/sets/sets_summary.csv")):
    summ[(int(r["seed"]), r["frac"])] = r
def pos(seed, frac):
    r = summ[(seed, frac)]; return int(r["rows"]), int(r["calc_pos"]), int(r["asymmetry_pos"])
# V learning-curve runs (f32..f2: 2,295-step schedule with checkpoint selection)
for frac in ["f32","f16","f8","f4","f2"]:
    for seed in (0,1,2):
        d = f"runs_lc/V/{frac}_s{seed}"
        tl = json.load(open(f"{d}/train_log.json")); sel = json.load(open(f"{d}/selected.json"))["selected"]
        n, pc, pa = pos(seed, frac)
        rows.append(dict(arm="V", size=frac, seed=seed, subset_rows=n, calc_pos=pc, asym_pos=pa, steps_scheduled=tl["steps"], selected_step=int(sel), epochs_or_updates_per_epoch="", epochs_run="", rows_per_epoch=""))
# V full size 4,591 steps (no selection), V_ctl 2,295 (selection), VC, V2x
for seed in (0,1,2):
    tl = json.load(open(f"runs_vlm/v1_s{seed}/train_log.json"))
    rows.append(dict(arm="V", size="f1", seed=seed, subset_rows=tl["rows"], calc_pos=3715, asym_pos=5155, steps_scheduled=tl["steps"], selected_step=tl["steps"], epochs_or_updates_per_epoch="", epochs_run="", rows_per_epoch=""))
    d = f"runs_lc/V/ctl_s{seed}"; tl = json.load(open(f"{d}/train_log.json")); sel = json.load(open(f"{d}/selected.json"))["selected"]
    rows.append(dict(arm="V_ctl", size="f1", seed=seed, subset_rows=tl["rows"], calc_pos=3715, asym_pos=5155, steps_scheduled=tl["steps"], selected_step=int(sel), epochs_or_updates_per_epoch="", epochs_run="", rows_per_epoch=""))
    for arm in ("VC","V2x"):
        tl = json.load(open(f"runs_s8/{arm}/f1_s{seed}/train_log.json"))
        rows.append(dict(arm=arm, size="f1", seed=seed, subset_rows=tl["rows"], calc_pos=3715, asym_pos=5155, steps_scheduled=tl["steps"], selected_step=tl["steps"], epochs_or_updates_per_epoch="", epochs_run="", rows_per_epoch=""))
# D (cached features), all sizes
for frac in ["f32","f16","f8","f4","f2","f1"]:
    for seed in (0,1,2):
        d = f"runs_lc/D/{frac}_s{seed}"
        lg = json.load(open(f"{d}/train_log.json"))
        best = -1; be = 0
        for e in lg:
            if e["val_score"] > best: best = e["val_score"]; be = e["epoch"] + 1
        n, pc, pa = pos(seed, frac)
        per = int(round(sum(e["n_train_epoch"] for e in lg) / len(lg)))
        rows.append(dict(arm="D", size=frac, seed=seed, subset_rows=n, calc_pos=pc, asym_pos=pa, steps_scheduled="", selected_step="", epochs_or_updates_per_epoch=per // BS, epochs_run=len(lg), rows_per_epoch=per, selected_epoch=be))
keys = ["arm","size","seed","subset_rows","calc_pos","asym_pos","steps_scheduled","selected_step","epochs_or_updates_per_epoch","epochs_run","rows_per_epoch","selected_epoch"]
w = csv.DictWriter(sys.stdout, fieldnames=keys, lineterminator="\n"); w.writeheader()
for r in rows: w.writerow({k: r.get(k, "") for k in keys})
