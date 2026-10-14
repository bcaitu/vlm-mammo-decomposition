#!/usr/bin/env python3
"""s10db: writes runs_s10db/tasks.json from runs_s10db/vbudget.csv (copy of data/lc_training_budget.csv: aggregate metadata of the VLM runs).
D_bV: for every training size and seed, updates = the step of the V checkpoint that was used (selected_step; 4,591 at the whole pool); D_b2x: whole pool, updates = steps of V2x (9,182).
usage: ~/envs/cbmammo/bin/python analysis_embed/s10db_make_tasks.py"""
import json, os, csv
os.chdir(os.path.expanduser("~/cbmammo"))
rows = list(csv.DictReader(open("runs_s10db/vbudget.csv"))); tasks = []
PY = "~/envs/cbmammo/bin/python"
def cmd(frac, seed, upd, out):
    return (f"cd ~/cbmammo && PYTHONPATH=$PWD:$PWD/analysis_embed {PY} analysis_embed/lc_dtrain_budget.py --config configs/cb_embed_dinov2.yaml --feat runs_lc/Dfeat "
            f"--rows runs_lc/sets/subset_s{seed}_{frac}.json --seed {seed} --updates {upd} --out {out} --lr $(cat runs_lc/lr_D.txt) --workers 4")
for r in rows:
    if r["arm"] == "V":
        frac, seed, upd = r["size"], int(r["seed"]), int(float(r["selected_step"]))
        tasks.append(dict(name=f"DB_{frac}_s{seed}", updates=upd, out=f"runs_s10db/DB/{frac}_s{seed}", cmd=cmd(frac, seed, upd, f"runs_s10db/DB/{frac}_s{seed}")))
    if r["arm"] == "V2x" and r["size"] == "f1":
        seed, upd = int(r["seed"]), int(float(r["steps_scheduled"]))
        tasks.append(dict(name=f"DB2x_f1_s{seed}", updates=upd, out=f"runs_s10db/DB/f1x2_s{seed}", cmd=cmd("f1", seed, upd, f"runs_s10db/DB/f1x2_s{seed}")))
tasks.sort(key=lambda t: (-t["updates"], t["name"]))
assert len(tasks) == 21, len(tasks)
json.dump(tasks, open("runs_s10db/tasks.json", "w"), indent=1); print(len(tasks), "tasks;", sorted({t["updates"] for t in tasks}))
