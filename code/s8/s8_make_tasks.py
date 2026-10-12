#!/usr/bin/env python3
"""s8: writes runs_s8/tasks.json for lc_queue.py (pilot lr_vis, VC x3 seeds, V2x x3 seeds, scoring, probes)."""
import json
PY = "~/envs/cbmammo/bin/python"; T = []
def add(name, cmd, done, needs=(), prio=100, wait=()): T.append(dict(name=name, cmd="cd ~/cbmammo && " + cmd, done=list(done), needs=list(needs), gpu=True, prio=prio, manual=False, wait=list(wait)))
VS = "runs_lc/sets/val_sample.json"; META = "runs_lc/sets/meta.json"; TEST = "runs_lc/sets/test_embed.json"
for lr in ["2e-5", "1e-4"]:
    o = f"runs_s8/pilot/lrvis{lr}"
    add(f"pil_{lr}", f"{PY} analysis_embed/s8_train.py --rows runs_lc/sets/subset_s0_f16.json --out {o} --seed 0 --steps 2295 --ckpt 765,1530,2295 --lr 1e-4 --lr_vis {lr} --tower_lora --workers 4", [f"{o}/train_log.json"], prio=300)
    sc = " && ".join(f"{PY} analysis_embed/lc_score.py --rows {VS} --adapter {o}/ckpt_{k} --out {o}/vs_ck{k}.npz --no_feats --bs 1 --workers 4" for k in (765, 1530, 2295))
    add(f"vsc_pil_{lr}", f"{sc} && {PY} analysis_embed/lc_pick.py --run {o} --meta {META}", [f"{o}/selected.json"], needs=[f"pil_{lr}"], prio=290)
for s in (0, 1, 2):
    o = f"runs_s8/V2x/f1_s{s}"
    add(f"trn_V2x_s{s}", f"{PY} analysis_embed/s8_train.py --rows runs_lc/sets/subset_s{s}_f1.json --out {o} --seed {s} --steps 9182 --ckpt 2295,4591,6886,9182 --lr 1e-4 --workers 4", [f"{o}/train_log.json"], prio=250 - s)
    add(f"ext_V2x_s{s}", f"{PY} analysis_embed/lc_score.py --rows {TEST} --adapter {o}/ckpt_9182 --out {o}/ext.npz --no_feats --bs 1 --workers 4", [f"{o}/ext.npz"], needs=[f"trn_V2x_s{s}"], prio=200)
for s in (0, 1, 2):
    o = f"runs_s8/VC/f1_s{s}"
    add(f"trn_VC_s{s}", f"{PY} analysis_embed/s8_train.py --rows runs_lc/sets/subset_s{s}_f1.json --out {o} --seed {s} --steps 4591 --ckpt 1530,3061,4591 --lr 1e-4 --lr_vis $(cat runs_s8/lr_vis.txt) --tower_lora --workers 4", [f"{o}/train_log.json"], prio=280 - s, wait=["runs_s8/GATE2"])
    add(f"ext_VC_s{s}", f"{PY} analysis_embed/lc_score.py --rows runs_lc/sets/ext_s{s}_f1.json --adapter {o}/ckpt_4591 --out {o}/ext.npz --bs 1 --workers 4", [f"{o}/ext.npz"], needs=[f"trn_VC_s{s}"], prio=220)
    add(f"prb_VC_s{s}", f"{PY} analysis_embed/lc_probe.py --feats {o}/ext.npz --train runs_lc/sets/subset_s{s}_f1.json --vs {VS} --test {TEST} --meta {META} --out {o}/probe.npz --grid {o}/probe_grid.csv", [f"{o}/probe.npz"], needs=[f"ext_VC_s{s}"], prio=190)
json.dump(T, open("runs_s8/tasks.json", "w"), indent=1); print(len(T), "tasks")
