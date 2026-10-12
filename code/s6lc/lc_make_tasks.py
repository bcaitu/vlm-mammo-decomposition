#!/usr/bin/env python3
"""s6lc: writes the union row lists and the task list for lc_queue.py (stage main).
Two manual gates: gate1 = runs_lc/PREREG_FROZEN (pre-registration, code and row lists hashed; opens the LR-independent tasks), gate2 = runs_lc/LR_FROZEN (pilot choice of the learning rates recorded; opens the LR-dependent tasks).
usage: PYTHONPATH=$PWD:$PWD/analysis_embed python3 analysis_embed/lc_make_tasks.py --stage main --out runs_lc/tasks_main.json"""
import argparse, json, os, random
ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["main"], default="main"); ap.add_argument("--out", required=True); a = ap.parse_args()
S = "runs_lc/sets"; PY = "analysis_embed"; FR = ["f32", "f16", "f8", "f4", "f2"]; SEEDS = [0, 1, 2]
vs, te = json.load(open(f"{S}/val_sample.json")), json.load(open(f"{S}/test_embed.json"))
def union(*ls):
    out, seen = [], set()
    for l in ls:
        for x in l:
            if x not in seen: seen.add(x); out.append(x)
    return out
for s in SEEDS:
    for f in FR + ["f1"]:
        fn = f"{S}/ext_s{s}_{f}.json"
        if not os.path.exists(fn): json.dump(union(json.load(open(f"{S}/subset_s{s}_{f}.json")), vs, te), open(fn, "w"))
if not os.path.exists(f"{S}/ext_base.json"): json.dump(union(json.load(open(f"{S}/subset_s0_f1.json")), vs, te), open(f"{S}/ext_base.json", "w"))
if not os.path.exists(f"{S}/test_sub3000.json"): json.dump(sorted(random.Random(20261007).sample(te, 3000)), open(f"{S}/test_sub3000.json", "w"))
tasks = []
def py(c): return f"~/envs/cbmammo/bin/python {c}"
def T(name, cmd, done, needs=(), gate=1, gpu=True, prio=0, manual=False, wait=()):
    nd = list(needs) + ([] if manual else ["gate1"] if gate == 1 else ["gate1", "gate2"])
    tasks.append(dict(name=name, cmd=f"cd ~/cbmammo && {cmd}", done=done if isinstance(done, list) else [done], needs=nd, gpu=gpu, prio=prio, manual=manual, wait=list(wait)))
def vsc(run, needs, name, prio, gate, wait=()):
    cmds = [f"{PY}/lc_score.py --rows {S}/val_sample.json --adapter {run}/ckpt_{ck} --out {run}/vs_ck{ck}.npz --no_feats --bs 1 --workers 4" for ck in (765, 1530, 2295)] + [f"{PY}/lc_pick.py --run {run} --meta {S}/meta.json"]
    T(name, " && ".join(py(c) for c in cmds), f"{run}/selected.json", needs=needs, prio=prio, gate=gate, wait=wait)
T("gate1", "true", "runs_lc/PREREG_FROZEN", manual=True, gpu=False); T("gate2", "true", "runs_lc/LR_FROZEN", manual=True, gpu=False)
LRV, LRD = "$(cat runs_lc/lr_V.txt)", "$(cat runs_lc/lr_D.txt)"
# ---- LR-independent (gate1): cached DINOv2 features, D pilot, pilot scoring, base-model extraction and arms A, v1 top-size extraction, blank control
T("dfeat", py(f"{PY}/lc_dfeat.py --config configs/cb_embed_dinov2.yaml --out runs_lc/Dfeat --workers 4"), "runs_lc/Dfeat/done.flag", prio=300)
for dlr in ["1e-4", "3e-4", "1e-3"]:
    T(f"Dp_{dlr}", py(f"{PY}/lc_dtrain.py --config configs/cb_embed_dinov2.yaml --feat runs_lc/Dfeat --rows {S}/subset_s0_f16.json --seed 0 --out runs_lc/D/f16_s0_lr{dlr} --lr {dlr} --workers 4"), f"runs_lc/D/f16_s0_lr{dlr}/run.json", needs=["dfeat"], prio=250)
for lr in ["5e-5", "1e-4", "2e-4"]:
    vsc(f"runs_lc/V/f16_s0_lr{lr}", [], f"vsc_pilot_{lr}", 280, 1, wait=[f"runs_lc/V/f16_s0_lr{lr}/train_log.json"])
for k in range(5):
    T(f"bext_{k}", py(f"{PY}/lc_score.py --rows {S}/ext_base.json --out runs_lc/A/base_shard{k}.npz --tokens runs_lc/A/tokens --shard {k}/5 --bs 1 --workers 4"), f"runs_lc/A/base_shard{k}.npz", prio=200)
bshards = ",".join(f"runs_lc/A/base_shard{k}.npz" for k in range(5)); bneeds = [f"bext_{k}" for k in range(5)]
for s in SEEDS:
    run = f"runs_lc/V/f1_s{s}"   # top size: the three adapters of v1 (4,591 steps, final adapter only); features on pool + VS + test
    T(f"ext_f1_s{s}", py(f"{PY}/lc_score.py --rows {S}/ext_s{s}_f1.json --adapter runs_vlm/v1_s{s} --out {run}/ext.npz --bs 1 --workers 4"), f"{run}/ext.npz", prio=210)
    T(f"prb_f1_s{s}", py(f"{PY}/lc_probe.py --feats {run}/ext.npz --train {S}/subset_s{s}_f1.json --vs {S}/val_sample.json --test {S}/test_embed.json --meta {S}/meta.json --out {run}/probe.npz --grid {run}/probe_grid.csv"), f"{run}/probe.npz", needs=[f"ext_f1_s{s}"], prio=190)
T("blank_v1_s0", py(f"{PY}/lc_score.py --rows {S}/test_sub3000.json --adapter runs_vlm/v1_s0 --blank --no_feats --out runs_lc/V/blank_s0/test.npz --bs 1 --workers 4"), "runs_lc/V/blank_s0/test.npz", prio=40)
for s in SEEDS:
    for f in ["f1", "f2", "f4", "f8", "f16", "f32"]:
        pr = {"f1": 195, "f2": 185, "f4": 175, "f8": 165, "f16": 155, "f32": 145}[f]
        if not (f == "f1" and s > 0):   # identical training set for all seeds: the linear probes are deterministic, compute once
            T(f"prbA_{f}_s{s}", py(f"{PY}/lc_probe.py --feats {bshards} --train {S}/subset_s{s}_{f}.json --vs {S}/val_sample.json --test {S}/test_embed.json --meta {S}/meta.json --out runs_lc/A/probe_{f}_s{s}.npz --grid runs_lc/A/probe_{f}_s{s}_grid.csv"), f"runs_lc/A/probe_{f}_s{s}.npz", needs=bneeds, prio=pr)
        T(f"tfA_{f}_s{s}", py(f"{PY}/lc_tfhead.py --tokens runs_lc/A/tokens --train {S}/subset_s{s}_{f}.json --vs {S}/val_sample.json --test {S}/test_embed.json --meta {S}/meta.json --out runs_lc/A/tf_{f}_s{s}.npz --seed {s} --workers 4"), f"runs_lc/A/tf_{f}_s{s}.npz", needs=bneeds, prio=pr + 2)
# ---- LR-dependent (gate1 + gate2): V runs at f32..f2 with checkpoint selection, features and probes; D at all sizes; control at the top size
est = {"f32": 60, "f16": 70, "f8": 80, "f4": 90, "f2": 95}
for s in SEEDS:
    for f in FR:
        run = f"runs_lc/V/{f}_s{s}"; pr = est[f]
        T(f"trn_{f}_s{s}", py(f"{PY}/lc_train.py --rows {S}/subset_s{s}_{f}.json --out {run} --seed {s} --steps 2295 --ckpt 765,1530,2295 --lr {LRV} --workers 4"), f"{run}/train_log.json", gate=2, prio=100 + pr // 10)
        vsc(run, [f"trn_{f}_s{s}"], f"vsc_{f}_s{s}", 92, 2)
        T(f"ext_{f}_s{s}", py(f"{PY}/lc_score.py --rows {S}/ext_s{s}_{f}.json --adapter {run}/selected --out {run}/ext.npz --bs 1 --workers 4"), f"{run}/ext.npz", needs=[f"vsc_{f}_s{s}"], gate=2, prio=60 + pr // 10)
        T(f"prb_{f}_s{s}", py(f"{PY}/lc_probe.py --feats {run}/ext.npz --train {S}/subset_s{s}_{f}.json --vs {S}/val_sample.json --test {S}/test_embed.json --meta {S}/meta.json --out {run}/probe.npz --grid {run}/probe_grid.csv"), f"{run}/probe.npz", needs=[f"ext_{f}_s{s}"], gate=2, prio=55)
    for f in ["f1"] + FR:
        T(f"D_{f}_s{s}", py(f"{PY}/lc_dtrain.py --config configs/cb_embed_dinov2.yaml --feat runs_lc/Dfeat --rows {S}/subset_s{s}_{f}.json --seed {s} --out runs_lc/D/{f}_s{s} --lr {LRD} --workers 4"), f"runs_lc/D/{f}_s{s}/run.json", needs=["dfeat"], gate=2, prio=120)
    run = f"runs_lc/V/ctl_s{s}"   # control: top-size training with the same 2,295-step budget as the smaller sizes
    T(f"trn_ctl_s{s}", py(f"{PY}/lc_train.py --rows {S}/subset_s{s}_f1.json --out {run} --seed {s} --steps 2295 --ckpt 765,1530,2295 --lr {LRV} --workers 4"), f"{run}/train_log.json", gate=2, prio=98)
    vsc(run, [f"trn_ctl_s{s}"], f"vsc_ctl_s{s}", 92, 2)
    T(f"tst_ctl_s{s}", py(f"{PY}/lc_score.py --rows {S}/test_embed.json --adapter {run}/selected --out {run}/test.npz --no_feats --bs 1 --workers 4"), f"{run}/test.npz", needs=[f"vsc_ctl_s{s}"], gate=2, prio=60)
json.dump(tasks, open(a.out, "w"), indent=1); print(a.stage, len(tasks), "tasks ->", a.out)
