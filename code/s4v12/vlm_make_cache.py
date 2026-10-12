#!/usr/bin/env python3
"""Builds the image cache for the VLM study (paper 4): every view of the training rows, 1,000 validation rows and the grid-2 evaluation subsample of the eight test sets,
resized to a longest side of 896 px, 1-99.5 percentile window, 8-bit grayscale PNG. File names are hashes; the index maps sample ids to files (server only).
usage: PYTHONPATH=$PWD python3 analysis_embed/vlm_make_cache.py --out /raid/home/user0/cbmammo_cache/vlm896 --workers 48 [--limit 40]"""
import argparse, hashlib, json, os, sys, time, random
import numpy as np, torch
from multiprocessing import Pool
from PIL import Image
from cbmammo import data as D
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=48); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--side", type=int, default=896)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
TRAIN = ["manifests/cbis_ddsm.jsonl", "manifests/vindr.jsonl", "manifests/embed.jsonl"]
TESTM = TRAIN + ["manifests/cmmd.jsonl", "manifests/cdd_cesm.jsonl", "manifests/inbreast.jsonl"]
recs = {r["sample_id"]: r for r in D.load_manifest(TESTM)}
role = {}
for sid, r in recs.items():
    if r["split"] == "train" and r["dataset"] in ("cbis_ddsm", "vindr", "embed"): role[sid] = "train"
val = sorted(sid for sid, r in recs.items() if r["split"] == "val" and r["dataset"] in ("cbis_ddsm", "vindr", "embed"))
random.Random(0).shuffle(val)
for sid in val[:1000]: role[sid] = "val"
SETS = ["test_cbis_ddsm", "test_cdd_cesm", "test_cmmd", "test_embed", "test_embed_fuji", "test_embed_ge", "test_inbreast", "test_vindr"]
run = "runs/cb_embed_dinov2_s0"; g2 = "runs_stage2/grid2_cb_embed_dinov2_s0"
for sp in SETS:
    sids = torch.load(os.path.join(run, f"dump_{sp}.pt"), map_location="cpu")["sample_id"]
    rows = [json.loads(l)["dump_row"] for l in open(os.path.join(g2, f"gen_{sp}.jsonl"))]
    for i in rows: role[sids[i]] = sp
tasks = []
for sid, ro in role.items():
    for v, p in recs[sid]["views"].items():
        if p: tasks.append((sid, v, p, ro))
tasks.sort(key=lambda t: (t[3], t[0], t[1]))
if a.limit: random.Random(1).shuffle(tasks); tasks = tasks[: a.limit]
print("rows", len(role), "images", len(tasks), {r: sum(1 for t in tasks if t[3] == r) for r in sorted(set(t[3] for t in tasks))}, flush=True)
def fname(sid, v): return hashlib.sha1(f"{sid}|{v}".encode()).hexdigest()[:20] + ".png"
def work(t):
    sid, v, p, ro = t; out = os.path.join(a.out, fname(sid, v))
    if os.path.exists(out): return (sid, v, fname(sid, v), os.path.getsize(out), "")
    try:
        x = np.asarray(D.read_image(p)).astype(np.float32)
        if x.ndim == 3: x = x[..., 0]
        lo, hi = np.percentile(x, 1), np.percentile(x, 99.5); x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
        im = Image.fromarray((x * 255).astype(np.uint8)); w, h = im.size; s = a.side / max(w, h)
        im = im.resize((max(32, round(w * s)), max(32, round(h * s))), Image.BILINEAR if s > 1 else Image.LANCZOS)
        im.save(out + ".tmp", format="PNG", compress_level=1); os.replace(out + ".tmp", out)
        return (sid, v, fname(sid, v), os.path.getsize(out), "")
    except Exception as e:
        return (sid, v, "", 0, str(e)[:120])
t0 = time.time(); res = []
with Pool(a.workers) as pool:
    for i, r in enumerate(pool.imap(work, tasks, chunksize=16), 1):
        res.append(r)
        if i % 2000 == 0: print(f"{i}/{len(tasks)} done {time.time()-t0:.0f}s, errors {sum(1 for x in res if x[4])}", flush=True)
idx = {}
for sid, v, f, size, err in res:
    if f: idx.setdefault(sid, {"sample_id": sid, "role": role[sid], "dataset": recs[sid]["dataset"], "views": {}})["views"][v] = f
with open(os.path.join(a.out, "index.jsonl"), "w") as fo:
    for sid in sorted(idx): fo.write(json.dumps(idx[sid]) + "\n")
errs = [x for x in res if x[4]]
json.dump(dict(images=len(tasks), rows=len(idx), errors=len(errs), bytes=int(sum(x[3] for x in res)), seconds=round(time.time() - t0), side=a.side), open(os.path.join(a.out, "cache_meta.json"), "w"))
print("finished", len(tasks), "errors", len(errs), "GB", round(sum(x[3] for x in res) / 2**30, 1), round(time.time() - t0), flush=True)
