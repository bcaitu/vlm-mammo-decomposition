#!/usr/bin/env python3
"""Image cache for the fresh evaluation rows (paper 4, s4v3eval); same preprocessing as vlm_make_cache.py (896 px, 1-99.5 percentile window)."""
import argparse, hashlib, json, os, sys, time
import numpy as np
from multiprocessing import Pool
from PIL import Image
from cbmammo import data as D
ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=32); ap.add_argument("--side", type=int, default=896)
a = ap.parse_args()
ALL = ["manifests/cbis_ddsm.jsonl", "manifests/vindr.jsonl", "manifests/embed.jsonl", "manifests/cmmd.jsonl", "manifests/cdd_cesm.jsonl", "manifests/inbreast.jsonl"]
recs = {r["sample_id"]: r for r in D.load_manifest(ALL)}
rows = json.load(open(a.rows)); tasks = []
for sp, lst in rows.items():
    for r in lst:
        for v, p in recs[r["sample_id"]]["views"].items():
            if p: tasks.append((r["sample_id"], v, p))
def fname(sid, v): return hashlib.sha1(f"{sid}|{v}".encode()).hexdigest()[:20] + ".png"
def work(t):
    sid, v, p = t; out = os.path.join(a.out, fname(sid, v))
    if os.path.exists(out): return ""
    try:
        x = np.asarray(D.read_image(p)).astype(np.float32)
        if x.ndim == 3: x = x[..., 0]
        lo, hi = np.percentile(x, 1), np.percentile(x, 99.5); x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
        im = Image.fromarray((x * 255).astype(np.uint8)); w, h = im.size; s = a.side / max(w, h)
        im = im.resize((max(32, round(w * s)), max(32, round(h * s))), Image.BILINEAR if s > 1 else Image.LANCZOS)
        im.save(out + ".tmp", format="PNG", compress_level=1); os.replace(out + ".tmp", out); return ""
    except Exception as e:
        return str(e)[:120]
t0 = time.time(); err = 0
with Pool(a.workers) as pool:
    for i, r in enumerate(pool.imap(work, tasks, chunksize=8), 1):
        err += bool(r)
        if i % 1000 == 0: print(f"{i}/{len(tasks)} done {time.time()-t0:.0f}s, errors {err}", flush=True)
print("finished", len(tasks), "errors", err, round(time.time() - t0), flush=True)
