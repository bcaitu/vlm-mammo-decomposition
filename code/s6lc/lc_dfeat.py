#!/usr/bin/env python3
"""s6lc (paper 4, question 1): arm D on cached features. The encoder of the stage-1 model (DINOv2 ViT-B/14, frozen) is run once, WITHOUT augmentation, on every row needed (pooled train manifests, all val rows, EMBED test) and the
2x2-pooled token grid of both views is stored as one float16 array (rows, 2, 768, 37, 23). The stage-1 head is then trained on these features (lc_dtrain.py) at all subset sizes: this removes the CPU cost of image preprocessing
(the account is limited to 20 CPU cores) at the price of dropping the training-time augmentation of the original recipe.
usage: PYTHONPATH=$PWD:$PWD/analysis_embed python3 analysis_embed/lc_dfeat.py --config configs/cb_embed_dinov2.yaml --out runs_lc/Dfeat --workers 6"""
import argparse, json, os, sys, time
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from cbmammo import data as D
from cbmammo.model import FrozenEncoder
ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
if os.path.exists(os.path.join(a.out, "done.flag")): print("[skip] done"); raise SystemExit
cfg = yaml.safe_load(open(a.config)); e, dc = cfg["encoder"], cfg["data"]; size = tuple(dc["size"])
recs = D.load_manifest(dc["train_manifests"])
rows = sorted([r for r in recs if r["split"] in ("train", "val") or (r["dataset"] == "embed" and r["split"] == "test")], key=lambda r: r["sample_id"])
if a.limit: rows = rows[: a.limit]
enc = FrozenEncoder(e["name"], True, None, e.get("token_pool", 2), 0).cuda().eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): g = enc(torch.zeros(1, 3, *size, device="cuda")).shape
C, h, w = int(g[1]), int(g[2]), int(g[3]); print("rows", len(rows), "grid", C, h, w, flush=True)
mm = np.lib.format.open_memmap(os.path.join(a.out, "feat.npy"), mode="w+", dtype=np.float16, shape=(len(rows), 2, C, h, w))
dl = DataLoader(D.BreastDataset(rows, size, False, dc.get("clahe", True), dc.get("cache_dir")), batch_size=a.bs, shuffle=False, num_workers=a.workers, collate_fn=D.collate, pin_memory=True, prefetch_factor=4)
t0 = time.time(); i0 = 0
with torch.no_grad():
    for bi, b in enumerate(dl):
        B = b["cc"].shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16): f = enc(torch.cat([b["cc"], b["mlo"]], 0).cuda(non_blocking=True))
        f = f.float().half().cpu().numpy(); mm[i0: i0 + B, 0] = f[:B]; mm[i0: i0 + B, 1] = f[B:]; i0 += B
        if (bi + 1) % 100 == 0: print(f"{i0}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
mm.flush()
json.dump(dict(sample_id=[r["sample_id"] for r in rows], view_mask=[[bool(r["views"].get("CC")), bool(r["views"].get("MLO"))] for r in rows], shape=[len(rows), 2, C, h, w]), open(os.path.join(a.out, "index.json"), "w"))
open(os.path.join(a.out, "done.flag"), "w").write(str(round(time.time() - t0))); print("finished", len(rows), round(time.time() - t0), flush=True)
