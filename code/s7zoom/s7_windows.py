#!/usr/bin/env python3
"""s7zoom (direction C go/no-go): windows for the oracle-crop resolution-ladder test. VinDr-Mammo, official TRAINING split only (no EMBED, no CBIS, no test rows).
Positives: WxW native-pixel windows around 'Suspicious Calcification' boxes (window centre jittered by up to +-W/4 so that the lesion is not always centred).
Negatives: 'same' = windows from the positive images that intersect NO annotated box of any category; 'norm' = windows from images of breasts with BI-RADS 1-2 and no findings, studies disjoint from the positives.
A window must contain >= 80% breast tissue (intensity > 12 on the 1/8-scale image).
usage: python3 s7_windows.py --out runs_zoom/windows.json [--seed 20261008] [--win 512] [--neg_same 2] [--neg_norm 1]"""
import argparse, json, os
import numpy as np, pandas as pd
from PIL import Image
R = "/raid/datasets/vlm_mammogram_data/data/vindr-mammo"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=20261008); ap.add_argument("--win", type=int, default=512)
ap.add_argument("--neg_same", type=int, default=2); ap.add_argument("--neg_norm", type=int, default=1); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
rng = np.random.default_rng(a.seed); W = a.win
fa = pd.read_csv(f"{R}/finding_annotations.csv"); ba = pd.read_csv(f"{R}/breast-level_annotations.csv")
fa = fa[fa.split == "training"].copy(); ba = ba[ba.split == "training"].copy()
path = lambda s, i: f"{R}/images2/{s}/{i}.jpg"
def cover_ii(p):
    im = Image.open(p); w0, h0 = im.size; im.draft("L", (max(1, w0 // 8), max(1, h0 // 8))); im = im.convert("L"); x = np.asarray(im) > 12
    ii = np.pad(x.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0))); return ii, w0 / x.shape[1], h0 / x.shape[0], w0, h0
def coverage(ii, fx, fy, x0, y0):
    a_, b_, c_, d_ = int(x0 / fx), int(y0 / fy), int((x0 + W) / fx), int((y0 + W) / fy)
    c_ = min(c_, ii.shape[1] - 1); d_ = min(d_, ii.shape[0] - 1); area = max(1, (c_ - a_) * (d_ - b_))
    return (ii[d_, c_] - ii[b_, c_] - ii[d_, a_] + ii[b_, a_]) / area
def hits(boxes, x0, y0): return any(not (bx2 <= x0 or bx1 >= x0 + W or by2 <= y0 or by1 >= y0 + W) for bx1, by1, bx2, by2 in boxes)
is_calc = fa.finding_categories.astype(str).str.contains("Suspicious Calcification") & fa.xmin.notna()
pos = fa[is_calc].copy()
if a.limit: pos = pos.sample(min(a.limit, len(pos)), random_state=1)
rows = []; cache = {}
def get_ii(p):
    if p not in cache: cache[p] = cover_ii(p)
    return cache[p]
for r in pos.itertuples():
    p = path(r.study_id, r.image_id); ii, fx, fy, w0, h0 = get_ii(p)
    cx, cy = (r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2; x0 = int(np.clip(cx - W / 2 + rng.uniform(-W / 4, W / 4), 0, w0 - W)); y0 = int(np.clip(cy - W / 2 + rng.uniform(-W / 4, W / 4), 0, h0 - W))
    rows.append(dict(img=p, x0=x0, y0=y0, w=W, label=1, stratum="pos", group=r.study_id, image_id=r.image_id, box=[float(r.xmin), float(r.ymin), float(r.xmax), float(r.ymax)], birads=str(r.finding_birads)))
npos = len(rows); allbox = {(s, i): [(b.xmin, b.ymin, b.xmax, b.ymax) for b in g.itertuples() if not np.isnan(b.xmin)] for (s, i), g in fa.groupby(["study_id", "image_id"])}
pos_imgs = pos[["study_id", "image_id"]].drop_duplicates().itertuples(index=False)
pos_img_list = list(pos_imgs); n_same = 0
for k in range(npos):
    for _ in range(a.neg_same):
        for _try in range(300):
            s, i = pos_img_list[rng.integers(len(pos_img_list))]; p = path(s, i); ii, fx, fy, w0, h0 = get_ii(p)
            x0, y0 = int(rng.integers(0, w0 - W + 1)), int(rng.integers(0, h0 - W + 1))
            if coverage(ii, fx, fy, x0, y0) >= 0.8 and not hits(allbox.get((s, i), []), x0, y0):
                rows.append(dict(img=p, x0=x0, y0=y0, w=W, label=0, stratum="same", group=s, image_id=i, box=None, birads="")); n_same += 1; break
pos_studies = set(pos.study_id)
bad = set(map(tuple, fa[fa.finding_categories.astype(str).str.replace("'", "").str.replace("[", "").str.replace("]", "").str.strip() != "No Finding"][["study_id", "image_id"]].itertuples(index=False, name=None)))
norm = ba[ba.breast_birads.astype(str).isin(["BI-RADS 1", "BI-RADS 2"]) & ~ba.study_id.isin(pos_studies)].copy()
norm = norm[[(s, i) not in bad for s, i in zip(norm.study_id, norm.image_id)]].sample(frac=1.0, random_state=2)
n_norm = 0; it = norm.itertuples()
while n_norm < npos * a.neg_norm:
    r = next(it); p = path(r.study_id, r.image_id)
    if not os.path.exists(p): continue
    ii, fx, fy, w0, h0 = cover_ii(p)
    for _try in range(100):
        x0, y0 = int(rng.integers(0, w0 - W + 1)), int(rng.integers(0, h0 - W + 1))
        if coverage(ii, fx, fy, x0, y0) >= 0.8:
            rows.append(dict(img=p, x0=x0, y0=y0, w=W, label=0, stratum="norm", group=r.study_id, image_id=r.image_id, box=None, birads=str(r.breast_birads))); n_norm += 1; break
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(rows, open(a.out, "w"))
import collections; print("windows", len(rows), dict(collections.Counter(x["stratum"] for x in rows)), "| groups", len({x["group"] for x in rows}), "| positive studies", len(pos_studies))
