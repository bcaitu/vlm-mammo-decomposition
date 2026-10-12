#!/usr/bin/env python3
"""s7zoom: frozen Qwen3-VL-8B vision tower + merger features of crops at three resolutions (the 'resolution ladder'): downsample factors 1, 2, 4 of a WxW native-pixel window
(4x ~ the pixel pitch of the usual 896-px whole-image view of a 3518-px VinDr image: 3.93x; 2x ~ the 1792-px view). Intensity: 1-99.5 percentile window of the WHOLE image, as in the whole-image cache.
Crops are antialiased with LANCZOS (as in the cache) and are NOT upsampled: the processor's minimum size is lowered so that 128x128 stays 128x128.
Saves per window and level: mean and max over merged image tokens (4096-d), number of tokens; plus six low-level statistics (columns 0-3 low-frequency statistics of the ds4 crop = shortcut baseline; 4-5 high-frequency (Laplacian) energy at native and ds4 resolution).
usage: CUDA_VISIBLE_DEVICES=k python3 s7_crop_feats.py --windows runs_zoom/windows.json --out runs_zoom/feats.npz [--limit 24] [--bs 16]"""
import argparse, json, os, time, collections
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
ap = argparse.ArgumentParser(); ap.add_argument("--windows", required=True); ap.add_argument("--out", required=True); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--levels", default="1,2,4"); a = ap.parse_args()
LV = [int(x) for x in a.levels.split(",")]; wins = json.load(open(a.windows))
if a.limit: rng = np.random.default_rng(0); wins = [wins[i] for i in sorted(rng.choice(len(wins), min(a.limit, len(wins)), replace=False))]
proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct"); ip = proc.image_processor
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16).cuda().eval(); dev = next(model.parameters()).device
SIZE = {"shortest_edge": 4096, "longest_edge": 16777216}
def feats(imgs):
    o = ip(images=imgs, size=SIZE, return_tensors="pt"); thw = o["image_grid_thw"]; pv = o["pixel_values"].to(dev, torch.bfloat16)
    with torch.no_grad(): out = model.model.get_image_features(pixel_values=pv, image_grid_thw=thw.to(dev))
    e = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out[0]
    if isinstance(e, (list, tuple)): parts = [x for x in e]
    else: parts = list(torch.split(e, (thw.prod(-1) // 4).tolist(), 0))
    return [(p.float().mean(0).cpu().numpy(), p.float().amax(0).cpu().numpy(), p.shape[0]) for p in parts]
by_img = collections.defaultdict(list)
for k, w in enumerate(wins): by_img[w["img"]].append(k)
N = len(wins); M = {L: np.zeros((N, 4096), np.float16) for L in LV}; X = {L: np.zeros((N, 4096), np.float16) for L in LV}; NT = {L: np.zeros(N, np.int32) for L in LV}; ST = np.zeros((N, 6), np.float32)
buf = {L: [] for L in LV}; t0 = time.time(); done = 0
def flush(L):
    if not buf[L]: return
    idx = [k for k, _ in buf[L]]; res = feats([im for _, im in buf[L]])
    for k, (m, x, n) in zip(idx, res): M[L][k] = m; X[L][k] = x; NT[L][k] = n
    buf[L] = []
for p, ks in by_img.items():
    im0 = Image.open(p).convert("L"); arr = np.asarray(im0).astype(np.float32); lo, hi = np.percentile(arr[::2, ::2], 1), np.percentile(arr[::2, ::2], 99.5); nrm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    for k in ks:
        w = wins[k]; c = (nrm[w["y0"]:w["y0"] + w["w"], w["x0"]:w["x0"] + w["w"]] * 255).astype(np.uint8); cf = c.astype(np.float32)
        lap = lambda q: np.abs(4 * q[1:-1, 1:-1] - q[:-2, 1:-1] - q[2:, 1:-1] - q[1:-1, :-2] - q[1:-1, 2:]).mean()
        pil = Image.fromarray(c); c4 = np.asarray(pil.resize((w["w"] // 4, w["w"] // 4), Image.LANCZOS)).astype(np.float32)
        ST[k] = [c4.mean(), c4.std(), np.percentile(c4, 5), np.percentile(c4, 95), lap(cf), lap(c4)]   # 0-3: low-frequency statistics of the ds4 crop (shortcut baseline); 4-5: high-frequency energy at native and at ds4
        for L in LV:
            im = pil if L == 1 else pil.resize((w["w"] // L, w["w"] // L), Image.LANCZOS)
            buf[L].append((k, im.convert("RGB")))
            if len(buf[L]) >= a.bs: flush(L)
    done += len(ks)
    if (done // 50) != ((done - len(ks)) // 50): print(f"{done}/{N} windows {time.time() - t0:.0f}s", flush=True)
for L in LV: flush(L)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
np.savez(a.out, idx=np.arange(N), label=np.array([w["label"] for w in wins]), stratum=np.array([w["stratum"] for w in wins]), group=np.array([w["group"] for w in wins]), stats=ST,
         **{f"mean_ds{L}": M[L] for L in LV}, **{f"max_ds{L}": X[L] for L in LV}, **{f"ntok_ds{L}": NT[L] for L in LV}, box_size=np.array([(w["box"][2] - w["box"][0]) if w["box"] else -1 for w in wins]))
print("saved", a.out, N, "windows; tokens per level", {L: int(NT[L][0]) for L in LV})
