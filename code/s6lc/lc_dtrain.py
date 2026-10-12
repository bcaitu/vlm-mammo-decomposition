#!/usr/bin/env python3
"""s6lc (paper 4, question 1): arm D = stage-1 concept head (cb_embed_dinov2 hyper-parameters, 20 epochs, patience 4, EMBED negatives resampled 1:1 per epoch) trained on the CACHED frozen DINOv2 token features of lc_dfeat.py
(no image augmentation) on a nested subset of the pooled training rows; dumps 'val' and 'test_embed' predictions in the format of train_stage1.py.
usage: PYTHONPATH=$PWD:$PWD/analysis_embed python3 analysis_embed/lc_dtrain.py --config configs/cb_embed_dinov2.yaml --feat runs_lc/Dfeat --rows runs_lc/sets/subset_s0_f16.json --seed 0 --out runs_lc/D/f16_s0 --lr 3e-4"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, yaml
from torch.utils.data import DataLoader, Dataset
from cbmammo import data as D
from cbmammo.concepts import HEADS
from cbmammo.model import build, head_weights, masked_ce
from cbmammo.train_stage1 import set_seed, class_weights, predict, val_score
ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--feat", required=True); ap.add_argument("--rows", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True); ap.add_argument("--lr", type=float); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--epochs", type=int); ap.add_argument("--max_rows", type=int, default=0); ap.add_argument("--allow_missing", action="store_true")
a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); cfg["name"] = "lc_d"; tr, dc = cfg["train"], cfg["data"]
if a.lr: tr["lr"] = a.lr
if a.epochs: tr["epochs"] = a.epochs
out = a.out; os.makedirs(out, exist_ok=True)
if os.path.exists(os.path.join(out, "run.json")): print("[skip] finished", out); raise SystemExit
set_seed(a.seed); dev = torch.device("cuda"); amp = tr.get("amp", True)
idx = json.load(open(os.path.join(a.feat, "index.json"))); row_of = {s: i for i, s in enumerate(idx["sample_id"])}; FM = np.load(os.path.join(a.feat, "feat.npy"), mmap_mode="r")
class CachedEnc(nn.Module):                         # replaces the frozen encoder: the 'images' passed to the model are the cached token grids
    def __init__(s, dim): super().__init__(); s.dim = dim; s.trainable = False
    def forward(s, f): return f.float()
class DS(Dataset):
    def __init__(s, recs): s.r = recs
    def __len__(s): return len(s.r)
    def __getitem__(s, i):
        r = s.r[i]; k = row_of[r["sample_id"]]; x = torch.from_numpy(np.array(FM[k]))
        return {"cc": x[0], "mlo": x[1], "labels": torch.tensor([r["labels"][h.name] for h in HEADS], dtype=torch.long), "view_mask": torch.tensor(idx["view_mask"][k]), "idx": i}
want = set(json.load(open(a.rows))); recs = D.load_manifest(dc["train_manifests"])
if a.allow_missing: recs = [r for r in recs if r["sample_id"] in row_of]; want = {s for s in want if s in row_of}      # smoke tests only
else: assert all((r["sample_id"] in row_of) for r in recs if r["split"] in ("train", "val") or (r["dataset"] == "embed" and r["split"] == "test")), "features missing for some rows"
train_r_full = [r for r in recs if r["split"] == "train" and r["sample_id"] in want]; assert len(train_r_full) == len(want), (len(train_r_full), len(want))
if a.max_rows: train_r_full = train_r_full[: a.max_rows]
val_r = [r for r in recs if r["split"] == "val"]; test_r = [r for r in recs if r["dataset"] == "embed" and r["split"] == "test"]
embed_neg_ratio = tr.get("embed_neg_ratio"); dataset_balanced_val = bool(tr.get("dataset_balanced_val", False))
mk = lambda rs, train: DataLoader(DS(rs), batch_size=tr["batch_size"], shuffle=train, num_workers=a.workers, collate_fn=D.collate, drop_last=train, pin_memory=True)
train_r = D.epoch_sample(train_r_full, embed_neg_ratio, seed=a.seed); tl, vl = mk(train_r, True), mk(val_r, False); val_ds = [r["dataset"] for r in val_r]
print(f"train {len(train_r_full)} (epoch sample {len(train_r)})  val {len(val_r)}  test {len(test_r)}", flush=True)
model = build(cfg); dim = model.enc.dim; model.enc = CachedEnc(dim); model = model.to(dev)                       # build() created the timm encoder only to size the projection
params = [p for p in model.parameters() if p.requires_grad]; base_lr = tr.get("lr", 3e-4)
opt = torch.optim.AdamW([{"params": params, "lr": base_lr}], lr=base_lr, weight_decay=tr.get("wd", 0.05)); steps = tr["epochs"] * max(1, len(tl)); warm = int(0.05 * steps)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1, s / max(1, steps)))))
hw = head_weights(cfg["arm"], cfg["model"].get("aux_lambda", 0.1), cfg["model"].get("target_weight", 1.0)); cw = class_weights(train_r, dev, tr.get("class_weight_power", 0.5))
best, bad, log = -1, 0, []
for ep in range(tr["epochs"]):
    if embed_neg_ratio is not None:
        train_r = D.epoch_sample(train_r_full, embed_neg_ratio, seed=a.seed * 10007 + ep); tl = mk(train_r, True)
    model.train(); t0 = time.time(); tot = 0
    for i, b in enumerate(tl):
        with torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16): o = model(b["cc"].to(dev, non_blocking=True), b["mlo"].to(dev, non_blocking=True), b["view_mask"].to(dev))
        loss, parts = masked_ce(o["logits"], b["labels"].to(dev), hw, cw); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); tot += float(loss)
    pv = predict(model, vl, dev, amp); score, per = val_score(pv, val_ds if dataset_balanced_val else None)
    log.append({"epoch": ep, "train_loss": tot / max(1, len(tl)), "n_train_epoch": len(train_r), "val_score": score, "val_f1": per, "sec": time.time() - t0}); print(json.dumps(log[-1]), flush=True)
    if score > best: best, bad = score, 0; sd = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.startswith("enc.")}
    else:
        bad += 1
        if bad >= tr.get("patience", 5): break
json.dump(log, open(os.path.join(out, "train_log.json"), "w"), indent=1); model.load_state_dict(sd, strict=False)
for name, rs in (("val", val_r), ("test_embed", test_r)):
    pr = predict(model, mk(rs, False), dev, amp); pr["sample_id"] = [rs[i]["sample_id"] for i in pr["idx"].tolist()]; torch.save(pr, os.path.join(out, f"dump_{name}.pt")); print("[dump]", name, len(rs), flush=True)
json.dump({"config": cfg, "seed": a.seed, "best_val": best, "rows": len(train_r_full), "epochs_run": len(log)}, open(os.path.join(out, "run.json"), "w"), indent=1)
