#!/usr/bin/env python3
"""s6lc (paper 4, question 1): arm D = the stage-1 concept head (cb_embed_dinov2 recipe: frozen DINOv2 ViT-B/14, 2-layer fusion head, 20 epochs, patience 4, EMBED negatives resampled 1:1 each epoch) retrained
on a nested subset of the pooled training rows. Same code path as cbmammo.train_stage1.main except: the training rows are restricted to --rows, and only the 'val' and 'test_embed' predictions are dumped.
usage: PYTHONPATH=$PWD:$PWD/analysis_embed python3 analysis_embed/lc_stage1.py --config configs/cb_embed_dinov2.yaml --rows runs_lc/sets/subset_s0_f16.json --seed 0 --out runs_lc/D/f16_s0 [--lr 3e-4]"""
import argparse, json, math, os, random, time
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from cbmammo import data as D
from cbmammo.concepts import HEADS, IGNORE
from cbmammo.model import build, head_weights, masked_ce
from cbmammo.train_stage1 import set_seed, device_of, class_weights, predict, val_score
ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--rows", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
ap.add_argument("--lr", type=float); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--max_rows", type=int, default=0); ap.add_argument("--epochs", type=int); ap.add_argument("--max_eval", type=int, default=0)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config)); cfg["name"] = "lc_d"; tr, dc = cfg["train"], cfg["data"]; tr["workers"] = a.workers
if a.lr: tr["lr"] = a.lr
if a.epochs: tr["epochs"] = a.epochs
out = a.out; os.makedirs(out, exist_ok=True)
if os.path.exists(os.path.join(out, "run.json")): print("[skip] finished", out); raise SystemExit
open(os.path.join(out, "_running"), "w").write(str(os.getpid())); set_seed(a.seed); dev = device_of(cfg); amp = tr.get("amp", True)
want = set(json.load(open(a.rows)))
recs = D.load_manifest(dc["train_manifests"])
train_r_full = [r for r in recs if r["split"] == "train" and r["sample_id"] in want]; assert len(train_r_full) == len(want), (len(train_r_full), len(want))
if a.max_rows: train_r_full = train_r_full[: a.max_rows]
val_r = [r for r in recs if r["split"] == "val"]; test_r = [r for r in recs if r["dataset"] == "embed" and r["split"] == "test"]
if a.max_eval: val_r, test_r = val_r[: a.max_eval], test_r[: a.max_eval]
embed_neg_ratio = tr.get("embed_neg_ratio"); dataset_balanced_val = bool(tr.get("dataset_balanced_val", False)); size = tuple(dc.get("size", [1024, 640]))
mk = lambda rs, train: DataLoader(D.BreastDataset(rs, size, train, dc.get("clahe", True), dc.get("cache_dir")), batch_size=tr["batch_size"], shuffle=train, num_workers=tr.get("workers", 4), collate_fn=D.collate, drop_last=train, pin_memory=dev.type == "cuda")
train_r = D.epoch_sample(train_r_full, embed_neg_ratio, seed=a.seed); tl, vl = mk(train_r, True), mk(val_r, False); val_ds = [r["dataset"] for r in val_r]
print(f"train {len(train_r_full)} (epoch sample {len(train_r)})  val {len(val_r)}  test {len(test_r)}  device {dev}", flush=True)
model = build(cfg).to(dev); params = [p for p in model.parameters() if p.requires_grad]
enc_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("enc.")]; oth_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("enc.")]
base_lr = tr.get("lr", 3e-4); groups = [{"params": oth_p, "lr": base_lr}] + ([{"params": enc_p, "lr": base_lr * tr.get("enc_lr_mult", 0.1)}] if enc_p else [])
opt = torch.optim.AdamW(groups, lr=base_lr, weight_decay=tr.get("wd", 0.05)); steps = tr["epochs"] * max(1, len(tl)); warm = int(0.05 * steps)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1, s / max(1, steps)))))
hw = head_weights(cfg["arm"], cfg["model"].get("aux_lambda", 0.1), cfg["model"].get("target_weight", 1.0)); cw = class_weights(train_r, dev, tr.get("class_weight_power", 0.5))
best, bad, log = -1, 0, []
for ep in range(tr["epochs"]):
    if embed_neg_ratio is not None:
        train_r = D.epoch_sample(train_r_full, embed_neg_ratio, seed=a.seed * 10007 + ep); tl = mk(train_r, True)
    model.train(); t0 = time.time(); tot = 0
    for i, b in enumerate(tl):
        with torch.autocast(dev.type, enabled=amp and dev.type == "cuda", dtype=torch.bfloat16): o = model(b["cc"].to(dev), b["mlo"].to(dev), b["view_mask"].to(dev))
        loss, parts = masked_ce(o["logits"], b["labels"].to(dev), hw, cw); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); tot += float(loss)
        if i % tr.get("log_every", 100) == 0: print(f"ep {ep} it {i}/{len(tl)} loss {float(loss):.4f}", flush=True)
    pv = predict(model, vl, dev, amp); score, per = val_score(pv, val_ds if dataset_balanced_val else None)
    log.append({"epoch": ep, "train_loss": tot / max(1, len(tl)), "n_train_epoch": len(train_r), "val_score": score, "val_f1": per, "sec": time.time() - t0}); print(json.dumps(log[-1]), flush=True)
    if score > best:
        best, bad = score, 0; torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("enc.m.") or model.enc.trainable}, os.path.join(out, "best.pt"))
    else:
        bad += 1
        if bad >= tr.get("patience", 5): break
json.dump(log, open(os.path.join(out, "train_log.json"), "w"), indent=1)
model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=dev), strict=False)
for name, rs in (("val", val_r), ("test_embed", test_r)):
    pr = predict(model, mk(rs, False), dev, amp); pr["sample_id"] = [rs[i]["sample_id"] for i in pr["idx"].tolist()]; torch.save(pr, os.path.join(out, f"dump_{name}.pt")); print("[dump]", name, len(rs), flush=True)
json.dump({"config": cfg, "seed": a.seed, "best_val": best, "rows": len(train_r_full)}, open(os.path.join(out, "run.json"), "w"), indent=1)
os.remove(os.path.join(out, "_running"))
