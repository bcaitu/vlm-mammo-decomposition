#!/usr/bin/env python3
"""s6lc (paper 4, question 1): trained transformer head on the FROZEN vision-tower tokens of Qwen3-VL-8B (layer-0 image tokens = vision tower + merger output, pooled 2x2, both views).
Head: LayerNorm -> Linear(4096, 512) + view / row / column embeddings -> [CLS] + 2 pre-norm transformer layers (8 heads, ff 1024, dropout 0.1) -> Linear to (mass, calc, asymmetry, distortion).
Loss: masked binary cross-entropy (unknown labels ignored), pos_weight = n_neg / n_pos (<= 20). AdamW lr 3e-4, wd 0.05, 1 epoch warm-up + cosine over <= 30 epochs, batch 64, bf16. Epoch chosen on the VS mean AUC (mass, calc, asymmetry), patience 6.
usage: lc_tfhead.py --tokens runs_lc/A/tokens --train rows.json --vs val_sample.json --test test_embed.json --meta meta.json --out out.npz --seed 0"""
import argparse, glob, json, math, os, re, sys, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument("--tokens", required=True); ap.add_argument("--train", required=True); ap.add_argument("--vs", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--meta", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=6)
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--max_rows", type=int, default=0)
a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda")
FIND = ["mass", "calc", "asymmetry", "distortion"]; meta = json.load(open(a.meta))
info, mm, DIM = {}, {}, 4096
for f in sorted(glob.glob(os.path.join(a.tokens, "index_shard*.json"))):
    k = int(re.search(r"shard(\d+)", f).group(1)); j = json.load(open(f)); DIM = j["dim"]
    mm[k] = np.memmap(os.path.join(a.tokens, f"tokens_shard{k}.bin"), dtype=np.float16, mode="r").reshape(-1, DIM)
    for sid, off, n, mt in j["rows"]: info[sid] = (k, off, n, mt)
tr_ids, vs_ids, te_ids = json.load(open(a.train)), json.load(open(a.vs)), json.load(open(a.test))
if a.max_rows: tr_ids = tr_ids[: a.max_rows]
for nm, L in (("train", tr_ids), ("vs", vs_ids), ("test", te_ids)):
    miss = [s for s in L if s not in info]; assert not miss, (nm, len(miss), miss[:2])
def pos_of(mt):
    v, r, c = [], [], []
    for vi, (gh, gw) in enumerate(mt):
        v += [vi] * (gh * gw); r += np.repeat(np.arange(gh), gw).tolist(); c += np.tile(np.arange(gw), gh).tolist()
    return np.array(v), np.minimum(np.array(r), 63), np.minimum(np.array(c), 63)
class DS(torch.utils.data.Dataset):
    def __init__(s, ids): s.ids = ids
    def __len__(s): return len(s.ids)
    def __getitem__(s, i):
        k, off, n, mt = info[s.ids[i]]; v, r, c = pos_of(mt); assert len(v) == n
        return torch.from_numpy(np.array(mm[k][off: off + n])), torch.from_numpy(v), torch.from_numpy(r), torch.from_numpy(c), torch.tensor(meta[s.ids[i]]["y"]), i
def coll(b):
    n = max(x[0].shape[0] for x in b); B = len(b); X = torch.zeros(B, n, DIM, dtype=torch.float16); V = torch.zeros(B, n, dtype=torch.long); R = torch.zeros_like(V); Cc = torch.zeros_like(V); P = torch.ones(B, n, dtype=torch.bool)
    for i, (x, v, r, c, y, _) in enumerate(b): m = x.shape[0]; X[i, :m] = x; V[i, :m] = v; R[i, :m] = r; Cc[i, :m] = c; P[i, :m] = False
    return X, V, R, Cc, P, torch.stack([x[4] for x in b]), torch.tensor([x[5] for x in b])
class Head(nn.Module):
    def __init__(s, d=512, L=2, H=8):
        super().__init__(); s.ln = nn.LayerNorm(DIM); s.proj = nn.Linear(DIM, d); s.ev = nn.Embedding(2, d); s.er = nn.Embedding(64, d); s.ec = nn.Embedding(64, d); s.cls = nn.Parameter(torch.zeros(1, 1, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, H, 2 * d, 0.1, batch_first=True, norm_first=True, activation="gelu"), L); s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, len(FIND)))
    def forward(s, X, V, R, Cc, P):
        h = s.proj(s.ln(X.float())) + s.ev(V) + s.er(R) + s.ec(Cc); h = torch.cat([s.cls.expand(len(h), -1, -1), h], 1)
        m = torch.cat([torch.zeros(len(h), 1, dtype=torch.bool, device=h.device), P], 1); return s.out(s.enc(h, src_key_padding_mask=m)[:, 0])
def loader(ids, shuffle): return torch.utils.data.DataLoader(DS(ids), batch_size=a.bs, shuffle=shuffle, num_workers=a.workers, collate_fn=coll, drop_last=shuffle, pin_memory=True, prefetch_factor=4, persistent_workers=shuffle)
@torch.no_grad()
def predict(model, ids):
    model.eval(); out = np.zeros((len(ids), len(FIND)), np.float32)
    for X, V, R, Cc, P, y, ix in loader(ids, False):
        with torch.autocast("cuda", dtype=torch.bfloat16): z = model(X.to(dev, non_blocking=True), V.to(dev), R.to(dev), Cc.to(dev), P.to(dev))
        out[ix.numpy()] = z.float().cpu().numpy()
    return out
Ytr = np.array([meta[s]["y"] for s in tr_ids]); Yvs = np.array([meta[s]["y"] for s in vs_ids])
pw = torch.tensor([min(20.0, float((Ytr[:, j] == 0).sum()) / max(1.0, float((Ytr[:, j] == 1).sum()))) for j in range(len(FIND))], device=dev)
def vs_auc(S):
    r = []
    for j in range(3):
        k = Yvs[:, j] >= 0; r.append(roc_auc_score(Yvs[k, j], S[k, j]) if len(set(Yvs[k, j].tolist())) == 2 else float("nan"))
    return r
model = Head().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
tl = loader(tr_ids, True); spe = len(tl); total = a.epochs * spe; warm = spe
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
best, bad, best_sd, log, t0 = -1.0, 0, None, [], time.time()
for ep in range(a.epochs):
    model.train(); tot = 0.0
    for X, V, R, Cc, P, y, _ in tl:
        y = y.to(dev); msk = (y >= 0).float()
        with torch.autocast("cuda", dtype=torch.bfloat16): z = model(X.to(dev, non_blocking=True), V.to(dev), R.to(dev), Cc.to(dev), P.to(dev))
        l = nn.functional.binary_cross_entropy_with_logits(z.float(), y.clamp(min=0).float(), pos_weight=pw, reduction="none"); loss = (l * msk).sum() / msk.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += float(loss)
    au = vs_auc(predict(model, vs_ids)); m = float(np.nanmean(au)); log.append(dict(epoch=ep, loss=tot / spe, vs_auc=au, vs_mean=m, sec=round(time.time() - t0)))
    print(json.dumps(log[-1]), flush=True)
    if m > best: best, bad, best_sd = m, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= a.patience: break
model.load_state_dict(best_sd); S = predict(model, te_ids)
np.savez(a.out, test_sid=np.array(te_ids), **{f"score_{f}": S[:, j] for j, f in enumerate(FIND)}, best_vs_mean=best, log=np.array(json.dumps(log)))
print("done", round(time.time() - t0), "best VS mean AUC", round(best, 4), "epochs", len(log), flush=True)
