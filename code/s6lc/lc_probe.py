#!/usr/bin/env python3
"""s6lc (paper 4, question 1): linear probes on hidden-state features (class-balanced L2-regularised logistic regression, objective C * sum_i w_i logloss_i + 0.5 ||w||^2 as in scikit-learn, fitted on the GPU with L-BFGS, 150 iterations).
For mass, calc and asymmetry: grid pooling {img, last} x layer {0,8,...,36} x C {0.001, 0.01, 0.1}; the configuration is chosen on the validation-sample (VS) AUC. Saves the scores of every test row for (a) the chosen configuration ('best'),
(b) the layer-0 image-mean probe ('lin', the probe on the frozen vision tower output), and the VS AUC of the whole grid.
usage: lc_probe.py --feats f1.npz[,f2.npz] --train rows.json --vs val_sample.json --test test_embed.json --meta meta.json --out out.npz --grid out_grid.csv"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument("--feats", required=True); ap.add_argument("--train", required=True); ap.add_argument("--vs", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--meta", required=True); ap.add_argument("--out", required=True); ap.add_argument("--grid", default=""); ap.add_argument("--iters", type=int, default=150)
a = ap.parse_args()
FIND = ["mass", "calc", "asymmetry"]; CS = [0.001, 0.01, 0.1]
meta = json.load(open(a.meta)); dev = torch.device("cuda")
sid, FI, FL = [], [], []
for p in a.feats.split(","):
    z = np.load(p); sid += [str(x) for x in z["sample_id"]]; FI.append(z["F_img"]); FL.append(z["F_last"]); layers = [int(x) for x in z["layers"]]
FI, FL = (FI[0] if len(FI) == 1 else np.concatenate(FI)), (FL[0] if len(FL) == 1 else np.concatenate(FL)); ix = {s: i for i, s in enumerate(sid)}
tr_ids, vs_ids, te_ids = json.load(open(a.train)), json.load(open(a.vs)), json.load(open(a.test))
for nm, L in (("train", tr_ids), ("vs", vs_ids), ("test", te_ids)):
    miss = [s for s in L if s not in ix]; assert not miss, (nm, len(miss), miss[:2])
tr, vs, te = [np.array([ix[s] for s in L]) for L in (tr_ids, vs_ids, te_ids)]
Y = {f: {nm: np.array([meta[s]["y"][j] for s in L]) for nm, L in (("tr", tr_ids), ("vs", vs_ids), ("te", te_ids))} for j, f in enumerate(FIND)}
def fit(X, y, C):
    n, d = X.shape; w = torch.zeros(d, device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
    npos = float(y.sum()); sw = torch.where(y == 1, n / (2 * npos), n / (2 * (n - npos)))
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=a.iters, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)
    def closure():
        opt.zero_grad(); z = X @ w + b; loss = C * (sw * Fn.binary_cross_entropy_with_logits(z, y, reduction="none")).sum() + 0.5 * (w * w).sum(); loss.backward(); return loss
    opt.step(closure); return w.detach(), b.detach()
def auc(sc, y): return float(roc_auc_score(y, sc)) if len(set(y.tolist())) == 2 else float("nan")
rows = []; best = {f: dict(auc=-1) for f in FIND}; lin = {f: dict(auc=-1) for f in FIND}; t0 = time.time()
for pool, F in (("img", FI), ("last", FL)):
    for li, layer in enumerate(layers):
        if pool == "last" and layer == 0: continue            # the last prompt token carries no image information at the embedding layer
        X = torch.from_numpy(F[:, li, :]).to(dev).float()
        for fi, f in enumerate(FIND):
            ktr = Y[f]["tr"] >= 0; kvs = Y[f]["vs"] >= 0
            Xtr = X[torch.from_numpy(tr[ktr]).to(dev)]; mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd
            ytr = torch.from_numpy(Y[f]["tr"][ktr]).to(dev).float()
            Xvs = (X[torch.from_numpy(vs[kvs]).to(dev)] - mu) / sd; Xte = (X[torch.from_numpy(te).to(dev)] - mu) / sd
            for C in CS:
                w, b = fit(Xtr, ytr, C); a_vs = auc((Xvs @ w + b).cpu().numpy(), Y[f]["vs"][kvs]); rows.append(dict(finding=f, pool=pool, layer=layer, C=C, auc_vs=a_vs))
                if a_vs > best[f]["auc"]: best[f] = dict(auc=a_vs, pool=pool, layer=layer, C=C, score=(Xte @ w + b).cpu().numpy())
                if pool == "img" and layer == 0 and a_vs > lin[f]["auc"]: lin[f] = dict(auc=a_vs, pool=pool, layer=layer, C=C, score=(Xte @ w + b).cpu().numpy())
        del X; torch.cuda.empty_cache()
out = dict(test_sid=np.array(te_ids), cfg=np.array(json.dumps({f: {k: v for k, v in best[f].items() if k != "score"} for f in FIND} | {"lin_" + f: {k: v for k, v in lin[f].items() if k != "score"} for f in FIND})))
for f in FIND: out["best_" + f] = best[f]["score"]; out["lin_" + f] = lin[f]["score"]
np.savez(a.out, **out)
if a.grid:
    import pandas as pd; pd.DataFrame(rows).to_csv(a.grid, index=False)
print("done", round(time.time() - t0), {f: (best[f]["pool"], best[f]["layer"], best[f]["C"], round(best[f]["auc"], 3)) for f in FIND})
