#!/usr/bin/env python3
"""s7zoom: patient-grouped cross-validated linear probes on the crop features of each resolution level; patient-cluster bootstrap for AUC and for the contrast native (ds1) minus ds4.
Primary: mean pooling. Secondary: max pooling. Shortcut baseline: logistic regression on 4 low-frequency statistics of the ds4 crop (mean, sd, 5th and 95th percentile); reference: single-statistic AUC of high-frequency (Laplacian) energy at native and ds4 resolution (not part of the rule).
Decision rule (pre-registered in s7zoom_prereg_ru.md): GO if lower 95% bound of AUC(ds1)-AUC(ds4) >= 0.05 and AUC(ds1) >= 0.80 and shortcut AUC <= 0.65; NO-GO if upper bound of the difference < 0.03 or AUC(ds1) < 0.65; else UNCLEAR.
usage: python3 s7_probe.py --feats runs_zoom/feats.npz --out runs_zoom/probe_results.json [--boot 2000]"""
import argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument("--feats", required=True); ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=20261008); a = ap.parse_args()
z = np.load(a.feats, allow_pickle=True); y = z["label"].astype(int); g = z["group"].astype(str); stratum = z["stratum"].astype(str); N = len(y)
CS = [1e-4, 1e-3, 1e-2, 1e-1]
def oof_scores(X, seed=0):
    out = np.zeros(N); chosen = []
    for tr, te in GroupKFold(5).split(X, y, g):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te]); best, bc = -1, CS[0]
        for C in CS:
            s = []
            for itr, iva in GroupKFold(3).split(Xtr, y[tr], g[tr]):
                m = LogisticRegression(C=C, class_weight="balanced", max_iter=300).fit(Xtr[itr], y[tr][itr]); s.append(roc_auc_score(y[tr][iva], m.decision_function(Xtr[iva])))
            if np.mean(s) > best: best, bc = np.mean(s), C
        m = LogisticRegression(C=bc, class_weight="balanced", max_iter=300).fit(Xtr, y[tr]); out[te] = m.decision_function(Xte); chosen.append(bc)
    return out, chosen
res = {"n": int(N), "n_pos": int(y.sum()), "n_groups": int(len(set(g))), "levels": {}}; S = {}
for pool in ("mean", "max"):
    for L in (1, 2, 4):
        k = f"{pool}_ds{L}"
        if k not in z.files: continue
        X = z[k].astype(np.float32); s, ch = oof_scores(X); S[k] = s; res["levels"][k] = dict(auc=float(roc_auc_score(y, s)), C=[float(c) for c in ch], ntok=int(z[f"ntok_ds{L}"][0]),
            auc_vs_same=float(roc_auc_score(y[(stratum == "pos") | (stratum == "same")], s[(stratum == "pos") | (stratum == "same")])), auc_vs_norm=float(roc_auc_score(y[(stratum == "pos") | (stratum == "norm")], s[(stratum == "pos") | (stratum == "norm")])))
        print(k, res["levels"][k]["auc"], flush=True)
st, _ = oof_scores(z["stats"][:, :4].astype(np.float32)); res["shortcut_auc"] = float(roc_auc_score(y, st)); S["shortcut"] = st   # low-frequency statistics only
res["hf_energy_auc_native"] = float(roc_auc_score(y, z["stats"][:, 4])); res["hf_energy_auc_ds4"] = float(roc_auc_score(y, z["stats"][:, 5]))   # reference: single high-frequency statistic, no cross-validation
rng = np.random.default_rng(a.seed); ug = np.array(sorted(set(g))); idx_by = {u: np.where(g == u)[0] for u in ug}; boots = {k: [] for k in S}; pairs = {"mean_ds1-mean_ds4": ("mean_ds1", "mean_ds4"), "mean_ds2-mean_ds4": ("mean_ds2", "mean_ds4"), "mean_ds1-mean_ds2": ("mean_ds1", "mean_ds2"), "max_ds1-max_ds4": ("max_ds1", "max_ds4")}
for b in range(a.boot):
    pick = rng.choice(ug, len(ug), replace=True); ii = np.concatenate([idx_by[u] for u in pick]); yy = y[ii]
    if yy.min() == yy.max(): continue
    for k in S: boots[k].append(roc_auc_score(yy, S[k][ii]))
res["ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items() if len(v)}
res["diff"] = {}
for name, (p, q) in pairs.items():
    if p in S and q in S:
        d = np.array(boots[p]) - np.array(boots[q]); res["diff"][name] = dict(delta=float(res["levels"][p]["auc"] - res["levels"][q]["auc"]), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
if "mean_ds1-mean_ds4" in res["diff"]:
    d = res["diff"]["mean_ds1-mean_ds4"]; a1 = res["levels"]["mean_ds1"]["auc"]
    res["verdict"] = "GO" if (d["lo"] >= 0.05 and a1 >= 0.80 and res["shortcut_auc"] <= 0.65) else "NO-GO" if (d["hi"] < 0.03 or a1 < 0.65) else "UNCLEAR"
    res["verdict_inputs"] = dict(delta_lo=d["lo"], delta_hi=d["hi"], auc_native=a1, shortcut_auc=res["shortcut_auc"])
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps({k: res[k] for k in ("verdict", "verdict_inputs", "diff", "shortcut_auc", "hf_energy_auc_native", "hf_energy_auc_ds4") if k in res}, indent=1))
