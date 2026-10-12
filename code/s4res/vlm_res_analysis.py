#!/usr/bin/env python3
"""Resolution test (paper 4, s4res): linear probes on hidden states of the BASE Qwen3-VL-8B (no adapter) for mass, calc and asymmetry at image resolutions 896, 1344 and 1792 px (long side), same development sample and probe protocol as s4diag.
Reports test-part AUC per resolution and the paired difference to 896 px (bootstrap over patients). Decision on calc and asymmetry (rule in the pre-registration).
usage: PYTHONPATH=$PWD:$PWD/analysis_embed python3 analysis_embed/vlm_res_analysis.py --out runs_vlm/_diag/tables_res"""
import argparse, json, os, sys, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbmammo import data as D
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--rows", default="runs_vlm/_diag/dev_rows.json"); ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--jobs", type=int, default=16); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(0)
SLOTS = ["mass", "calc", "asymmetry"]; CS = [0.001, 0.01, 0.1]; GAIN = 0.05; FLAT = 0.02
RES = {896: "runs_vlm/_diag/feat_base.npz", 1344: "runs_vlm/_diag/feat_base_1344.npz", 1792: "runs_vlm/_diag/feat_base_1792.npz"}
rows = json.load(open(a.rows))["dev_embed_val"]; sid = [r["sample_id"] for r in rows]; part = np.array([r["part"] for r in rows]); cluster = np.array([r["cluster"] for r in rows])
recs = {r["sample_id"]: r for r in D.load_manifest(["manifests/embed.jsonl"])}
Y = {s: np.array([1 if (recs[x]["values"] or {}).get(s) == "present" else 0 if (recs[x]["values"] or {}).get(s) == "absent" else -1 for x in sid]) for s in SLOTS}
FEAT = {}
for r_, p in RES.items():
    z = np.load(p); assert [str(x) for x in z["sample_id"]] == sid, r_
    FEAT[r_] = dict(last=z["F_last"], img=z["F_img"], layers=[int(x) for x in z["layers"]], ntok=(z["n_img_tokens"] if "n_img_tokens" in z.files else None))
def auc(sc, y): return float(roc_auc_score(y, sc)) if len(set(y)) == 2 else np.nan
def fit_cfg(X, y, tr, se, te, C):
    sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(C=C, max_iter=300, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
    return auc(clf.decision_function(sc.transform(X[se])), y[se]), clf.decision_function(sc.transform(X[te]))
def probe(F, y, ok):
    tr, se, te = [np.where(ok & (part == p))[0] for p in ("train", "select", "test")]; tasks = []
    for pool in ("last", "img"):
        for li in range(len(F["layers"])):
            X = F[pool][:, li].astype(np.float32)
            for C in CS: tasks.append((pool, li, C, X))
    res = Parallel(n_jobs=a.jobs)(delayed(fit_cfg)(X, y, tr, se, te, C) for (_, _, C, X) in tasks)
    best = int(np.nanargmax([r[0] for r in res])); pool, li, C, _ = tasks[best]
    return res[best][1], dict(pool=pool, layer=F["layers"][li], C=C, auc_select=res[best][0])
out, crit = [], {}
for s in SLOTS:
    y = Y[s]; ok = y >= 0; te = np.where(ok & (part == "test"))[0]; yt = y[te]; P_, info = {}, {}
    for r_ in RES: P_[r_], info[r_] = probe(FEAT[r_], y, ok)
    cl = [np.where(cluster[te] == c)[0] for c in np.unique(cluster[te])]
    def st(ix):
        yy = yt[ix]
        if yy.sum() < 5 or (1 - yy).sum() < 5: return None
        return {r_: auc(P_[r_][ix], yy) for r_ in RES}
    pt = st(np.arange(len(te))); reps = [r for r in (st(np.concatenate([cl[k] for k in rng.integers(0, len(cl), len(cl))])) for _ in range(a.boot)) if r]
    row = dict(slot=s, n_pos_test=int(yt.sum()), n_neg_test=int((1 - yt).sum()))
    for r_ in RES:
        row[f"auc_{r_}"] = pt[r_]; row[f"auc_{r_}_lo"], row[f"auc_{r_}_hi"] = np.percentile([x[r_] for x in reps], [2.5, 97.5])
        row[f"cfg_{r_}"] = f"{info[r_]['pool']}/L{info[r_]['layer']}/C{info[r_]['C']}"
        if r_ != 896:
            d = [x[r_] - x[896] for x in reps]; row[f"gain_{r_}"] = pt[r_] - pt[896]; row[f"gain_{r_}_lo"], row[f"gain_{r_}_hi"] = np.percentile(d, [2.5, 97.5])
    out.append(row); crit[s] = dict(gain_1792=row["gain_1792"], gain_1344=row["gain_1344"])
for r_ in (1344, 1792):
    z = FEAT[r_]["ntok"]; crit[f"mean_image_tokens_{r_}"] = float(z.mean()) if z is not None else None
crit["mean_image_tokens_896"] = None
g = {s: crit[s]["gain_1792"] for s in ("calc", "asymmetry")}
crit["verdict"] = "resolution is a bottleneck" if all(v >= GAIN for v in g.values()) else "no gain from resolution" if all(v <= FLAT for v in g.values()) else "unclear"
pd.DataFrame(out).to_csv(f"{a.out}/res_main.csv", index=False); json.dump(crit, open(f"{a.out}/res_criteria.json", "w"), indent=1); print(json.dumps(crit)); print("done")
