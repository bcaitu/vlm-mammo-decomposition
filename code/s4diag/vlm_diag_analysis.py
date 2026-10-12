#!/usr/bin/env python3
"""Analysis of the perception-versus-verbalization diagnostic (paper 4, s4diag) on the development sample of EMBED validation breasts.
For mass, calc and asymmetry: greedy-generation sensitivity/specificity, AUC of the model's own log P(' No') score, AUC of linear probes on hidden states (layer, pooling and C chosen on the select part, reported on the test part),
AUC of the stage-1 concept head (mean of six runs). Bootstrap over patients on the test part.
usage: PYTHONPATH=$PWD:$PWD/analysis_embed python3 analysis_embed/vlm_diag_analysis.py --out runs_vlm/_diag/tables"""
import argparse, json, os, sys, warnings
import numpy as np, pandas as pd, torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_grid_analysis2 as G
from cbmammo import data as D
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--rows", default="runs_vlm/_diag/dev_rows.json"); ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--jobs", type=int, default=16); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(0)
SLOTS = ["mass", "calc", "asymmetry"]; SIDX = {"mass": 0, "calc": 1, "asymmetry": 2, "distortion": 3}; CS = [0.001, 0.01, 0.1]; V_AUC_GAP, V_SENS, P_AUC = 0.05, 0.10, 0.60
rows = json.load(open(a.rows))["dev_embed_val"]; sid = [r["sample_id"] for r in rows]; part = np.array([r["part"] for r in rows]); cluster = np.array([r["cluster"] for r in rows])
recs = {r["sample_id"]: r for r in D.load_manifest(["manifests/embed.jsonl"])}
Y = {s: np.array([1 if (recs[x]["values"] or {}).get(s) == "present" else 0 if (recs[x]["values"] or {}).get(s) == "absent" else -1 for x in sid]) for s in SLOTS}
SEEDS = [f"v1_s{k}" for k in range(3)]
gen = {}
for t in SEEDS:
    m = {}
    for l in open(f"runs_vlm/{t}/gen_diag/gen_dev_embed_val_orig.jsonl"):
        j = json.loads(l); m[j["sample_id"]] = j["slots"]
    gen[t] = [m[x] for x in sid]
FEAT = {}
for t in SEEDS + ["base"]:
    z = np.load(f"runs_vlm/_diag/feat_{t}.npz"); assert [str(x) for x in z["sample_id"]] == sid, t
    FEAT[t] = dict(last=z["F_last"], img=z["F_img"], logp=z["logp_no"], layers=[int(x) for x in z["layers"]])
S1 = {s: [] for s in SLOTS}
for arm in ("cb_embed",):
    for enc in ("dinov2", "ft2"):
        for k in range(3):
            d = torch.load(f"runs/{arm}_{enc}_s{k}/dump_val.pt", map_location="cpu"); ix = {x: i for i, x in enumerate(d["sample_id"])}
            for s in SLOTS: S1[s].append(d["p_" + s][[ix[x] for x in sid], 1].numpy())
def auc(sc, y):
    return float(roc_auc_score(y, sc)) if len(set(y)) == 2 else np.nan
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
    return res[best][1], dict(pool=pool, layer=F["layers"][li], C=C, auc_select=res[best][0]), [dict(pool=p, layer=F["layers"][l], C=c, auc_select=r[0]) for (p, l, c, _), r in zip(tasks, res)]
def sens_at_spec(sc, y, spec):
    thr = np.quantile(sc[y == 0], spec); return float((sc[y == 1] > thr).mean())
out, grid, crit = [], [], {}
for s in SLOTS:
    y = Y[s]; ok = y >= 0; te = np.where(ok & (part == "test"))[0]; yt = y[te]; j = SIDX[s]
    cnt = {p: (int(((y == 1) & (part == p)).sum()), int(((y == 0) & (part == p)).sum())) for p in ("train", "select", "test")}
    P_, info = {}, {}
    for t in SEEDS + ["base"]:
        sc, inf, g = probe(FEAT[t], y, ok); P_[t], info[t] = sc, inf
        for e in g: grid.append(dict(slot=s, model=t, **e))
    L_ = {t: -FEAT[t]["logp"][te, j] for t in SEEDS}; S_ = np.mean([p[te] for p in S1[s]], axis=0)
    pred = {t: np.array([g.get(s) == "present" for g in gen[t]])[te] for t in SEEDS}
    cl = [np.where(cluster[te] == c)[0] for c in np.unique(cluster[te])]
    def stats(ix):
        yy = yt[ix]
        if yy.sum() < 5 or (1 - yy).sum() < 5: return None
        r = dict(probe=np.mean([auc(P_[t][ix], yy) for t in SEEDS]), logit=np.mean([auc(L_[t][ix], yy) for t in SEEDS]), s1=auc(S_[ix], yy), base=auc(P_["base"][ix], yy),
                 sens=np.mean([pred[t][ix][yy == 1].mean() for t in SEEDS]), spec=np.mean([(~pred[t][ix][yy == 0]).mean() for t in SEEDS]))
        r["d_probe_s1"] = r["probe"] - r["s1"]; r["d_logit_s1"] = r["logit"] - r["s1"]; return r
    pt = stats(np.arange(len(te))); reps = [r for r in (stats(np.concatenate([cl[k] for k in rng.integers(0, len(cl), len(cl))])) for _ in range(a.boot)) if r]
    ci = lambda k: tuple(np.percentile([r[k] for r in reps], [2.5, 97.5]))
    row = dict(slot=s, n_pos_train=cnt["train"][0], n_neg_train=cnt["train"][1], n_pos_test=cnt["test"][0], n_neg_test=cnt["test"][1])
    for k in ("probe", "logit", "s1", "base", "sens", "spec", "d_probe_s1", "d_logit_s1"): row[k] = pt[k]; row[k + "_lo"], row[k + "_hi"] = ci(k)
    row["sens_at_spec90_logit"] = float(np.mean([sens_at_spec(L_[t], yt, 0.90) for t in SEEDS])); row["sens_at_spec90_probe"] = float(np.mean([sens_at_spec(P_[t], yt, 0.90) for t in SEEDS])); row["sens_at_spec90_s1"] = sens_at_spec(S_, yt, 0.90)
    row["probe_config"] = ";".join(f"{t}:{info[t]['pool']}/L{info[t]['layer']}/C{info[t]['C']}" for t in SEEDS + ["base"])
    out.append(row)
    crit[s] = dict(V=bool(pt["probe"] >= pt["s1"] - V_AUC_GAP and pt["sens"] <= V_SENS), P=bool(pt["probe"] <= P_AUC), logit_recoverable=bool(pt["logit"] >= pt["s1"] - V_AUC_GAP))
nV, nP = sum(c["V"] for c in crit.values()), sum(c["P"] for c in crit.values())
crit["verdict"] = "verbalization gap" if nV >= 2 and nP < 2 else "perception gap" if nP >= 2 and nV < 2 else "intermediate"; crit["n_V"], crit["n_P"] = nV, nP
pd.DataFrame(out).to_csv(f"{a.out}/diag_main.csv", index=False); pd.DataFrame(grid).to_csv(f"{a.out}/diag_probe_grid.csv", index=False); json.dump(crit, open(f"{a.out}/diag_criteria.json", "w"), indent=1)
print(json.dumps(crit)); print("done")
