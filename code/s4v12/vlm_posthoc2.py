#!/usr/bin/env python3
"""Post hoc image-dependence diagnostics (paper 4), aggregates only: AUC of the stated BI-RADS category for benign/malignant on the original, swapped and blank inputs (first 300 rows per set), and the share of slots that change when the image is swapped."""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_grid_analysis2 as G
def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else None
def sc(rows, key): return np.array([float(r[key]["birads"]) if (r.get(key) or {}).get("birads") is not None else np.nan for r in rows])
cbr = sorted(glob.glob("runs_stage2/grid2_cb_embed_*")); out = []
for sp in G.TESTS:
    O = [jl(f"runs_vlm/v1_s{s}/gen/gen_{sp}_orig.jsonl") for s in range(3)]; S = [jl(f"runs_vlm/v1_s{s}/gen/gen_{sp}_swap.jsonl") for s in range(3)]; B = [jl(f"runs_vlm/v1_s{s}/gen/gen_{sp}_blank.jsonl") for s in range(3)]
    m = len(S[0]); lab = np.array([{"benign": 0, "malignant": 1}.get((x["values"] or {}).get("malignant"), -1) for x in O[0]])
    row = dict(split=sp, n300=m)
    chg_cat, chg_slots = [], []
    for s in range(3):
        for i in range(m):
            a, b = O[s][i]["slots"], S[s][i]["slots"]
            chg_cat.append(int(a.get("birads") != b.get("birads"))); keys = [k for k in a if k in b]; chg_slots.append(np.mean([a[k] != b[k] for k in keys]) if keys else np.nan)
    row["share_birads_changed_by_swap"] = float(np.mean(chg_cat)); row["share_slots_changed_by_swap"] = float(np.nanmean(chg_slots))
    k_all = np.where(lab >= 0)[0]
    if np.sum(lab[k_all] == 0) >= 20 and np.sum(lab[k_all] == 1) >= 20:
        row["n_pos_full"] = int(np.sum(lab == 1)); row["n_neg_full"] = int(np.sum(lab == 0))
        row["auc_full_vlm"] = float(np.mean([G.auc_fast(sc(x, "slots")[k_all], lab[k_all]) for x in O]))
        cg = [jl(f"{r}/gen_{sp}.jsonl") for r in cbr]; row["auc_full_cb"] = float(np.mean([G.auc_fast(sc(x, "slots_generated")[k_all], lab[k_all]) for x in cg if x]))
        k3 = np.where(lab[:m] >= 0)[0]
        if np.sum(lab[:m][k3] == 0) >= 10 and np.sum(lab[:m][k3] == 1) >= 10:
            row["n_pos_300"] = int(np.sum(lab[:m][k3] == 1)); row["n_neg_300"] = int(np.sum(lab[:m][k3] == 0))
            for tag, X in (("orig", O), ("swap", S), ("blank", B)):
                row["auc300_" + tag] = float(np.mean([G.auc_fast(sc(x, "slots")[:m][k3], lab[:m][k3]) for x in X]))
    out.append(row)
pd.DataFrame(out).to_csv("runs_vlm/_v12_tables/posthoc_image_dependence.csv", index=False); print("done")
