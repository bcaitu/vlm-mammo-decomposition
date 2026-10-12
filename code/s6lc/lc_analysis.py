#!/usr/bin/env python3
"""s6lc analysis (paper 4, question 1): learning curves of the arms on the EMBED test split, patient-cluster bootstrap, confirmatory contrasts K1-K4.
Arms: V = LoRA VLM own score (-log P(' No')); VP = probe on the hidden states of V; A_lin / A_probe = linear probes on the base model (layer-0 image mean / best configuration on VS);
A_tf = transformer head on the frozen vision-tower tokens; D = stage-1 head (DINOv2); V_ctl = control at the top size with the 2,295-step budget; V0 = base model without fine-tuning.
AUC per finding on test rows with a known label; bootstrap: 2,000 resamples of test patients (seed 20261007); a point of a curve is the mean over seeds of the AUC, CI = percentile interval of the seed-mean over resamples.
usage: PYTHONPATH=$PWD:$PWD/analysis_embed python3 analysis_embed/lc_analysis.py --out runs_lc/analysis [--boot 2000]"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd, torch
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--sets", default="runs_lc/sets"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); SZ = ["f32", "f16", "f8", "f4", "f2", "f1"]; SEEDS = [0, 1, 2]; FIND = ["mass", "calc", "asymmetry"]; J = {f: i for i, f in enumerate(FIND)}
meta = json.load(open(f"{a.sets}/meta.json")); te = json.load(open(f"{a.sets}/test_embed.json")); n = len(te); tix = {s: i for i, s in enumerate(te)}
Y = {f: np.array([meta[s]["y"][J[f]] for s in te]) for f in FIND}; pats = np.array([meta[s]["p"] for s in te]); up, pinv = np.unique(pats, return_inverse=True)
rng = np.random.default_rng(20261007); R = a.boot
cnt = rng.multinomial(len(up), np.full(len(up), 1.0 / len(up)), size=R).astype(np.float32)          # (R, n_patients)
Wall = np.vstack([np.ones((1, len(up)), np.float32), cnt])[:, pinv]                                     # row 0 = point estimate; (R+1, n_rows)
jit = np.random.default_rng(1).random(n) * 1e-9
summ = pd.read_csv(f"{a.sets}/sets_summary.csv")
def zload(p): return np.load(p, allow_pickle=True) if os.path.exists(p) else None
def align(ids, vec):
    ix = {s: i for i, s in enumerate(ids)}; out = np.full(n, np.nan, np.float32)
    for s, i in tix.items():
        if s in ix: out[i] = vec[ix[s]]
    return out
def scores(arm, sz, seed):
    """-> {finding: array over test rows} or None."""
    try:
        if arm == "V":
            z = zload(f"runs_lc/V/{sz}_s{seed}/ext.npz")
            return None if z is None else {f: align([str(x) for x in z["sample_id"]], -z["logp_no"][:, J[f]]) for f in FIND}
        if arm == "V_ctl":
            z = zload(f"runs_lc/V/ctl_s{seed}/test.npz") if sz == "f1" else None
            return None if z is None else {f: align([str(x) for x in z["sample_id"]], -z["logp_no"][:, J[f]]) for f in FIND}
        if arm == "V_blank":
            z = zload(f"runs_lc/V/blank_s{seed}/test.npz") if sz == "f1" else None
            return None if z is None else {f: align([str(x) for x in z["sample_id"]], -z["logp_no"][:, J[f]]) for f in FIND}
        if arm == "VP":
            z = zload(f"runs_lc/V/{sz}_s{seed}/probe.npz"); return None if z is None else {f: align([str(x) for x in z["test_sid"]], z["best_" + f]) for f in FIND}
        if arm in ("A_lin", "A_probe"):
            z = zload(f"runs_lc/A/probe_{sz}_s{0 if sz == 'f1' else seed}.npz"); k = "lin_" if arm == "A_lin" else "best_"
            return None if z is None else {f: align([str(x) for x in z["test_sid"]], z[k + f]) for f in FIND}
        if arm == "A_tf":
            z = zload(f"runs_lc/A/tf_{sz}_s{seed}.npz"); return None if z is None else {f: align([str(x) for x in z["test_sid"]], z["score_" + f]) for f in FIND}
        if arm in ("D", "D_ref"):
            p = f"runs_lc/D/{sz}_s{seed}/dump_test_embed.pt" if arm == "D" else (f"runs/cb_embed_dinov2_s{seed}/dump_test_embed.pt" if sz == "f1" else "")
            if not p or not os.path.exists(p): return None
            d = torch.load(p, map_location="cpu"); return {f: align(list(d["sample_id"]), d["p_" + f][:, 1].numpy()) for f in FIND}
        if arm == "V0":
            out = None
            for k in range(5):
                z = zload(f"runs_lc/A/base_shard{k}.npz")
                if z is None: return None
                ids = [str(x) for x in z["sample_id"]]; part = {f: align(ids, -z["logp_no"][:, J[f]]) for f in FIND}
                out = part if out is None else {f: np.where(np.isnan(out[f]), part[f], out[f]) for f in FIND}
            return out
    except Exception as e:
        print("load error", arm, sz, seed, repr(e)[:150]); return None
def boot_auc(s, f):
    k = (Y[f] >= 0) & ~np.isnan(s)
    if k.sum() < 50: return None
    y = Y[f][k].astype(np.float32); sc = s[k] + jit[k]; o = np.argsort(sc); W = Wall[:, k][:, o]; ys = y[o]
    pos = W * ys; neg = W * (1 - ys); cumneg = np.cumsum(neg, axis=1) - neg
    return (pos * cumneg).sum(1) / (pos.sum(1) * neg.sum(1))
ARMS = ["V", "VP", "A_lin", "A_probe", "A_tf", "D", "D_ref", "V_ctl", "V_blank", "V0"]; AUC = {}; rows = []
for arm in ARMS:
    for sz in (["f1"] if arm in ("V_ctl", "V_blank", "V0", "D_ref") else SZ):
        for f in FIND:
            lst = []
            for sd in SEEDS:
                sc = scores(arm, sz, sd)
                if sc is None: continue
                b = boot_auc(sc[f], f)
                if b is None: continue
                lst.append(b); k = (Y[f] >= 0) & ~np.isnan(sc[f]); rows.append(dict(arm=arm, finding=f, size=sz, seed=sd, auc=float(b[0]), n_pos=int((Y[f][k] == 1).sum()), n_neg=int((Y[f][k] == 0).sum())))
            if lst: AUC[(arm, f, sz)] = np.array(lst)            # (n_seeds, R+1)
pd.DataFrame(rows).to_csv(f"{a.out}/lc_auc_by_run.csv", index=False)
def npos(f, sz):
    c = {"mass": "mass_pos", "calc": "calc_pos", "asymmetry": "asymmetry_pos"}[f]; return float(summ[summ["frac"] == sz][c].mean())
cur = []
for (arm, f, sz), A in AUC.items():
    m = A.mean(0); cur.append(dict(arm=arm, finding=f, size=sz, n_pos_train=npos(f, sz), n_seeds=len(A), auc=float(m[0]), lo=float(np.percentile(m[1:], 2.5)), hi=float(np.percentile(m[1:], 97.5)), sd_seeds=float(A[:, 0].std(ddof=1)) if len(A) > 1 else float("nan")))
pd.DataFrame(cur).to_csv(f"{a.out}/lc_curve.csv", index=False)
def vs_mean(arm, sz):
    """mean over seeds and findings of the validation-sample AUC of the arm's selected configuration (used only to choose the strongest frozen-tower read-out A*)."""
    v = []
    for sd in SEEDS:
        if arm == "A_tf":
            z = zload(f"runs_lc/A/tf_{sz}_s{sd}.npz")
            if z is not None: v.append(float(z["best_vs_mean"]))
        else:
            g = f"runs_lc/A/probe_{sz}_s{0 if sz == 'f1' else sd}_grid.csv"
            if os.path.exists(g):
                d = pd.read_csv(g); v.append(float(d.groupby("finding")["auc_vs"].max().mean()))
    return float(np.mean(v)) if v else float("nan")
def diff(a_, b_, f, sz):
    if (a_, f, sz) not in AUC or (b_, f, sz) not in AUC: return None
    return AUC[(a_, f, sz)].mean(0) - AUC[(b_, f, sz)].mean(0)
def cls(d, marg):
    lo, hi = np.percentile(d[1:], 2.5), np.percentile(d[1:], 97.5)
    if lo > marg: return "above by more than %.2f" % marg
    if hi < -marg: return "below by more than %.2f" % marg
    if lo > -marg and hi < marg: return "equivalent within +-%.2f" % marg
    if hi < marg: return "not above by more than %.2f" % marg
    return "inconclusive"
con = []; astar = {}
for sz in SZ:
    a1, a2 = vs_mean("A_probe", sz), vs_mean("A_tf", sz); astar[sz] = "A_tf" if (not np.isnan(a2) and (np.isnan(a1) or a2 >= a1)) else "A_probe"
def add(name, f, sz, d, marg):
    if d is None: return
    con.append(dict(contrast=name, finding=f, size=sz, delta=float(d[0]), lo=float(np.percentile(d[1:], 2.5)), hi=float(np.percentile(d[1:], 97.5)), margin=marg, verdict=cls(d, marg)))
for f in FIND:
    for sz in SZ:
        As = astar[sz]
        for A_ in (As,):
            add(f"K1 VP - A* ({As})", f, sz, diff("VP", A_, f, sz), 0.03)
        add("K2 V - VP", f, sz, diff("V", "VP", f, sz), 0.03)
        add("K3 V - D", f, sz, diff("V", "D", f, sz), 0.05)
        if sz == "f1": add("K3-ref V - D_ref (stage-1 runs with augmentation)", f, sz, diff("V", "D_ref", f, sz), 0.05); add("D - D_ref (cached, no augmentation vs original)", f, sz, diff("D", "D_ref", f, sz), 0.03)
        add("decomp: D - A_tf (encoder)", f, sz, diff("D", "A_tf", f, sz), 0.03)
        add("decomp: V - A_tf (read-out)", f, sz, diff("V", "A_tf", f, sz), 0.03)
        add("V - A_lin", f, sz, diff("V", "A_lin", f, sz), 0.03)
    add("step budget: V(v1, 4591) - V_ctl(2295)", f, "f1", diff("V", "V_ctl", f, "f1"), 0.03)
    add("V - V0 (fine-tuning gain, top)", f, "f1", diff("V", "V0", f, "f1"), 0.05)
    if all((("V", f, s) in AUC) for s in ("f4", "f2", "f1")):
        sl = ((AUC[("V", f, "f1")].mean(0) - AUC[("V", f, "f2")].mean(0)) + (AUC[("V", f, "f2")].mean(0) - AUC[("V", f, "f4")].mean(0))) / 2
        lo, hi = np.percentile(sl[1:], 2.5), np.percentile(sl[1:], 97.5); v = "rising (lower bound >= 0.02 per doubling)" if lo >= 0.02 else "plateau (upper bound < 0.02 per doubling)" if hi < 0.02 else "inconclusive"
        con.append(dict(contrast="K4 V gain per doubling, last two doublings", finding=f, size="f4-f1", delta=float(sl[0]), lo=float(lo), hi=float(hi), margin=0.02, verdict=v))
C = pd.DataFrame(con); C.to_csv(f"{a.out}/lc_contrasts.csv", index=False)
json.dump(dict(astar=astar, n_boot=R, arms_found=sorted({k[0] for k in AUC})), open(f"{a.out}/lc_meta.json", "w"), indent=1)
print(C[C["size"].isin(["f1", "f4-f1"])].round(3).to_string(index=False)); print("A* by size:", astar); print("curve rows", len(cur))
