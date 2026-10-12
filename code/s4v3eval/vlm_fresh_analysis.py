#!/usr/bin/env python3
"""Analysis of the second VLM study (paper 4, s4v3eval): lesion-detection balanced accuracy, BI-RADS balanced recall, AUC of the stated category for malignancy, descriptor checks and image-dependence controls.
--mode fresh : the pre-registered analysis on the fresh rows (runs_vlm/_fresh/fresh_rows.json; VLM in runs_vlm/v1_s*/gen_fresh, decoders in runs_vlm/_fresh/dec/*).
--mode seen  : exploratory run of the same code on the grid-2 evaluation subsample that was already seen (all eight sets).
usage: PYTHONPATH=$PWD:$PWD/analysis_embed python3 analysis_embed/vlm_fresh_analysis.py --mode fresh --out runs_vlm/_fresh/tables"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_grid_analysis2 as G
ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["fresh", "seen"], required=True); ap.add_argument("--out", required=True)
ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--rows", default="runs_vlm/_fresh/fresh_rows.json"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
LES = ["mass", "calc", "asymmetry", "distortion"]; DESC = dict(G.PARENT); EQ = 0.03; MINC = 30; OMIT_MAX = 0.10; N_V2 = 300
DEC_SETS = ["test_embed", "test_embed_fuji", "test_embed_ge", "test_vindr"]; AUC_SETS = ["test_cmmd"]
ALLSETS = ["test_cbis_ddsm", "test_cdd_cesm", "test_cmmd", "test_embed", "test_embed_fuji", "test_embed_ge", "test_inbreast", "test_vindr"]
SETS = DEC_SETS + AUC_SETS if a.mode == "fresh" else ALLSETS
def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else None
pct = G.pct; rng = np.random.default_rng(a.seed)
if a.mode == "seen":
    from cbmammo import data as D
    recs = {r["sample_id"]: r for r in D.load_manifest(["manifests/cbis_ddsm.jsonl", "manifests/vindr.jsonl", "manifests/embed.jsonl", "manifests/cmmd.jsonl", "manifests/cdd_cesm.jsonl", "manifests/inbreast.jsonl"])}
else: fresh = json.load(open(a.rows))
STAGE1 = [f"{arm}_embed_{enc}_s{s}" for arm in ("cb", "opq") for enc in ("dinov2", "ft2") for s in range(3)]
def decoder_file(run, sp): return (f"runs_stage2/grid2_{run}/gen_{sp}.jsonl" if a.mode == "seen" else f"runs_vlm/_fresh/dec/{run}/gen_{sp}.jsonl")
def vlm_file(s, sp, var): return (f"runs_vlm/v1_s{s}/gen/gen_{sp}_{var}.jsonl" if a.mode == "seen" else f"runs_vlm/v1_s{s}/gen_fresh/gen_{sp}_{var}.jsonl")
def to_score(s):
    try: return float(s.get("birads"))
    except Exception: return np.nan
def model_arrays(slots, refs):
    return dict(les={h: np.array([s.get(h) == "present" for s in slots]) for h in LES}, cat=np.array([str(s.get("birads")) if s.get("birads") is not None else "NA" for s in slots]), score=np.array([to_score(s) for s in slots]), slots=slots)
def bacc(P, y, w):
    pos, neg = (y == 1) * w, (y == 0) * w
    return 0.5 * ((P * pos).sum(1) / pos.sum() + ((~P) * neg).sum(1) / neg.sum())
rows_main, rows_seed, rows_slot, rows_desc, rows_v2, crit_sets = [], [], [], [], [], {}
for sp in SETS:
    base = jl(vlm_file(0, sp, "orig")); n = len(base); sid = [r["sample_id"] for r in base]; refs = [r["values"] or {} for r in base]
    if a.mode == "fresh":
        assert [r["sample_id"] for r in fresh[sp]] == sid, sp; cluster = np.array([r["cluster"] for r in fresh[sp]])
    else: pats = {}; cluster = np.array([pats.setdefault(recs[s]["patient_id"], len(pats)) for s in sid])
    cl_list = [np.where(cluster == c)[0] for c in np.unique(cluster)]
    names, slots_by = [], []
    for s in range(3):
        rr = jl(vlm_file(s, sp, "orig")); assert [r["sample_id"] for r in rr] == sid; names.append(f"vlm_s{s}"); slots_by.append([r["slots"] for r in rr])
    for run in STAGE1:
        rr = jl(decoder_file(run, sp)); m = {r["sample_id"]: r for r in rr}; assert set(sid) <= set(m), (sp, run); names.append(run); slots_by.append([m[x]["slots_generated"] for x in sid])
    M = len(names); arr = [model_arrays(s, refs) for s in slots_by]
    iv = [i for i, nme in enumerate(names) if nme.startswith("vlm")]; icb = [i for i, nme in enumerate(names) if nme.startswith("cb_")]; iop = [i for i, nme in enumerate(names) if nme.startswith("opq_")]
    Y = {h: np.array([1 if r.get(h) == "present" else 0 if r.get(h) == "absent" else -1 for r in refs]) for h in LES}
    inc_les = [h for h in LES if (Y[h] == 1).sum() >= MINC and (Y[h] == 0).sum() >= MINC]
    refcat = np.array([str(r.get("birads")) if r.get("birads") is not None else "NA" for r in refs]); classes = [k for k in sorted(set(refcat) - {"NA"}) if (refcat == k).sum() >= MINC]
    lab = np.array([{"benign": 0, "malignant": 1}.get(r.get("malignant"), -1) for r in refs]); has_auc = (lab == 1).sum() >= MINC and (lab == 0).sum() >= MINC; keep = np.where(lab >= 0)[0]
    P = {h: np.stack([x["les"][h] for x in arr]) for h in LES}; CATP = np.stack([x["cat"] for x in arr]); SC = np.stack([x["score"] for x in arr])
    def metrics(w, idx=None):
        out = {}
        if inc_les: out["det"] = np.mean([bacc(P[h], Y[h], w) for h in inc_les], axis=0)
        if classes: out["cat"] = np.mean([((CATP == k) * (refcat == k) * w).sum(1) / ((refcat == k) * w).sum() for k in classes], axis=0)
        if has_auc:
            ii = np.arange(n) if idx is None else idx; ii = ii[lab[ii] >= 0]; out["auc"] = np.array([G.auc_fast(SC[m][ii], lab[ii].astype(float)) for m in range(M)])
        return out
    pt = metrics(np.ones(n)); reps = {k: [] for k in pt}
    for _ in range(a.boot):
        idx = G.boot_indices(cl_list, rng); w = np.bincount(idx, minlength=n).astype(float); mm = metrics(w, idx)
        for k in mm: reps[k].append(mm[k])
    reps = {k: np.array(v) for k, v in reps.items()}
    floor = {"det": 0.5, "cat": (1.0 / len(classes)) if classes else np.nan, "auc": 0.5}
    for k in pt:
        v, c, o = pt[k][iv].mean(), pt[k][icb].mean(), pt[k][iop].mean()
        d1 = reps[k][:, iv].mean(1) - reps[k][:, icb].mean(1); d2 = reps[k][:, iv].mean(1) - reps[k][:, iop].mean(1); dv = reps[k][:, iv].mean(1) - floor[k]
        l1, h1 = pct(d1); l2, h2 = pct(d2); lf, hf = pct(dv)
        rows_main.append(dict(split=sp, metric=k, n=n, patients=len(cl_list), slots_or_classes=";".join(inc_les if k == "det" else classes if k == "cat" else ["malignancy"]), vlm=v, cb=c, opq=o, floor=floor[k],
                              diff_vlm_cb=v - c, lo_cb=l1, hi_cb=h1, diff_vlm_opq=v - o, lo_opq=l2, hi_opq=h2, vlm_minus_floor=v - floor[k], lo_floor=lf, hi_floor=hf,
                              noninf=bool(l1 >= -EQ), better=bool(l1 > EQ), worse=bool(h1 < -EQ), above_floor=bool(lf > 0)))
        for i in iv: rows_seed.append(dict(split=sp, metric=k, seed=names[i], value=pt[k][i]))
    # per-slot sensitivity and specificity (descriptive)
    for h in LES:
        if (Y[h] == 1).sum() and (Y[h] == 0).sum():
            for grp, ix in (("vlm", iv), ("cb", icb), ("opq", iop)):
                rows_slot.append(dict(split=sp, slot=h, source=grp, n_pos=int((Y[h] == 1).sum()), n_neg=int((Y[h] == 0).sum()), sensitivity=float(np.mean([P[h][m][Y[h] == 1].mean() for m in ix])), specificity=float(np.mean([(~P[h][m])[Y[h] == 0].mean() for m in ix]))))
        elif (Y[h] == 1).sum():
            for grp, ix in (("vlm", iv), ("cb", icb), ("opq", iop)):
                rows_slot.append(dict(split=sp, slot=h, source=grp, n_pos=int((Y[h] == 1).sum()), n_neg=0, sensitivity=float(np.mean([P[h][m][Y[h] == 1].mean() for m in ix])), specificity=np.nan))
    # descriptors: parent stated, omission given parent stated, accuracy given parent stated
    for d, par in DESC.items():
        q = np.array([r.get(d) is not None and r.get(par) == "present" for r in refs])
        if q.sum() == 0: continue
        for grp, ix in (("vlm", iv), ("cb", icb)):
            st, om, ac = [], [], []
            for m in ix:
                sl = arr[m]["slots"]; ps = np.array([sl[i].get(par) == "present" for i in range(n)]); qq = q & ps
                st.append(ps[q].mean()); om.append(np.mean([sl[i].get(d) is None for i in np.where(qq)[0]]) if qq.sum() else np.nan)
                ac.append(np.mean([sl[i].get(d) == refs[i].get(d) for i in np.where(qq)[0]]) if qq.sum() else np.nan)
            rows_desc.append(dict(split=sp, descriptor=d, source=grp, n_labelled=int(q.sum()), parent_stated_share=float(np.mean(st)), n_qualifying=float(np.mean([ (q & np.array([arr[m]["slots"][i].get(par) == "present" for i in range(n)])).sum() for m in ix])),
                                  omission_given_parent=float(np.nanmean(om)) if not np.all(np.isnan(om)) else np.nan, accuracy_given_parent=float(np.nanmean(ac)) if not np.all(np.isnan(ac)) else np.nan))
    # image-dependence controls on the first N_V2 rows
    sw = [jl(vlm_file(s, sp, "swap")) for s in range(3)]; bl = [jl(vlm_file(s, sp, "blank")) for s in range(3)]
    if sw[0] is not None and bl[0] is not None:
        m2 = len(sw[0]); sub = np.arange(m2)
        for var, X in (("orig", None), ("swap", sw), ("blank", bl)):
            res = {}
            for s in range(3):
                sl = slots_by[s][:m2] if X is None else [r["slots"] for r in X[s]]
                if X is not None: assert [r["sample_id"] for r in X[s]] == sid[:m2]
                Pv = {h: np.array([[x.get(h) == "present" for x in sl]]) for h in LES}
                if inc_les: res.setdefault("det", []).append(float(np.mean([bacc(Pv[h], Y[h][sub], np.ones(m2))[0] for h in inc_les])))
                if has_auc:
                    sc = np.array([to_score(x) for x in sl]); k2 = np.where(lab[sub] >= 0)[0]; res.setdefault("auc", []).append(G.auc_fast(sc[k2], lab[sub][k2].astype(float)))
            for k, v in res.items(): rows_v2.append(dict(split=sp, metric=k, variant=var, n=m2, value=float(np.nanmean(v))))
pd.DataFrame(rows_main).to_csv(f"{a.out}/{a.mode}_main.csv", index=False); pd.DataFrame(rows_seed).to_csv(f"{a.out}/{a.mode}_seeds.csv", index=False)
pd.DataFrame(rows_slot).to_csv(f"{a.out}/{a.mode}_slots.csv", index=False); pd.DataFrame(rows_desc).to_csv(f"{a.out}/{a.mode}_descriptors.csv", index=False)
v2 = pd.DataFrame(rows_v2); v2.to_csv(f"{a.out}/{a.mode}_v2_values.csv", index=False)
# decisions
M_ = pd.DataFrame(rows_main); crit = {"mode": a.mode}
def family(metric, sets):
    d = M_[(M_.metric == metric) & (M_.split.isin(sets))]; k = len(d)
    r = dict(sets=int(k), noninf=int(d.noninf.sum()), better=int(d.better.sum()), worse=int(d.worse.sum()), above_floor=int(d.above_floor.sum()))
    r["outcome"] = ("better" if r["better"] >= 3 else "noninferior" if r["noninf"] >= 3 else "worse" if r["worse"] >= 3 else "no stable difference") if (a.mode == "fresh" and k >= 3) else "not defined (fewer than 3 sets)"
    return r
if a.mode == "fresh":
    crit["P1_detection"] = family("det", DEC_SETS); crit["P2_category"] = family("cat", DEC_SETS)
    d = M_[(M_.metric == "auc") & (M_.split == "test_cmmd")]
    crit["P3_auc_cmmd"] = dict(outcome=("better" if bool(d.better.iloc[0]) else "noninferior" if bool(d.noninf.iloc[0]) else "worse" if bool(d.worse.iloc[0]) else "inconclusive"), diff=float(d.diff_vlm_cb.iloc[0]), lo=float(d.lo_cb.iloc[0]), hi=float(d.hi_cb.iloc[0])) if len(d) else "not defined"
    D_ = pd.DataFrame(rows_desc); q = D_[(D_.source == "vlm") & (D_.n_qualifying >= 20)]
    crit["P4_descriptors"] = dict(cells=int(len(q)), max_omission_given_parent=float(q.omission_given_parent.max()) if len(q) else None, passed=bool(len(q) > 0 and (q.omission_given_parent <= OMIT_MAX).all()))
    ctl = {}
    for sp in DEC_SETS + AUC_SETS:
        for metric in ("det", "auc"):
            g = v2[(v2.split == sp) & (v2.metric == metric)].set_index("variant").value
            if len(g) == 3 and g["orig"] - 0.5 >= 0.05: ctl[f"{sp}:{metric}"] = dict(orig=float(g["orig"]), swap=float(g["swap"]), blank=float(g["blank"]), D_swap=float((g["orig"] - g["swap"]) / (g["orig"] - 0.5)), D_blank=float((g["orig"] - g["blank"]) / (g["orig"] - 0.5)))
    el = [v for k, v in ctl.items() if k.endswith(":det")]
    crit["P5_controls"] = dict(eligible_detection_sets=len(el), swap_ok=int(sum(v["D_swap"] >= 0.8 for v in el)), passed=bool(len(el) >= 3 and sum(v["D_swap"] >= 0.8 for v in el) >= 3), detail=ctl)
json.dump(crit, open(f"{a.out}/{a.mode}_criteria.json", "w"), indent=1); print(json.dumps(crit)[:1500]); print("done")
