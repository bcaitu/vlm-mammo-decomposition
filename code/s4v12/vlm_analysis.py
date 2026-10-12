#!/usr/bin/env python3
"""Analysis of the VLM study V1-V2 (paper 4), following prereg/s4v12_prereg_ru.md. Aggregates only; no sample identifiers are written.
Definition note (written before the first look at results): the macro slot accuracy M1 is the one used for R1 in Chapter 6 (s2_grid_analysis2.py): mean over the heads with >= MIN_N labelled
rows of the nine concept slots AND the BI-RADS category; M1 without the category is given as a secondary column. M2 is the category alone.
usage: PYTHONPATH=$PWD python3 analysis_embed/vlm_analysis.py --out runs_vlm/_v12_tables [--v1_dirs runs_vlm/v1_s0/gen,...] [--v0_dir runs_vlm/v0/gen] [--boot 2000]"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_grid_analysis2 as G
from cbmammo import data as D
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--min_n", type=int, default=G.MIN_N)
ap.add_argument("--v1_dirs", default="runs_vlm/v1_s0/gen,runs_vlm/v1_s1/gen,runs_vlm/v1_s2/gen"); ap.add_argument("--v0_dir", default="runs_vlm/v0/gen"); ap.add_argument("--sets", default="")
ap.add_argument("--stage2_root", default="runs_stage2"); ap.add_argument("--n_v2", type=int, default=300)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(0)
SETS = a.sets.split(",") if a.sets else G.TESTS; H = G.HEADS; DESC = list(G.PARENT); EQ = G.EQ
v1_dirs = [d for d in a.v1_dirs.split(",") if d]
def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else None
def as_rows(rows, key): return [dict(sample_id=r["sample_id"], gt_values=r["values"], slots=r[key]) for r in rows]
# prior report: modal class per head in the training labels
M = ["manifests/cbis_ddsm.jsonl", "manifests/vindr.jsonl", "manifests/embed.jsonl"]
tr = D.load_manifest(M, splits=["train"]); prior = {}
for h in H:
    vals = [r["values"].get(h) for r in tr if r["values"].get(h) is not None]; prior[h] = max(set(vals), key=vals.count)
json.dump(prior, open(os.path.join(a.out, "vlm_prior_report.json"), "w"))
def macro(arr, idx, hs):
    sub = arr[hs][:, idx]; cnt = np.sum(~np.isnan(sub), axis=1); ok = cnt > 0
    return float(np.mean(np.nansum(sub, axis=1)[ok] / cnt[ok])) if ok.any() else float("nan")
def pct(x): return G.pct(x)
cb_runs = sorted(glob.glob(os.path.join(a.stage2_root, "grid2_cb_embed_*"))); op_runs = sorted(glob.glob(os.path.join(a.stage2_root, "grid2_opq_embed_*")))
rows_diff, rows_cat, rows_auc, rows_desc, rows_v2, rows_om = [], [], [], [], [], []
crit = dict(V1_1=dict(better=0, noninf=0, worse=0, sets=0), V1_2=dict(better=0, noninf=0, worse=0, sets=0), V1_3=dict(all_ok=True, max=0.0), V2_1=dict(eligible=0, ok=0, D={}))
for sp in SETS:
    v1 = [jl(os.path.join(d, f"gen_{sp}_orig.jsonl")) for d in v1_dirs]; v1 = [x for x in v1 if x]
    if not v1: continue
    n = len(v1[0]); v0 = jl(os.path.join(a.v0_dir, f"gen_{sp}_orig.jsonl")) if a.v0_dir else None
    cbg = [(jl(os.path.join(r, f"gen_{sp}.jsonl")) or [])[:n] for r in cb_runs]; opg = [(jl(os.path.join(r, f"gen_{sp}.jsonl")) or [])[:n] for r in op_runs]
    # alignment check: reference values of the concept route equal those of the VLM rows
    mism = sum(1 for i in range(min(n, 200)) if cbg and cbg[0] and (cbg[0][i].get("gt_values") or {}) != v1[0][i]["values"])
    if mism: print("WARNING reference mismatch", sp, mism, flush=True)
    C_v1 = [G.correctness(as_rows(x, "slots"), "slots", H) for x in v1]
    C_cb = [G.correctness(r, "slots_generated", H) for r in cbg if r]; C_op = [G.correctness(r, "slots_generated", H) for r in opg if r]
    C_v0 = G.correctness(as_rows(v0, "slots"), "slots", H) if v0 else None
    lab_n = np.sum(~np.isnan(C_v1[0]), axis=1); hs = [i for i in range(len(H)) if lab_n[i] >= a.min_n]
    if not hs: continue
    hs_c = [i for i in hs if H[i] != "birads"]
    pats = {}
    for j, x in enumerate(v1[0]): pats.setdefault(G.patient_of(x["sample_id"]), []).append(j)
    clusters = [np.array(v) for v in pats.values()]; full = np.arange(n); ib = H.index("birads")
    def mm(arrs, idx, sel): return float(np.mean([macro(x, idx, sel) for x in arrs]))
    pe = dict(cb=mm(C_v1, full, hs) - mm(C_cb, full, hs), op=mm(C_v1, full, hs) - mm(C_op, full, hs))
    reps = dict(cb=[], op=[], v0=[], cb_nc=[])
    for _ in range(a.boot):
        idx = G.boot_indices(clusters, rng); v = mm(C_v1, idx, hs)
        reps["cb"].append(v - mm(C_cb, idx, hs)); reps["op"].append(v - mm(C_op, idx, hs))
        if C_v0 is not None: reps["v0"].append(v - macro(C_v0, idx, hs))
        reps["cb_nc"].append(mm(C_v1, idx, hs_c) - mm(C_cb, idx, hs_c) if hs_c else np.nan)
    for lab, key, est in (("VLM minus concept route (cb decoders)", "cb", pe["cb"]), ("VLM minus opaque decoders", "op", pe["op"])):
        lo, hi = pct(reps[key]); rows_diff.append(dict(split=sp, contrast=lab, n_heads=len(hs), diff_mean=est, ci_lo=lo, ci_hi=hi, n_seeds=len(C_v1), n_refs=len(C_cb if key == "cb" else C_op)))
    if hs_c:
        lo, hi = pct(reps["cb_nc"]); rows_diff.append(dict(split=sp, contrast="VLM minus cb decoders, concept slots only (no category)", n_heads=len(hs_c), diff_mean=mm(C_v1, full, hs_c) - mm(C_cb, full, hs_c), ci_lo=lo, ci_hi=hi, n_seeds=len(C_v1), n_refs=len(C_cb)))
    if C_v0 is not None:
        lo, hi = pct(reps["v0"]); rows_diff.append(dict(split=sp, contrast="VLM minus zero-shot V0", n_heads=len(hs), diff_mean=mm(C_v1, full, hs) - macro(C_v0, full, hs), ci_lo=lo, ci_hi=hi, n_seeds=len(C_v1), n_refs=1))
    for s_i, x in enumerate(C_v1): rows_diff.append(dict(split=sp, contrast=f"VLM seed {s_i} macro", n_heads=len(hs), diff_mean=macro(x, full, hs), ci_lo=np.nan, ci_hi=np.nan, n_seeds=1, n_refs=0))
    lo, hi = pct(reps["cb"]); c = crit["V1_1"]; c["sets"] += 1; c["noninf"] += int(lo >= -EQ); c["better"] += int(lo > EQ); c["worse"] += int(hi < -EQ)
    # M2: category alone
    if ib in hs:
        e2 = mm([x[[ib]] for x in C_v1], full, [0]) - mm([x[[ib]] for x in C_cb], full, [0]); r2 = []
        for _ in range(a.boot):
            idx = G.boot_indices(clusters, rng); r2.append(mm([x[[ib]] for x in C_v1], idx, [0]) - mm([x[[ib]] for x in C_cb], idx, [0]))
        lo2, hi2 = pct(r2); rows_cat.append(dict(split=sp, contrast="category accuracy, VLM minus cb decoders", diff_mean=e2, ci_lo=lo2, ci_hi=hi2, vlm=mm([x[[ib]] for x in C_v1], full, [0]), cb=mm([x[[ib]] for x in C_cb], full, [0])))
        c = crit["V1_2"]; c["sets"] += 1; c["noninf"] += int(lo2 >= -EQ); c["better"] += int(lo2 > EQ); c["worse"] += int(hi2 < -EQ)
    # M3: AUC of the category in the text
    lab = np.array([{"benign": 0, "malignant": 1}.get((x["values"] or {}).get("malignant"), -1) for x in v1[0]])
    if np.sum(lab == 0) >= 20 and np.sum(lab == 1) >= 20:
        keep = np.where(lab >= 0)[0]
        def sc(rows, key): return np.array([float(r[key]["birads"]) if (r.get(key) or {}).get("birads") is not None else np.nan for r in rows])
        Sv = [sc(x, "slots") for x in v1]; Sc = [sc(r, "slots_generated") for r in cbg if r]
        ev = np.mean([G.auc_fast(s[keep], lab[keep]) for s in Sv]) - np.mean([G.auc_fast(s[keep], lab[keep]) for s in Sc]); ra = []
        pos = {int(j): i for i, j in enumerate(keep)}; kc = [np.array([pos[int(j)] for j in cl if int(j) in pos]) for cl in clusters]; kc = [k for k in kc if len(k)]
        for _ in range(a.boot):
            ii = G.boot_indices(kc, rng); ra.append(np.nanmean([G.auc_fast(s[keep][ii], lab[keep][ii]) for s in Sv]) - np.nanmean([G.auc_fast(s[keep][ii], lab[keep][ii]) for s in Sc]))
        lo3, hi3 = pct(ra); rows_auc.append(dict(split=sp, contrast="AUC of the category in text, VLM minus cb decoders", diff_mean=float(ev), ci_lo=lo3, ci_hi=hi3, n_pos=int(np.sum(lab == 1)), n_neg=int(np.sum(lab == 0))))
    # descriptive: omission, spurious, inconsistency; V1-3 omission of labelled descriptors
    def stats(rows, tag, seed):
        om = nl = sp_n = sp_d = inc = 0; omd = nld = 0
        for r in rows:
            v, s = r["values"], r["slots"]
            for h in H:
                if v.get(h) is not None:
                    nl += 1; om += int(s.get(h) is None)
                    if h in DESC: nld += 1; omd += int(s.get(h) is None)
            for h, p in G.PARENT.items():
                if v.get(p) == "absent": sp_d += 1; sp_n += int(s.get(h) is not None)
            b = s.get("birads"); named = [s.get(k) == "present" for k in ("mass", "calc", "asymmetry", "distortion")]
            inc += int((b in ("4", "5") and not any(named)) or (b in ("1", "2") and (s.get("mass") == "present" or s.get("distortion") == "present")))
        return dict(split=sp, variant=tag, seed=seed, n=len(rows), omission_labelled=om / max(nl, 1), omission_descriptors=(omd / nld) if nld else np.nan, spurious=sp_n / max(sp_d, 1), inconsistent=inc / max(len(rows), 1))
    for s_i, x in enumerate(v1): rows_desc.append(stats(x, "orig", s_i))
    if v0: rows_desc.append(stats(v0, "V0 zero-shot", -1))
    od = [r["omission_descriptors"] for r in rows_desc if r["split"] == sp and r["variant"] == "orig" and r["omission_descriptors"] == r["omission_descriptors"]]
    if od: crit["V1_3"]["max"] = max(crit["V1_3"]["max"], max(od)); crit["V1_3"]["all_ok"] &= bool(max(od) <= 0.10)
    # V2: swap and blank on the first n_v2 rows
    sw = [jl(os.path.join(d, f"gen_{sp}_swap.jsonl")) for d in v1_dirs]; bl = [jl(os.path.join(d, f"gen_{sp}_blank.jsonl")) for d in v1_dirs]
    if all(sw) and all(bl):
        m = len(sw[0]); sub = np.arange(m); hh = [i for i in hs if np.sum(~np.isnan(C_v1[0][i][sub])) > 0]
        def acc_of(rows): return mm([G.correctness(as_rows(x, "slots"), "slots", H) for x in rows], sub, hh)
        o_acc = mm(C_v1, sub, hh); s_acc = acc_of(sw); b_acc = acc_of(bl)
        pr_rows = [dict(sample_id="", gt_values=r["values"], slots=prior) for r in v1[0][:m]]; p_acc = macro(G.correctness(pr_rows, "slots", H), sub, hh)
        den = o_acc - p_acc; Dv = (o_acc - s_acc) / den if den > 0 else np.nan
        rows_v2.append(dict(split=sp, n=m, acc_orig=o_acc, acc_swap=s_acc, acc_blank=b_acc, acc_prior=p_acc, denominator=den, D_swap=Dv, D_blank=(o_acc - b_acc) / den if den > 0 else np.nan))
        if den >= 0.05: crit["V2_1"]["eligible"] += 1; crit["V2_1"]["ok"] += int(Dv >= 0.8); crit["V2_1"]["D"][sp] = float(Dv)
        for tag, runs_ in (("swap", sw), ("blank", bl)):
            for s_i, x in enumerate(runs_): rows_desc.append(stats(x, tag, s_i))
    print("done", sp, flush=True)
def outcome(c, nsets):
    return "VLM clearly better" if c["better"] >= 4 else "concept route clearly better" if c["worse"] >= 4 else "VLM not worse" if c["noninf"] >= 6 else "no stable difference"
for k in ("V1_1", "V1_2"): crit[k]["outcome"] = outcome(crit[k], crit[k]["sets"])
crit["V2_1"]["passed"] = bool(crit["V2_1"]["eligible"] >= 5 and crit["V2_1"]["ok"] == crit["V2_1"]["eligible"])
crit["V1_3"]["passed"] = bool(crit["V1_3"]["all_ok"])
for n_, r_ in (("vlm_v1_macro_diff.csv", rows_diff), ("vlm_v1_category.csv", rows_cat), ("vlm_v1_auc.csv", rows_auc), ("vlm_descriptive.csv", rows_desc), ("vlm_v2.csv", rows_v2)): pd.DataFrame(r_).to_csv(os.path.join(a.out, n_), index=False)
json.dump(crit, open(os.path.join(a.out, "vlm_criteria.json"), "w"), indent=1, default=float); print(json.dumps(crit, default=float)[:800])
