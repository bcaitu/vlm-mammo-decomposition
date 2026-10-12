#!/usr/bin/env python3
"""Post hoc diagnostics for the VLM study (paper 4), run after the pre-registered analysis: (a) macro slot accuracy of trivial prior reports (global modal class of the training labels; modal class of the test set itself),
(b) descriptor omission given that the parent finding is stated, VLM versus concept-route decoders, (c) image-dependence index with the set-specific prior. Aggregates only."""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_grid_analysis2 as G
H = G.HEADS; DESC = list(G.PARENT)
def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else None
def macro(arr, hs):
    sub = arr[hs]; cnt = np.sum(~np.isnan(sub), axis=1); ok = cnt > 0
    return float(np.mean(np.nansum(sub, axis=1)[ok] / cnt[ok]))
prior = json.load(open("runs_vlm/_v12_tables/vlm_prior_report.json"))
v1d = [f"runs_vlm/v1_s{s}/gen" for s in range(3)]; cbr = sorted(glob.glob("runs_stage2/grid2_cb_embed_*")); opr = sorted(glob.glob("runs_stage2/grid2_opq_embed_*"))
out_p, out_o, out_d = [], [], []
for sp in G.TESTS:
    v1 = [jl(f"{d}/gen_{sp}_orig.jsonl") for d in v1d]; n = len(v1[0])
    cb = [jl(f"{r}/gen_{sp}.jsonl")[:n] for r in cbr]; op = [jl(f"{r}/gen_{sp}.jsonl")[:n] for r in opr]; v0 = jl(f"runs_vlm/v0/gen/gen_{sp}_orig.jsonl")
    R = lambda rows, key, ref: [dict(sample_id="", gt_values=ref, slots=r[key]) for r in rows]
    refs = [r["values"] for r in v1[0]]
    mk = lambda rows, key: G.correctness([dict(gt_values=refs[i], **{key: rows[i][key]}) for i in range(n)], key, H)
    C_v = [G.correctness([dict(gt_values=refs[i], slots=x[i]["slots"]) for i in range(n)], "slots", H) for x in v1]
    C_c = [G.correctness([dict(gt_values=refs[i], slots=x[i]["slots_generated"]) for i in range(n)], "slots", H) for x in cb]
    C_o = [G.correctness([dict(gt_values=refs[i], slots=x[i]["slots_generated"]) for i in range(n)], "slots", H) for x in op]
    C_0 = G.correctness([dict(gt_values=refs[i], slots=v0[i]["slots"]) for i in range(n)], "slots", H)
    hs = [i for i in range(len(H)) if np.sum(~np.isnan(C_v[0][i])) >= G.MIN_N]
    modal = {}
    for h in H:
        vals = [r.get(h) for r in refs if r.get(h) is not None]; modal[h] = max(set(vals), key=vals.count) if vals else None
    C_pg = G.correctness([dict(gt_values=refs[i], slots=prior) for i in range(n)], "slots", H)
    C_ps = G.correctness([dict(gt_values=refs[i], slots=modal) for i in range(n)], "slots", H)
    out_p.append(dict(split=sp, n=n, heads=len(hs), vlm=np.mean([macro(x, hs) for x in C_v]), cb=np.mean([macro(x, hs) for x in C_c]), opaque=np.mean([macro(x, hs) for x in C_o]), v0=macro(C_0, hs),
                      prior_global=macro(C_pg, hs), prior_set=macro(C_ps, hs)))
    # descriptor omission given the parent stated present in the report
    def om(rows_slots):
        n_ = o_ = pn = 0
        for i, s in enumerate(rows_slots):
            for h, p in G.PARENT.items():
                if refs[i].get(h) is not None:
                    pn += 1
                    if s.get(p) == "present": n_ += 1; o_ += int(s.get(h) is None)
        return pn, n_, o_
    for tag, lists in (("VLM", [[r["slots"] for r in x] for x in v1]), ("cb decoders", [[r["slots_generated"] for r in x] for x in cb])):
        res = [om(l) for l in lists]
        pn = np.mean([r[0] for r in res]); st = np.mean([r[1] for r in res]); oo = np.mean([r[2] for r in res])
        out_o.append(dict(split=sp, source=tag, labelled_descriptors=pn, parent_stated=st, parent_stated_share=st / pn if pn else np.nan, omitted_given_parent=oo / st if st else np.nan))
    # D with the set-specific prior on the first 300 rows
    sw = [jl(f"{d}/gen_{sp}_swap.jsonl") for d in v1d]; m = len(sw[0]); sub = np.arange(m)
    hh = [i for i in hs if np.sum(~np.isnan(C_v[0][i][sub])) > 0]
    o_acc = np.mean([macro(x[:, sub], hh) for x in C_v])
    s_acc = np.mean([macro(G.correctness([dict(gt_values=refs[i], slots=x[i]["slots"]) for i in range(m)], "slots", H), hh) for x in sw])
    p_acc = macro(C_ps[:, sub], hh); den = o_acc - p_acc
    out_d.append(dict(split=sp, acc_orig=o_acc, acc_swap=s_acc, acc_prior_set=p_acc, denominator=den, D_swap_set_prior=(o_acc - s_acc) / den if den > 0 else np.nan))
os.makedirs("runs_vlm/_v12_tables", exist_ok=True)
for n_, r_ in (("posthoc_prior.csv", out_p), ("posthoc_omission.csv", out_o), ("posthoc_D_set_prior.csv", out_d)): pd.DataFrame(r_).to_csv(f"runs_vlm/_v12_tables/{n_}", index=False)
print("done")
