#!/usr/bin/env python3
"""s10an (paper 4): analyses on saved scores and on the manifests (no training, no GPU).
B1  paired contrast VP - A_probe at the full size (and the reproduction of K1 against A_tf);
B2  one examination per patient: full-size contrasts on the subset of the earliest examination of each patient (primary) and on 200 random draws of one examination per patient (secondary);
B3  V (own images, seed 0) against the black-image control V_blank on the same 3,000 test breasts (runs_lc/sets/test_sub3000.json);
B4  counts of present / absent / unknown labels per data set, partition and finding;
B5  composition of the EMBED partitions (age, density, vendor, examination type, race as coded) and positives by examination type in the test set.
Only aggregate tables are written (cells below 10 are shown as '<10'); no row-level output. Protocol: prereg/s10an_prereg_ru.md.
usage: PYTHONPATH=$PWD:$PWD/analysis_embed ~/envs/cbmammo/bin/python analysis_embed/s10an_analysis.py --out runs_s10an/analysis [--boot 2000] [--check-only]"""
import argparse, csv, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s10_common import *
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--draws", type=int, default=200)
ap.add_argument("--check-only", action="store_true", help="only reproduce known contrasts of s6lc / s8 and stop (no new quantity is computed)"); ap.add_argument("--dry-run", action="store_true", help="code test only (small bootstrap, reproduction check not enforced); outputs are not read and are deleted"); ap.add_argument("--skip", default="", help="comma list of blocks to skip, e.g. B2")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); skip = set(x for x in a.skip.split(",") if x)
t0 = time.time(); X = Ctx(boot=a.boot); log = open(os.path.join(a.out, "s10an.log"), "a")
def say(*m):
    s = " ".join(str(x) for x in m); print(s, flush=True); log.write(s + "\n"); log.flush()
say("start", time.strftime("%F %T"), "rows", X.n, "patients", len(X.up), "boot", X.R)

ARMS = ["V", "VP", "VC", "VCP", "V2x", "V_ctl", "D", "A_probe", "A_tf"]
def auc_all(rowmask=None):
    return {arm: X.arm_auc(arm, "f1", SEEDS, rowmask) for arm in ARMS}
def dif(T, a_, b_, f):
    if f not in T.get(a_, {}) or f not in T.get(b_, {}): return None
    return mean_boot(T[a_][f]) - mean_boot(T[b_][f])
# contrast table: name, arm a, arm b, rule, rule label
CON = [("K1 VP - A_tf (A* at the full size)", "VP", "A_tf", lambda d: cls(d, 0.03)), ("K1b VP - A_probe", "VP", "A_probe", lambda d: cls(d, 0.03)),
       ("K2 V - VP", "V", "VP", lambda d: cls(d, 0.03)), ("K3 V - D", "V", "D", lambda d: cls(d, 0.05)),
       ("S1 VC - V", "VC", "V", verdict_gain), ("S2 VC - D", "VC", "D", verdict_gap), ("S3 V2x - V", "V2x", "V", verdict_gain),
       ("S4 VCP - VP", "VCP", "VP", verdict_gain), ("S5 VC - VCP", "VC", "VCP", verdict_gap), ("desc V - V_ctl", "V", "V_ctl", lambda d: cls(d, 0.03))]
def contrasts(T, tag):
    rows = []
    for name, a_, b_, rule in CON:
        for f in FIND:
            d = dif(T, a_, b_, f)
            if d is None: continue
            lo, hi = ci(d); rows.append(dict(set=tag, contrast=name, finding=f, delta=float(d[0]), lo=lo, hi=hi, verdict=rule(d)))
    return pd.DataFrame(rows)

# ---------------- reproduction check against the frozen analyses of s6lc and s8
T_full = auc_all()
C_full = contrasts(T_full, "full test")
ref = []
for p, mp in (("runs_lc/analysis/lc_contrasts.csv", {"K2 V - VP": "K2 V - VP", "K3 V - D": "K3 V - D"}), ("runs_s8/analysis/s8_contrasts.csv", {"S1 VC - V (tower gain)": "S1 VC - V", "S2 VC - D (gap to the stage-1 head)": "S2 VC - D", "S3 V2x - V (step-budget gain)": "S3 V2x - V", "S5 VC - VCP (interface loss in the tower arm)": "S5 VC - VCP"})):
    if not os.path.exists(p): say("MISSING reference file", p); sys.exit(2)
    R_ = pd.read_csv(p)
    if "size" in R_.columns: R_ = R_[R_["size"] == "f1"]
    for old, new in mp.items():
        for f in FIND:
            r = R_[(R_["contrast"] == old) & (R_["finding"] == f)]; m = C_full[(C_full["contrast"] == new) & (C_full["finding"] == f)]
            if len(r) != 1 or len(m) != 1: say("check: pair not found", old, f); sys.exit(2)
            err = max(abs(float(r["delta"].iloc[0]) - float(m["delta"].iloc[0])), abs(float(r["lo"].iloc[0]) - float(m["lo"].iloc[0])), abs(float(r["hi"].iloc[0]) - float(m["hi"].iloc[0])))
            ref.append(dict(contrast=new, finding=f, max_abs_error=err))
chk = pd.DataFrame(ref); chk.to_csv(os.path.join(a.out, "s10an_check.csv"), index=False)
say("reproduction check: max abs error", float(chk["max_abs_error"].max()), "over", len(chk), "contrasts")
if chk["max_abs_error"].max() > 1e-5 and not a.dry_run: say("REPRODUCTION FAILED: stop"); sys.exit(3)
if a.check_only: say("check-only: done"); sys.exit(0)

# ---------------- B1
if "B1" not in skip:
    B1 = C_full[C_full["contrast"].isin(["K1 VP - A_tf (A* at the full size)", "K1b VP - A_probe"])].copy(); B1.to_csv(os.path.join(a.out, "b1_paired.csv"), index=False)
    d = {f: dif(T_full, "A_probe", "A_tf", f) for f in FIND}
    pd.DataFrame([dict(contrast="A_probe - A_tf", finding=f, delta=float(d[f][0]), lo=ci(d[f])[0], hi=ci(d[f])[1]) for f in FIND]).to_csv(os.path.join(a.out, "b1_probe_vs_tf.csv"), index=False)
    say("B1 done", round(time.time() - t0))

# ---------------- manifests, examinations
def load_manifest(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
MAN = {n: load_manifest(f"manifests/{n}.jsonl") for n in ("cbis_ddsm", "vindr", "embed")}
CLIN = {}
with open("/raid/datasets/vlm_mammogram_data/data/EMBED/tables/EMBED_OpenData_clinical_reduced.csv", encoding="utf-8", newline="") as fh:
    for r in csv.DictReader(fh):
        k = (r["empi_anon"], r["acc_anon"])
        if k not in CLIN: CLIN[k] = dict(age=r["age_at_study"], date=r["study_date_anon"], den=r["tissueden"], race=r["RACE_DESC"], eth=r["ETHNIC_GROUP_DESC"])
        else:
            for kk, src in (("age", "age_at_study"), ("date", "study_date_anon"), ("den", "tissueden"), ("race", "RACE_DESC")):
                if not CLIN[k][kk] and r[src]: CLIN[k][kk] = r[src]
EM = {r["sample_id"]: r for r in MAN["embed"]}
def parse(s): _, p, st, side = s.split(":"); return p, st, side
pat_row = [parse(s)[0] for s in X.te]; stu_row = [parse(s)[1] for s in X.te]
say("test rows with clinical record:", sum((p, s) in CLIN for p, s in zip(pat_row, stu_row)), "of", X.n)

# ---------------- B2
if "B2" not in skip:
    by_p = {}
    for p, s in zip(pat_row, stu_row): by_p.setdefault(p, set()).add(s)
    n_nodate = 0; first = {}
    for p, ex in by_p.items():
        key = lambda s: ((CLIN.get((p, s), {}).get("date") or "9999-99-99"), s)
        n_nodate += sum(1 for s in ex if not CLIN.get((p, s), {}).get("date"))
        first[p] = sorted(ex, key=key)[0]
    m1 = np.array([first[p] == s for p, s in zip(pat_row, stu_row)])
    nex = np.array([len(by_p[p]) for p in by_p]); say("B2: exams per patient mean", nex.mean(), "median", np.median(nex), "max", nex.max(), "| exams without date", n_nodate, "| subset rows", int(m1.sum()), "patients", len(by_p))
    T1 = auc_all(m1); C1 = contrasts(T1, "earliest examination per patient")
    C = pd.concat([C_full, C1]); C.to_csv(os.path.join(a.out, "b2_exam_primary.csv"), index=False)
    cnt = []
    for f in FIND:
        k = m1 & (X.Y[f] >= 0); cnt.append(dict(finding=f, rows=int(k.sum()), n_pos=int((X.Y[f][k] == 1).sum()), n_neg=int((X.Y[f][k] == 0).sum())))
    pd.DataFrame(cnt).to_csv(os.path.join(a.out, "b2_exam_counts.csv"), index=False)
    # secondary: random draws of one examination per patient (point estimates only)
    rng = np.random.default_rng(20261010); plist = list(by_p.keys()); elist = [sorted(by_p[p]) for p in plist]; ix = {(p, s): i for i, (p, s) in enumerate(zip(pat_row, stu_row))}
    S = {(arm, sd): X.scores(arm, "f1", sd) for arm in ARMS for sd in SEEDS}
    def point_auc(sc, f, mask):
        k = (X.Y[f] >= 0) & ~np.isnan(sc) & mask
        return float(roc_auc_score(X.Y[f][k], sc[k])) if k.sum() > 50 else np.nan
    rows_d = []
    for dr in range(a.draws):
        chosen = {p: elist[i][rng.integers(len(elist[i]))] for i, p in enumerate(plist)}
        mk = np.array([chosen[p] == s for p, s in zip(pat_row, stu_row)])
        A = {}
        for arm in ARMS:
            for f in FIND:
                v = [point_auc(S[(arm, sd)][f], f, mk) for sd in SEEDS if S[(arm, sd)] is not None]
                if v: A[(arm, f)] = float(np.nanmean(v))
        for name, a_, b_, rule in CON:
            for f in FIND:
                if (a_, f) in A and (b_, f) in A: rows_d.append(dict(draw=dr, contrast=name, finding=f, delta=A[(a_, f)] - A[(b_, f)]))
    Dd = pd.DataFrame(rows_d); g = Dd.groupby(["contrast", "finding"])["delta"]
    summ = pd.DataFrame(dict(mean=g.mean(), p2_5=g.quantile(0.025), p97_5=g.quantile(0.975), sd=g.std())).reset_index(); summ.to_csv(os.path.join(a.out, "b2_exam_draws.csv"), index=False)
    say("B2 done", round(time.time() - t0))

# ---------------- B3
if "B3" not in skip:
    sub = set(json.load(open("runs_lc/sets/test_sub3000.json"))); msk = np.array([s in sub for s in X.te])
    bl = X.scores("V_blank", "f1", 0); v0 = X.scores("V", "f1", 0); rows = []
    for f in FIND:
        k = msk & (X.Y[f] >= 0) & ~np.isnan(bl[f]) & ~np.isnan(v0[f])
        a_v = X.boot_auc(v0[f], f, k); a_b = X.boot_auc(bl[f], f, k); vm = X.arm_auc("V", "f1", SEEDS, k)[f]
        d = a_v - a_b; dm = mean_boot(vm) - a_b; lo, hi = ci(d); lb, hb = ci(a_b)
        rows.append(dict(finding=f, rows=int(k.sum()), n_pos=int((X.Y[f][k] == 1).sum()), n_neg=int((X.Y[f][k] == 0).sum()), auc_V_seed0=float(a_v[0]), auc_V_seedmean=float(mean_boot(vm)[0]), auc_blank=float(a_b[0]), blank_lo=lb, blank_hi=hb,
                         diff_V0_blank=float(d[0]), diff_lo=lo, diff_hi=hi, diffmean_V_blank=float(dm[0]), dm_lo=ci(dm)[0], dm_hi=ci(dm)[1], image_dependent=bool(lo > 0)))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "b3_blank.csv"), index=False); say("B3 done", round(time.time() - t0))

# ---------------- B4, B5 (aggregate counts)
def supp(n): return "<10" if 0 < n < 10 else int(n)
if "B4" not in skip:
    VS = set(json.load(open("runs_lc/sets/val_sample.json"))); TE = set(X.te); rows = []
    for ds, recs in MAN.items():
        parts = {"train (pool)": [r for r in recs if r["split"] == "train"], "validation split": [r for r in recs if r["split"] == "val"], "test split": [r for r in recs if r["split"] == "test"]}
        if ds == "embed": parts["validation sample (2,000)"] = [r for r in recs if r["sample_id"] in VS]; parts["test used in s6lc and s8 (EMBED test)"] = [r for r in recs if r["sample_id"] in TE]
        for pn, rs in parts.items():
            for f in ("mass", "calc", "asymmetry", "distortion"):
                for st, v in (("present", 1), ("absent", 0), ("unknown", -1)):
                    sel = [r for r in rs if r["labels"][f] == v]
                    rows.append(dict(dataset=ds, partition=pn, finding=f, state=st, breasts=supp(len(sel)), patients=supp(len({r["patient_id"] for r in sel})), partition_breasts=len(rs), partition_patients=len({r["patient_id"] for r in rs})))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "b4_label_counts.csv"), index=False); say("B4 done", round(time.time() - t0))
if "B5" not in skip:
    VS = set(json.load(open("runs_lc/sets/val_sample.json"))); TE = set(X.te); recs = MAN["embed"]
    parts = {"EMBED training pool": [r for r in recs if r["split"] == "train"], "EMBED validation sample": [r for r in recs if r["sample_id"] in VS], "EMBED test": [r for r in recs if r["sample_id"] in TE]}
    rows = []; DEN = {"1.0": "A", "2.0": "B", "3.0": "C", "4.0": "D"}
    for pn, rs in parts.items():
        cl = [CLIN.get((r["patient_id"], r["study_id"]), {}) for r in rs]; age = np.array([float(c["age"]) if c.get("age") not in (None, "") else np.nan for c in cl])
        ex = {(r["patient_id"], r["study_id"]) for r in rs}; per = {}
        for p, s in ex: per[p] = per.get(p, 0) + 1
        base = dict(partition=pn, breasts=len(rs), patients=len({r["patient_id"] for r in rs}), examinations=len(ex), exams_per_patient_mean=float(np.mean(list(per.values()))), exams_per_patient_median=float(np.median(list(per.values()))),
                    age_n=int((~np.isnan(age)).sum()), age_median=float(np.nanmedian(age)), age_q1=float(np.nanpercentile(age, 25)), age_q3=float(np.nanpercentile(age, 75)), age_min=float(np.nanmin(age)), age_max=float(np.nanmax(age)))
        rows.append(dict(base, variable="summary", category="", n=len(rs)))
        for var, getter in (("density (tissueden)", lambda r, c: DEN.get(c.get("den", ""), "missing")), ("vendor", lambda r, c: (r["embed_meta"] or {}).get("vendor") or "missing"),
                            ("examination type", lambda r, c: (r["embed_meta"] or {}).get("exam_type") or "missing"), ("race (RACE_DESC)", lambda r, c: c.get("race") or "missing")):
            cc = {}
            for r, c in zip(rs, cl): v = getter(r, c); cc[v] = cc.get(v, 0) + 1
            for v, n in sorted(cc.items(), key=lambda x: -x[1]): rows.append(dict(partition=pn, variable=var, category=v, n=supp(n)))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "b5_composition.csv"), index=False)
    rows = []
    for f in FIND:
        for et in ("screening", "diagnostic"):
            for st, v in (("present", 1), ("absent", 0), ("unknown", -1)):
                n = sum(1 for r in parts["EMBED test"] if (r["embed_meta"] or {}).get("exam_type") == et and r["labels"][f] == v); rows.append(dict(finding=f, exam_type=et, state=st, breasts=supp(n)))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "b5_test_by_examtype.csv"), index=False); say("B5 done", round(time.time() - t0))
say("all done", time.strftime("%F %T"), "elapsed s", round(time.time() - t0))
