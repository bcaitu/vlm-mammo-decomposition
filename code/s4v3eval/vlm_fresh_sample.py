#!/usr/bin/env python3
"""Draws the fresh evaluation rows of the second VLM study (paper 4, s4v3eval). Only reference labels are read; no model output is touched.
EMBED test sets: rows of patients that are not in the grid-2 evaluation subsample of the same set; stratum A = up to 500 breasts with a labelled lesion (mass, calc, asymmetry or distortion present),
stratum B = up to 500 breasts without a present lesion and with at least one of mass/calc labelled absent. VinDr-Mammo and CMMD: all test breasts that are not in the grid-2 subsample (patients can overlap with it).
usage: PYTHONPATH=$PWD python3 analysis_embed/vlm_fresh_sample.py --out runs_vlm/_fresh/fresh_rows.json"""
import argparse, collections, json, os, random, sys
import torch
from cbmammo import data as D
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
SEED, NA, NB = 20261005, 500, 500
LES = ("mass", "calc", "asymmetry", "distortion")
ALL = ["manifests/cbis_ddsm.jsonl", "manifests/vindr.jsonl", "manifests/embed.jsonl", "manifests/cmmd.jsonl", "manifests/cdd_cesm.jsonl", "manifests/inbreast.jsonl"]
recs = {r["sample_id"]: r for r in D.load_manifest(ALL)}
run_dir, g2 = "runs/cb_embed_dinov2_s0", "runs_stage2/grid2_cb_embed_dinov2_s0"
out = {}
for sp in ["test_embed", "test_embed_fuji", "test_embed_ge", "test_vindr", "test_cmmd"]:
    sids = torch.load(os.path.join(run_dir, f"dump_{sp}.pt"), map_location="cpu")["sample_id"]
    used = {json.loads(l)["dump_row"] for l in open(os.path.join(g2, f"gen_{sp}.jsonl"))}
    pat = lambda i: recs[sids[i]]["patient_id"]; val = lambda i: recs[sids[i]]["values"] or {}
    pool = [i for i in range(len(sids)) if i not in used]
    emb = sp.startswith("test_embed")
    if emb:
        used_pat = {pat(i) for i in used}; pool = [i for i in pool if pat(i) not in used_pat]
    rng = random.Random(f"{SEED}|{sp}")
    if emb:
        isA = lambda i: any(val(i).get(k) == "present" for k in LES)
        isB = lambda i: (not isA(i)) and (val(i).get("mass") == "absent" or val(i).get("calc") == "absent")
        A = [i for i in pool if isA(i)]; B = [i for i in pool if isB(i)]
        rng.shuffle(A); rng.shuffle(B)
        sel = [(i, "A") for i in A[:NA]] + [(i, "B") for i in B[:NB]]; rng.shuffle(sel)
        avail = (len(A), len(B))
    else:
        sel = [(i, "all") for i in pool]; rng.shuffle(sel); avail = (len(pool), 0)
    cl = {p: k for k, p in enumerate(sorted({pat(i) for i, _ in sel}))}
    out[sp] = [dict(sample_id=sids[i], cluster=cl[pat(i)], stratum=s) for i, s in sel]
    C = collections.Counter()
    for i, s in sel:
        C["stratum_" + s] += 1
        for k in LES: C[f"{k}={val(i).get(k)}"] += 1
    print(sp, "n", len(sel), "patients", len(cl), "available(A,B or all)", avail, dict(sorted(C.items())), flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"))
