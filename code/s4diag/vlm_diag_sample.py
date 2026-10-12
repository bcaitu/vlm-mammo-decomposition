#!/usr/bin/env python3
"""Development sample for the perception-versus-verbalization diagnostic of the VLM (paper 4, s4diag). Only reference labels are read.
EMBED validation breasts (rows of the stage-1 validation dump): stratum A = up to 1000 breasts with a present lesion (mass, calc, asymmetry or distortion), stratum B = up to 1000 without a present lesion and with mass or calc labelled absent.
Parts by patient hash: train 50 %, select 20 %, test 30 %.
usage: PYTHONPATH=$PWD python3 analysis_embed/vlm_diag_sample.py --out runs_vlm/_diag/dev_rows.json"""
import argparse, collections, hashlib, json, os, random
import torch
from cbmammo import data as D
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
SEED, NA, NB = 20261006, 1000, 1000
LES = ("mass", "calc", "asymmetry", "distortion")
recs = {r["sample_id"]: r for r in D.load_manifest(["manifests/embed.jsonl"])}
sids = torch.load("runs/cb_embed_dinov2_s0/dump_val.pt", map_location="cpu")["sample_id"]
pool = [s for s in sids if s in recs and recs[s]["dataset"] == "embed" and recs[s]["split"] == "val"]
val = lambda s: recs[s]["values"] or {}
isA = lambda s: any(val(s).get(k) == "present" for k in LES)
isB = lambda s: (not isA(s)) and (val(s).get("mass") == "absent" or val(s).get("calc") == "absent")
A = [s for s in pool if isA(s)]; B = [s for s in pool if isB(s)]
rng = random.Random(SEED); rng.shuffle(A); rng.shuffle(B)
sel = [(s, "A") for s in A[:NA]] + [(s, "B") for s in B[:NB]]; rng.shuffle(sel)
pats = sorted({recs[s]["patient_id"] for s, _ in sel}); cl = {p: k for k, p in enumerate(pats)}
def part(p):
    h = int(hashlib.sha1(f"diag|{p}".encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 50 else "select" if h < 70 else "test"
rows = [dict(sample_id=s, cluster=cl[recs[s]["patient_id"]], stratum=st, part=part(recs[s]["patient_id"])) for s, st in sel]
json.dump({"dev_embed_val": rows}, open(a.out, "w"))
C = collections.Counter()
for r in rows:
    C["part_" + r["part"]] += 1; C["stratum_" + r["stratum"]] += 1
    for k in LES:
        v = val(r["sample_id"]).get(k)
        if v in ("present", "absent"): C[f"{k}_{v}_{r['part']}"] += 1
print("pool", len(pool), "A", len(A), "B", len(B), "patients", len(pats), dict(sorted(C.items())))
