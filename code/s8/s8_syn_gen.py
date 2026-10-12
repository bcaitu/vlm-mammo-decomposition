#!/usr/bin/env python3
"""s8: builds a SCRATCH tree ~/s8_syn (runs_lc/sets, runs_lc/{V,A,D}, runs_s8/{VC,V2x}) with synthetic scores to test analysis_embed/s8_analysis.py before the single real run.
Never touches the real runs. usage: python3 s8_syn_gen.py [--drop_vc_seed2]"""
import json, os, sys, zlib
import numpy as np, torch
root = os.path.expanduser("~/s8_syn"); os.makedirs(root, exist_ok=True); os.chdir(root)
rng = np.random.default_rng(1); n, npat = 1200, 300
pat = rng.integers(0, npat, size=n); ids = [f"embed:{pat[i]}:{i}:{'L' if i % 2 else 'R'}" for i in range(n)]
Y = np.stack([(rng.random(n) < p).astype(int) for p in (0.06, 0.06, 0.09)], 1); Y[rng.random((n, 3)) < 0.1] = -1
meta = {ids[i]: {"y": [int(v) for v in Y[i]], "p": str(pat[i])} for i in range(n)}
os.makedirs("runs_lc/sets", exist_ok=True); json.dump(meta, open("runs_lc/sets/meta.json", "w")); json.dump(ids, open("runs_lc/sets/test_embed.json", "w"))
STR = dict(V=(1.2, 0.60, 0.40), VP=(1.3, 0.80, 0.60), VC=(1.3, 0.75, 0.55), VCP=(1.35, 0.85, 0.65), V2x=(1.25, 0.65, 0.45), D=(1.4, 0.85, 0.60), Ap=(1.2, 0.75, 0.60), Atf=(1.3, 0.70, 0.60))
def scores(arm, seed):
    r = np.random.default_rng(zlib.crc32(f'{arm}|{seed}'.encode())); s = np.zeros((n, 3))
    for j in range(3): s[:, j] = STR[arm][j] * (Y[:, j] == 1) + r.normal(size=n)
    return s
def order(seed):   # shuffled order and ~2% rows missing, to test alignment and NaN handling
    r = np.random.default_rng(seed + 100); o = r.permutation(n); return o[: int(n * 0.98)]
def save_ext(path, arm, seed):
    o = order(seed); s = scores(arm, seed)[o]; os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, sample_id=np.array([ids[i] for i in o]), logp_no=np.concatenate([-s, np.zeros((len(o), 1))], 1), feat_junk=np.zeros((len(o), 2), np.float16))
def save_probe(path, arm, seed):
    o = order(seed + 7); s = scores(arm, seed)[o]; os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, test_sid=np.array([ids[i] for i in o]), cfg=np.array("x"), **{f"best_{f}": s[:, j] for j, f in enumerate(["mass", "calc", "asymmetry"])}, **{f"lin_{f}": s[:, j] for j, f in enumerate(["mass", "calc", "asymmetry"])})
def save_tf(path, arm, seed):
    o = order(seed + 11); s = scores(arm, seed)[o]; os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, test_sid=np.array([ids[i] for i in o]), best_vs_mean=np.array(0.6), **{f"score_{f}": s[:, j] for j, f in enumerate(["mass", "calc", "asymmetry"])})
def save_d(path, arm, seed):
    o = order(seed + 13); s = scores(arm, seed)[o]; os.makedirs(os.path.dirname(path), exist_ok=True)
    d = {"sample_id": [ids[i] for i in o]}
    for j, f in enumerate(["mass", "calc", "asymmetry"]):
        p = 1 / (1 + np.exp(-s[:, j])); d["p_" + f] = torch.tensor(np.stack([1 - p, p], 1), dtype=torch.float32)
    torch.save(d, path)
drop = "--drop_vc_seed2" in sys.argv
for sd in (0, 1, 2):
    save_ext(f"runs_lc/V/f1_s{sd}/ext.npz", "V", sd); save_probe(f"runs_lc/V/f1_s{sd}/probe.npz", "VP", sd); save_tf(f"runs_lc/A/tf_f1_s{sd}.npz", "Atf", sd); save_d(f"runs_lc/D/f1_s{sd}/dump_test_embed.pt", "D", sd)
    save_ext(f"runs_s8/V2x/f1_s{sd}/ext.npz", "V2x", sd)
    if not (drop and sd == 2): save_ext(f"runs_s8/VC/f1_s{sd}/ext.npz", "VC", sd); save_probe(f"runs_s8/VC/f1_s{sd}/probe.npz", "VCP", sd)
save_probe("runs_lc/A/probe_f1_s0.npz", "Ap", 0)
print("synthetic tree written to", root)
