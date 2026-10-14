#!/usr/bin/env python3
"""s10 (paper 4): helpers shared by the two analyses s10an (analyses on saved scores) and s10db (D with the update budget of V).
Reproduces the test set, labels, patient-level bootstrap (2,000 resamples of test patients, seed 20261007, identical to s6lc and s8), the score loaders of s6lc / s8 and the verdict rules.
Run from ~/cbmammo with the cbmammo environment: ~/envs/cbmammo/bin/python analysis_embed/<script>.py"""
import json, os
import numpy as np, pandas as pd, torch

DBROOT = os.environ.get("S10DB_ROOT", "runs_s10db")       # only changed for the code test on stand-in files (see s10db_prereg_ru.md, 4.2); the main analysis uses the default
FIND = ["mass", "calc", "asymmetry"]; J = {f: i for i, f in enumerate(FIND)}; SEEDS = [0, 1, 2]; SZ = ["f32", "f16", "f8", "f4", "f2", "f1"]


class Ctx:
    def __init__(self, sets="runs_lc/sets", boot=2000):
        self.sets = sets; self.R = boot
        self.meta = json.load(open(f"{sets}/meta.json")); self.te = json.load(open(f"{sets}/test_embed.json")); self.n = n = len(self.te)
        self.tix = {s: i for i, s in enumerate(self.te)}
        self.Y = {f: np.array([self.meta[s]["y"][J[f]] for s in self.te]) for f in FIND}
        self.pats = np.array([self.meta[s]["p"] for s in self.te]); self.up, self.pinv = np.unique(self.pats, return_inverse=True)
        rng = np.random.default_rng(20261007)
        cnt = rng.multinomial(len(self.up), np.full(len(self.up), 1.0 / len(self.up)), size=boot).astype(np.float32)
        self.Wall = np.vstack([np.ones((1, len(self.up)), np.float32), cnt])[:, self.pinv]       # row 0 = point estimate
        self.jit = np.random.default_rng(1).random(n) * 1e-9

    # ---- score loading (as in lc_analysis.py and s8_analysis.py)
    def _zload(self, p): return np.load(p, allow_pickle=True) if os.path.exists(p) else None

    def align(self, ids, vec):
        ix = {s: i for i, s in enumerate(ids)}; out = np.full(self.n, np.nan, np.float32)
        for s, i in self.tix.items():
            if s in ix: out[i] = vec[ix[s]]
        return out

    def _logp(self, p):
        z = self._zload(p); return None if z is None else {f: self.align([str(x) for x in z["sample_id"]], -z["logp_no"][:, J[f]]) for f in FIND}

    def _probe(self, p):
        z = self._zload(p); return None if z is None else {f: self.align([str(x) for x in z["test_sid"]], z["best_" + f]) for f in FIND}

    def _dump(self, p):
        if not os.path.exists(p): return None
        d = torch.load(p, map_location="cpu"); return {f: self.align(list(d["sample_id"]), d["p_" + f][:, 1].numpy()) for f in FIND}

    def scores(self, arm, sz="f1", seed=0):
        """-> {finding: array over test rows (nan if absent)} or None."""
        try:
            if arm == "V": return self._logp(f"runs_lc/V/{sz}_s{seed}/ext.npz")
            if arm == "V_ctl": return self._logp(f"runs_lc/V/ctl_s{seed}/test.npz") if sz == "f1" else None
            if arm == "V_blank": return self._logp(f"runs_lc/V/blank_s{seed}/test.npz") if sz == "f1" else None
            if arm == "VP": return self._probe(f"runs_lc/V/{sz}_s{seed}/probe.npz")
            if arm == "A_probe": return self._probe(f"runs_lc/A/probe_{sz}_s{0 if sz == 'f1' else seed}.npz")
            if arm == "A_tf":
                z = self._zload(f"runs_lc/A/tf_{sz}_s{seed}.npz"); return None if z is None else {f: self.align([str(x) for x in z["test_sid"]], z["score_" + f]) for f in FIND}
            if arm == "D": return self._dump(f"runs_lc/D/{sz}_s{seed}/dump_test_embed.pt")
            if arm == "VC": return self._logp(f"runs_s8/VC/f1_s{seed}/ext.npz") if sz == "f1" else None
            if arm == "VCP": return self._probe(f"runs_s8/VC/f1_s{seed}/probe.npz") if sz == "f1" else None
            if arm == "V2x": return self._logp(f"runs_s8/V2x/f1_s{seed}/ext.npz") if sz == "f1" else None
            if arm == "D_bV": return self._dump(f"{DBROOT}/DB/{sz}_s{seed}/dump_test_embed.pt")
            if arm == "D_b2x": return self._dump(f"{DBROOT}/DB/f1x2_s{seed}/dump_test_embed.pt") if sz == "f1" else None
        except Exception as e:
            print("load error", arm, sz, seed, repr(e)[:150]); return None

    # ---- AUC with the shared bootstrap (rowmask: optional boolean mask over test rows)
    def boot_auc(self, s, f, rowmask=None):
        k = (self.Y[f] >= 0) & ~np.isnan(s)
        if rowmask is not None: k = k & rowmask
        if k.sum() < 50: return None
        y = self.Y[f][k].astype(np.float32); sc = s[k] + self.jit[k]; o = np.argsort(sc); W = self.Wall[:, k][:, o]; ys = y[o]
        pos = W * ys; neg = W * (1 - ys); cumneg = np.cumsum(neg, axis=1) - neg
        return (pos * cumneg).sum(1) / (pos.sum(1) * neg.sum(1))

    def arm_auc(self, arm, sz="f1", seeds=SEEDS, rowmask=None):
        """-> {finding: array (n_seeds, R+1)}"""
        out = {}
        for f in FIND:
            lst = []
            for sd in seeds:
                sc = self.scores(arm, sz, sd)
                if sc is None: continue
                b = self.boot_auc(sc[f], f, rowmask)
                if b is not None: lst.append(b)
            if lst: out[f] = np.array(lst)
        return out


def ci(d): return float(np.percentile(d[1:], 2.5)), float(np.percentile(d[1:], 97.5))


def cls(d, marg):
    """s6lc rule (K1, K2, K3): position of the interval relative to +-marg."""
    lo, hi = ci(d)
    if lo > marg: return "above by more than %.2f" % marg
    if hi < -marg: return "below by more than %.2f" % marg
    if lo > -marg and hi < marg: return "equivalent within +-%.2f" % marg
    if hi < marg: return "not above by more than %.2f" % marg
    return "inconclusive"


def verdict_gain(d, m=0.03):
    """s8 rule (S1, S3, S4)."""
    lo, hi = ci(d)
    if lo > 0: return "adds (lower bound > 0)" + (" and point >= 0.05" if d[0] >= 0.05 else "")
    if lo > -m and hi < m: return "equivalent within +-%.2f" % m
    if hi < 0: return "reduces (upper bound < 0)"
    return "unclear"


def verdict_gap(d, m=0.03):
    """s8 rule (S2, S5)."""
    lo, hi = ci(d)
    if hi < 0: return "below (upper bound < 0)"
    if lo > -m: return "not below (lower bound > -%.2f)" % m
    return "unclear"


def mean_boot(A): return A.mean(0)          # seed mean of the (n_seeds, R+1) array -> (R+1,)
