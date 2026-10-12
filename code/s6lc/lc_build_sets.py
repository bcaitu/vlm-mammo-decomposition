#!/usr/bin/env python3
"""s6lc (paper 4, question 1: learning curves): nested patient-level training subsets and the evaluation lists.
Pool = pooled train manifests (CBIS-DDSM + VinDr-Mammo + EMBED), the v1 training set. Subsets are nested prefixes of a seeded patient order (sha1), sizes 1/32 ... 1 of the pool rows.
Evaluation: VS = uniform sample of 2,000 EMBED val breasts (selection of checkpoints, probe regularisation); test = all EMBED test breasts (+ EMBED GE-scanner split for the shift check).
usage: PYTHONPATH=$PWD:$PWD/analysis_embed python3 analysis_embed/lc_build_sets.py --out runs_lc/sets"""
import argparse, hashlib, json, os, random, sys
from collections import Counter, defaultdict
from cbmammo import data as D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlm_common as C
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--vs_seed", type=int, default=20261007); ap.add_argument("--vs_n", type=int, default=2000)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
M = ["manifests/cbis_ddsm.jsonl", "manifests/vindr.jsonl", "manifests/embed.jsonl"]
recs = D.load_manifest(M)
FRACS = [("f32", 1 / 32), ("f16", 1 / 16), ("f8", 1 / 8), ("f4", 1 / 4), ("f2", 1 / 2), ("f1", 1.0)]
SLOTS = ["mass", "calc", "asymmetry", "distortion"]
def pk(r): return f"{r['dataset']}|{r['patient_id']}"
def yv(r): return [int(r["labels"][s]) for s in SLOTS]
def yvals(r):
    out = []
    for s in SLOTS:
        v = (r["values"] or {}).get(s); out.append(1 if v == "present" else 0 if v == "absent" else -1)
    return out
mis = Counter()
for r in recs:
    for s, x, y in zip(SLOTS, yv(r), yvals(r)):
        if x != y: mis[(s, x, y)] += 1
print("labels vs values mismatches (slot, labels, values): count", dict(mis))
train = [r for r in recs if r["split"] == "train"]
by_pat = defaultdict(list)
for r in train: by_pat[pk(r)].append(r)
N = len(train); print("pool rows", N, "patients", len(by_pat))
targets = {n: round(N * f) for n, f in FRACS}
summary = []; allfiles = {}
for seed in (0, 1, 2):
    pats = sorted(by_pat, key=lambda p: hashlib.sha1(f"{seed}|{p}".encode()).hexdigest())
    cum, cuts, ti = 0, {}, 0
    for i, p in enumerate(pats, 1):
        cum += len(by_pat[p])
        while ti < len(FRACS) - 1 and cum >= targets[FRACS[ti][0]]:
            cuts[FRACS[ti][0]] = i; ti += 1
    cuts["f1"] = len(pats)
    for name, _ in FRACS:
        rows = [r for p in pats[: cuts[name]] for r in by_pat[p]]
        ids = sorted(r["sample_id"] for r in rows)
        fn = f"subset_s{seed}_{name}.json"; json.dump(ids, open(os.path.join(a.out, fn), "w")); allfiles[fn] = ids
        c = {s: Counter(int(x[i]) for x in map(yv, rows)) for i, s in enumerate(SLOTS)}
        summary.append(dict(seed=seed, frac=name, target=targets[name], rows=len(rows), patients=cuts[name], embed_rows=sum(r["dataset"] == "embed" for r in rows),
                            **{f"{s}_pos": c[s][1] for s in SLOTS[:3]}, **{f"{s}_unk": c[s][-1] for s in SLOTS[:3]}))
emb_val = sorted([r for r in recs if r["dataset"] == "embed" and r["split"] == "val"], key=lambda r: r["sample_id"])
vs = sorted(random.Random(a.vs_seed).sample(emb_val, a.vs_n), key=lambda r: r["sample_id"])
test = sorted([r for r in recs if r["dataset"] == "embed" and r["split"] == "test"], key=lambda r: r["sample_id"])
ge = sorted([r for r in recs if r["dataset"] == "embed" and r["split"] == "test_embed_ge"], key=lambda r: r["sample_id"])
for fn, rs in (("val_sample.json", vs), ("test_embed.json", test), ("test_ge.json", ge)):
    json.dump([r["sample_id"] for r in rs], open(os.path.join(a.out, fn), "w"))
meta = {}
for r in train + vs + test + ge: meta[r["sample_id"]] = dict(p=pk(r), d=r["dataset"], s=r["split"], y=yv(r))
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"))
miss = {}
for nm, rs in (("pool", train), ("vs", vs), ("test", test), ("ge", ge)):
    bad = [r["sample_id"] for r in rs if not all(os.path.exists(os.path.join(C.CACHE, C.fname(r["sample_id"], v))) for v in C.views_of(r))]
    miss[nm] = bad; print("cache missing", nm, len(bad), "of", len(rs))
json.dump({k: [{"sample_id": x} for x in v] for k, v in miss.items() if v}, open(os.path.join(a.out, "rows_to_cache.json"), "w"))
pos = lambda rs, i: sum(1 for r in rs if yv(r)[i] == 1)
for nm, rs in (("vs", vs), ("test", test), ("ge", ge)):
    print(nm, len(rs), "patients", len({pk(r) for r in rs}), {s: (pos(rs, i), sum(1 for r in rs if yv(r)[i] == 0)) for i, s in enumerate(SLOTS[:3])})
import pandas as pd
df = pd.DataFrame(summary); df.to_csv(os.path.join(a.out, "sets_summary.csv"), index=False); print(df.to_string(index=False))
