
import sys, json, random, collections
sys.path.insert(0, "analysis_embed")
from cbmammo import data as D
from cbmammo.reports import extract
from vlm_text import build_segments, text_of
recs = D.load_manifest(["manifests/embed.jsonl", "manifests/vindr.jsonl", "manifests/cbis_ddsm.jsonl"], splits=["train"])
random.Random(0).shuffle(recs); recs = recs[:20000]
SL = ["density", "mass", "mass_shape", "mass_margin", "calc", "calc_morphology", "calc_distribution", "asymmetry", "distortion", "birads"]
tot = collections.Counter(); ok = collections.Counter(); bad = collections.defaultdict(list)
for r in recs:
    v = r["values"]; segs = build_segments(v, r["side"]); t = text_of(segs); e = extract(t)
    for k in SL:
        if v.get(k) is None: continue
        if k in ("mass_shape", "mass_margin") and v.get("mass") == "absent": continue
        if k in ("calc_morphology", "calc_distribution") and v.get("calc") == "absent": continue
        tot[k] += 1; ok[k] += int(e[k] == v[k])
        if e[k] != v[k] and len(bad[k]) < 2: bad[k].append((v[k], e[k], t[:200]))
print({k: (tot[k], round(ok[k] / tot[k], 4)) for k in SL if tot[k]})
for k, b in bad.items(): print(k, b)
r = next(x for x in recs if x["values"].get("mass") == "present" and x["values"].get("mass_shape")); print(build_segments(r["values"], r["side"]))
r = next(x for x in recs if x["values"].get("mass") is None); print(build_segments(r["values"], r["side"]))
