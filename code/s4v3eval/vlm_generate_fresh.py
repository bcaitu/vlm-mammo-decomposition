#!/usr/bin/env python3
"""Greedy report generation of the fine-tuned VLM on the fresh evaluation rows (paper 4, s4v3eval). Same prompt and decoding as vlm_generate.py.
variants: orig, swap (images of another breast of the same set, random derangement of the first --n_v2 rows), blank (black image).
usage: PYTHONPATH=$PWD python3 analysis_embed/vlm_generate_fresh.py --adapter runs_vlm/v1_s0 --rows runs_vlm/_fresh/fresh_rows.json --out runs_vlm/v1_s0/gen_fresh --variants orig,swap,blank"""
import argparse, json, os, sys, time, random
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoProcessor, AutoModelForImageTextToText
from cbmammo import data as D
from cbmammo.reports import extract
import vlm_common as C
ap = argparse.ArgumentParser(); ap.add_argument("--adapter", default=""); ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True); ap.add_argument("--model", default=C.MODEL)
ap.add_argument("--variants", default="orig"); ap.add_argument("--sets", default=""); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--max_new", type=int, default=170); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--n_v2", type=int, default=300)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
ALL = ["manifests/cbis_ddsm.jsonl", "manifests/vindr.jsonl", "manifests/embed.jsonl", "manifests/cmmd.jsonl", "manifests/cdd_cesm.jsonl", "manifests/inbreast.jsonl"]
recs = {r["sample_id"]: r for r in D.load_manifest(ALL)}; fresh = json.load(open(a.rows))
SETS = a.sets.split(",") if a.sets else list(fresh)
proc = AutoProcessor.from_pretrained(a.model)
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
model.eval(); pad = proc.tokenizer.pad_token_id
@torch.no_grad()
def run(rows, imgs_of):
    outs = []
    for i in range(0, len(rows), a.bs):
        items = []
        for r in rows[i:i + a.bs]:
            r2, im = imgs_of(r); items.append(C.encode_sample(proc, r2, None, im, "train"))
        b = {k: v.cuda() for k, v in C.collate(items, pad, left=True).items()}
        g = model.generate(**b, max_new_tokens=a.max_new, do_sample=False)
        outs += proc.tokenizer.batch_decode(g[:, b["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs
t0 = time.time()
for sp in SETS:
    rows = [recs[r["sample_id"]] for r in fresh[sp]]
    if a.limit: rows = rows[: a.limit]
    for var in a.variants.split(","):
        sub = rows if var == "orig" else rows[: a.n_v2 if not a.limit else a.limit]
        if var == "orig": f = lambda r: (r, [C.load_img(r["sample_id"], v) for v in C.views_of(r)])
        elif var == "swap":
            perm = list(range(len(sub))); random.Random(7).shuffle(perm)
            for j in range(len(perm)):
                if perm[j] == j: perm[j], perm[(j + 1) % len(perm)] = perm[(j + 1) % len(perm)], perm[j]
            other = {sub[j]["sample_id"]: sub[perm[j]] for j in range(len(sub))}
            f = lambda r: ({**r, "views": other[r["sample_id"]]["views"]}, [C.load_img(other[r["sample_id"]]["sample_id"], v) for v in C.views_of(other[r["sample_id"]])])
        elif var == "blank": f = lambda r: (r, [C.blank_img(C.load_img(r["sample_id"], v)) for v in C.views_of(r)])
        else: raise SystemExit("unknown variant " + var)
        texts = run(sub, f)
        with open(os.path.join(a.out, f"gen_{sp}_{var}.jsonl"), "w") as fo:
            for r, t in zip(sub, texts): fo.write(json.dumps(dict(sample_id=r["sample_id"], variant=var, text=t, slots=extract(t), values=r["values"])) + "\n")
        print(f"{sp} {var} {len(sub)} rows {time.time()-t0:.0f}s", flush=True)
print("finished", round(time.time() - t0), flush=True)
