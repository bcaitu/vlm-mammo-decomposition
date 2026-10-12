#!/usr/bin/env python3
"""Greedy report generation of the (fine-tuned) VLM on the grid-2 evaluation subsample of the eight test sets (paper 4).
variants: orig (own images), swap (images of another breast of the same set), blank (black image). V2 variants use the first --n_v2 rows of each set.
usage: PYTHONPATH=$PWD python3 analysis_embed/vlm_generate.py --adapter runs_vlm/v1_s0 --out runs_vlm/v1_s0/gen [--task zeroshot] [--sets test_vindr] [--limit 16]"""
import argparse, json, os, sys, time, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoProcessor, AutoModelForImageTextToText
from cbmammo import data as D
from cbmammo.reports import extract
import vlm_common as C
ap = argparse.ArgumentParser(); ap.add_argument("--adapter", default=""); ap.add_argument("--out", required=True); ap.add_argument("--model", default=C.MODEL); ap.add_argument("--task", default="train")
ap.add_argument("--variants", default="orig"); ap.add_argument("--sets", default=""); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--max_new", type=int, default=170); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--n_v2", type=int, default=300)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
SETS = a.sets.split(",") if a.sets else ["test_cbis_ddsm", "test_cdd_cesm", "test_cmmd", "test_embed", "test_embed_fuji", "test_embed_ge", "test_inbreast", "test_vindr"]
ALL = ["manifests/cbis_ddsm.jsonl", "manifests/vindr.jsonl", "manifests/embed.jsonl", "manifests/cmmd.jsonl", "manifests/cdd_cesm.jsonl", "manifests/inbreast.jsonl"]
recs = {r["sample_id"]: r for r in D.load_manifest(ALL)}
proc = AutoProcessor.from_pretrained(a.model)
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
model.eval(); pad = proc.tokenizer.pad_token_id
@torch.no_grad()
def run(rows, variant, imgs_of):
    outs = []
    for i in range(0, len(rows), a.bs):
        chunk = rows[i:i + a.bs]
        items = []
        for r in chunk:
            r2, im = imgs_of(r); items.append(C.encode_sample(proc, r2, None, im, a.task))
        b = {k: v.cuda() for k, v in C.collate(items, pad, left=True).items()}
        g = model.generate(**b, max_new_tokens=a.max_new, do_sample=False)
        txt = proc.tokenizer.batch_decode(g[:, b["input_ids"].shape[1]:], skip_special_tokens=True)
        outs += txt
    return outs
run_dir = "runs/cb_embed_dinov2_s0"; g2 = "runs_stage2/grid2_cb_embed_dinov2_s0"; t0 = time.time()
for sp in SETS:
    sids = torch.load(os.path.join(run_dir, f"dump_{sp}.pt"), map_location="cpu")["sample_id"]
    rows_idx = [json.loads(l)["dump_row"] for l in open(os.path.join(g2, f"gen_{sp}.jsonl"))]
    rows = [recs[sids[i]] for i in rows_idx]
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
        elif var == "blank":
            f = lambda r: (r, [C.blank_img(C.load_img(r["sample_id"], v)) for v in C.views_of(r)])
        else: raise SystemExit("unknown variant " + var)
        texts = run(sub, var, f)
        with open(os.path.join(a.out, f"gen_{sp}_{var}.jsonl"), "w") as fo:
            for r, t in zip(sub, texts): fo.write(json.dumps(dict(sample_id=r["sample_id"], variant=var, text=t, slots=extract(t), values=r["values"])) + "\n")
        print(f"{sp} {var} {len(sub)} rows {time.time()-t0:.0f}s", flush=True)
print("finished", round(time.time() - t0), flush=True)
