#!/usr/bin/env python3
"""Report generation of trained stage-2 decoders (concept route 'cb' or opaque route 'opq') on a given list of rows (paper 4, s4v3eval). Loads the saved decoder, no training.
usage: PYTHONPATH=$PWD python3 analysis_embed/cb_generate_rows.py --stage1 runs/cb_embed_dinov2_s0 --s2 runs_stage2/grid2_cb_embed_dinov2_s0 --rows runs_vlm/_fresh/fresh_rows.json --out runs_vlm/_fresh/dec/cb_embed_dinov2_s0
       --check N : instead regenerate the first N rows of the grid-2 files and count identical outputs (determinism check)"""
import argparse, json, os, sys, time
import torch
from cbmammo.train_stage2 import batches, load_dump
from cbmammo.train_stage2_v2 import BIRADS
from cbmammo.train_stage2_v4 import ReportLMB, ext
from cbmammo.reports import extract
ap = argparse.ArgumentParser(); ap.add_argument("--stage1", required=True); ap.add_argument("--s2", required=True); ap.add_argument("--rows", default=""); ap.add_argument("--out", default="")
ap.add_argument("--check", type=int, default=0); ap.add_argument("--sets", default=""); a = ap.parse_args()
cfg = json.load(open(os.path.join(a.s2, "run.json")))["config"]; arm = json.load(open(os.path.join(a.stage1, "run.json")))["config"]["arm"]
dev = torch.device("cuda"); cond_dim = load_dump(a.stage1, "val")["cond"].shape[1]
model = ReportLMB(cfg["lm"], arm, cond_dim, cfg.get("k_tokens", 8), cfg.get("lora_r", 16), cfg.get("lora_alpha", 32), cfg.get("load_in_4bit", False), dtype=torch.bfloat16, lm=None, tokenizer=None).to(dev)
model.load_state_dict(torch.load(os.path.join(a.s2, "decoder.pt"), map_location=dev), strict=False); model.eval()
bs, mnt = cfg.get("gen_batch_size", 32), cfg.get("max_new_tokens", 160); t0 = time.time()
def gen(d, sel):
    cond = d["cond"][sel]; br = [BIRADS[int(i)] for i in d["p_birads"][sel].argmax(1)]; ce = ext(cond, br); res = []
    with torch.no_grad():
        for sl in batches(ce.shape[0], bs): res += model.generate(ce[sl].to(dev), mnt)
    return res, br
if a.check:
    for sp in (a.sets.split(",") if a.sets else ["test_embed", "test_vindr"]):
        d = load_dump(a.stage1, sp); old = [json.loads(l) for l in open(os.path.join(a.s2, f"gen_{sp}.jsonl"))][: a.check]
        g, _ = gen(d, [o["dump_row"] for o in old]); same = sum(x == o["generated"] for x, o in zip(g, old))
        print(sp, "identical", same, "of", len(old), flush=True)
    sys.exit(0)
rows = json.load(open(a.rows)); os.makedirs(a.out, exist_ok=True)
for sp in (a.sets.split(",") if a.sets else list(rows)):
    d = load_dump(a.stage1, sp); idx = {s: i for i, s in enumerate(d["sample_id"])}; sel = [idx[r["sample_id"]] for r in rows[sp]]
    g, br = gen(d, sel)
    with open(os.path.join(a.out, f"gen_{sp}.jsonl"), "w") as fo:
        for r, x, b in zip(rows[sp], g, br): fo.write(json.dumps({"sample_id": r["sample_id"], "birads_input": b, "generated": x, "slots_generated": extract(x)}) + "\n")
    print(sp, len(sel), f"{time.time()-t0:.0f}s", flush=True)
print("finished", flush=True)
