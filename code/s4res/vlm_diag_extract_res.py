#!/usr/bin/env python3
"""Resolution test (paper 4, s4res): same extraction as vlm_diag_extract.py with an image cache of another resolution (--cache) and the number of image tokens saved.
Original description: Perception-versus-verbalization diagnostic of the VLM (paper 4, s4diag): one teacher-forced pass over the prompt and a canonical all-negative report.
Saves (i) hidden states at the last prompt token and the mean over image tokens at selected layers and (ii) log P(' No') at the start of the mass, calc, asymmetry and distortion sentences,
given the density sentence written by the model itself (from the greedy generation) and canonical negative sentences before the slot.
usage: PYTHONPATH=$PWD python3 analysis_embed/vlm_diag_extract.py --adapter runs_vlm/v1_s0 --rows runs_vlm/_diag/dev_rows.json --gen runs_vlm/v1_s0/gen_diag/gen_dev_embed_val_orig.jsonl --out runs_vlm/_diag/feat_v1_s0.npz  [--limit 8]"""
import argparse, json, os, re, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoProcessor, AutoModelForImageTextToText
from cbmammo import data as D
import vlm_common as C
ap = argparse.ArgumentParser(); ap.add_argument("--adapter", default=""); ap.add_argument("--rows", required=True); ap.add_argument("--gen", default=""); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--cache", default="")
a = ap.parse_args()
if a.cache: C.CACHE = a.cache
LAYERS = [0, 8, 16, 20, 24, 28, 32, 36]
SENT = [("mass", " No suspicious mass is seen."), ("calc", " No calcifications are seen."), ("asymmetry", " No asymmetry."), ("distortion", " No architectural distortion.")]
TAIL = " Impression: BI-RADS 1: negative."; FALLBACK = "The breasts are heterogeneously dense, which may obscure small masses (ACR C)."
recs = {r["sample_id"]: r for r in D.load_manifest(["manifests/cbis_ddsm.jsonl", "manifests/vindr.jsonl", "manifests/embed.jsonl"])}
rows = json.load(open(a.rows))["dev_embed_val"]
if a.limit: rows = rows[: a.limit]
gen = {}
if a.gen:
    for l in open(a.gen):
        j = json.loads(l); gen[j["sample_id"]] = j["text"]
proc = AutoProcessor.from_pretrained(C.MODEL); tok = proc.tokenizer
model = AutoModelForImageTextToText.from_pretrained(C.MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
model.eval(); IMG = tok.convert_tokens_to_ids("<|image_pad|>")
n = len(rows); FL, FI = [], []; NTOK = []
LOGP = np.full((n, len(SENT)), np.nan, dtype=np.float32); t0 = time.time()
with torch.no_grad():
    for i, r in enumerate(rows):
        rec = recs[r["sample_id"]]; imgs = [C.load_img(r["sample_id"], v) for v in C.views_of(rec)]
        G = gen.get(r["sample_id"], ""); m = re.search(r"^.*?\(ACR [A-D]\)\.", G); dens = m.group(0) if m else FALLBACK
        T = dens + "".join(s for _, s in SENT) + TAIL
        n0 = C.encode_sample(proc, rec, None, imgs, "train")["input_ids"].shape[1]
        item = C.encode_sample(proc, rec, [(T, False)], imgs, "train"); b = {k: v.cuda() for k, v in C.collate([item], tok.pad_token_id).items() if k != "labels"}
        out = model(**b, output_hidden_states=True); ids = b["input_ids"][0]
        enc = tok(T, add_special_tokens=False, return_offsets_mapping=True); offs = enc["offset_mapping"]
        poss = []
        for j, (name, s) in enumerate(SENT):
            ci = T.find(s); t = next(k for k, o in enumerate(offs) if o[0] == ci); pos = n0 + t
            assert tok.convert_ids_to_tokens(int(ids[pos])) == "ĠNo", (name, tok.convert_ids_to_tokens(int(ids[pos])))
            poss.append(pos)
        lp = torch.log_softmax(out.logits[0, [p - 1 for p in poss]].float(), -1)
        for j, pos in enumerate(poss): LOGP[i, j] = lp[j, int(ids[pos])].item()
        imask = ids == IMG; NTOK.append(int(imask.sum()))
        FL.append(np.stack([out.hidden_states[L][0, n0 - 1].float().cpu().numpy() for L in LAYERS]).astype(np.float16))
        FI.append(np.stack([out.hidden_states[L][0][imask].float().mean(0).cpu().numpy() for L in LAYERS]).astype(np.float16))
        del out
        if (i + 1) % 200 == 0: print(f"{i+1}/{n} {time.time()-t0:.0f}s", flush=True)
np.savez(a.out, sample_id=np.array([r["sample_id"] for r in rows]), layers=np.array(LAYERS), F_last=np.stack(FL), F_img=np.stack(FI), logp_no=LOGP, n_img_tokens=np.array(NTOK), slots=np.array([s for s, _ in SENT]))
print("finished", n, round(time.time() - t0), flush=True)
