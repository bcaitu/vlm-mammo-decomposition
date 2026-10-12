#!/usr/bin/env python3
"""s6lc (paper 4, question 1): teacher-forced pass of Qwen3-VL-8B-Instruct (base, or with a LoRA adapter) over the prompt and a CANONICAL all-negative report with a fixed density sentence.
Saves (i) log P(' No') at the start of the mass, calc, asymmetry and distortion sentences (the model's own score, as in s4diag), (ii) hidden states (mean over image tokens and last prompt token)
at layers 0, 8, 16, 20, 24, 28, 32, 36, (iii) optionally (--tokens DIR) the layer-0 image tokens (output of vision tower + merger) pooled 2x2, per image, for the frozen-tower heads.
Batches are LEFT-padded so that the response occupies the last K positions of every row.
usage: PYTHONPATH=$PWD:$PWD/analysis_embed python3 analysis_embed/lc_score.py --rows a.json[,b.json] [--adapter DIR] --out OUT.npz [--tokens DIR] [--shard 0/5] [--no_feats]"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoProcessor, AutoModelForImageTextToText
from cbmammo import data as D
import vlm_common as C
ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--adapter", default=""); ap.add_argument("--out", required=True)
ap.add_argument("--tokens", default=""); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--shard", default="0/1")
ap.add_argument("--no_feats", action="store_true"); ap.add_argument("--blank", action="store_true", help="replace the images by black images of the same size (control)"); ap.add_argument("--workers", type=int, default=6)
a = ap.parse_args()
LAYERS = [0, 8, 16, 20, 24, 28, 32, 36]
SENT = [("mass", " No suspicious mass is seen."), ("calc", " No calcifications are seen."), ("asymmetry", " No asymmetry."), ("distortion", " No architectural distortion.")]
DENS = "The breasts are heterogeneously dense, which may obscure small masses (ACR C)."; TAIL = " Impression: BI-RADS 1: negative."
T = DENS + "".join(s for _, s in SENT) + TAIL
M = ["manifests/cbis_ddsm.jsonl", "manifests/vindr.jsonl", "manifests/embed.jsonl"]
recs = {r["sample_id"]: r for r in D.load_manifest(M)}
ids = []
for f in a.rows.split(","):
    for x in json.load(open(f)):
        if x not in ids: ids.append(x)
si, sn = [int(x) for x in a.shard.split("/")]; ids = ids[si::sn]
if a.limit: ids = ids[: a.limit]
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
proc = AutoProcessor.from_pretrained(C.MODEL); tok = proc.tokenizer; pad = tok.pad_token_id
enc = tok(T, add_special_tokens=False, return_offsets_mapping=True); offs = enc["offset_mapping"]
K = len(enc["input_ids"]) + len(tok("<|im_end|>", add_special_tokens=False)["input_ids"]); NO = tok.convert_tokens_to_ids("ĠNo")
rel = [next(k for k, o in enumerate(offs) if o[0] == T.find(s)) for _, s in SENT]
class DS(torch.utils.data.Dataset):
    def __len__(s): return len(ids)
    def __getitem__(s, i):
        im = [C.load_img(ids[i], v) for v in C.views_of(recs[ids[i]])]
        if a.blank: im = [C.blank_img(x) for x in im]
        return C.encode_sample(proc, recs[ids[i]], [(T, False)], im, "train"), ids[i]
def coll(b):
    it, sid = zip(*b); return C.collate(list(it), pad, left=True), list(sid)
dl = torch.utils.data.DataLoader(DS(), batch_size=a.bs, shuffle=False, num_workers=a.workers, collate_fn=coll, prefetch_factor=4)
model = AutoModelForImageTextToText.from_pretrained(C.MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
model.eval(); IMG = tok.convert_tokens_to_ids("<|image_pad|>")
feats = (not a.no_feats) or bool(a.tokens)
LOGP = np.full((len(ids), len(SENT)), np.nan, dtype=np.float32); FL = FI = None; index = []; off = 0; tokf = None
if not a.no_feats: FI = np.empty((len(ids), len(LAYERS), 4096), np.float16); FL = np.empty((len(ids), len(LAYERS), 4096), np.float16)
if a.tokens:
    os.makedirs(a.tokens, exist_ok=True); tokf = open(os.path.join(a.tokens, f"tokens_shard{si}.bin"), "wb")
t0 = time.time(); i0 = 0; tm = dict(data=0.0, fwd=0.0, feat=0.0); tl = time.time()
with torch.no_grad():
    for bi, (b, sids) in enumerate(dl):
        tm['data'] += time.time() - tl; ta = time.time()
        b = {k: v.cuda(non_blocking=True) for k, v in b.items() if k != "labels"}; B = b["input_ids"].shape[0]; L = b["input_ids"].shape[1]; x = b["input_ids"]
        out = model(**b, output_hidden_states=feats, logits_to_keep=K + 1)
        lp = torch.log_softmax(out.logits.float(), -1); torch.cuda.synchronize(); tb = time.time(); tm['fwd'] += tb - ta
        for j, t in enumerate(rel):
            assert bool((x[:, L - K + t] == NO).all()), "slot token is not ' No'"
            LOGP[i0: i0 + B, j] = lp[:, t, NO].cpu().numpy()
        if feats:
            hs = out.hidden_states; im = (x == IMG); last = L - K - 1
            if not a.no_feats:
                for r in range(B):
                    FI[i0 + r] = torch.stack([hs[l][r][im[r]].float().mean(0) for l in LAYERS]).half().cpu().numpy()
                    FL[i0 + r] = torch.stack([hs[l][r, last].float() for l in LAYERS]).half().cpu().numpy()
            if a.tokens:
                thw = b["image_grid_thw"]; p = 0
                for r, sid in enumerate(sids):
                    toks = hs[0][r][im[r]]; pos = 0; parts = []; meta = []
                    for _ in range(len(C.views_of(recs[sid]))):
                        t_, h_, w_ = [int(v) for v in thw[p].tolist()]; p += 1; n = (t_ * h_ * w_) // 4; xx = toks[pos: pos + n]; pos += n
                        assert t_ == 1 and xx.shape[0] == (h_ // 2) * (w_ // 2), (t_, h_, w_, xx.shape)
                        y = Fn.avg_pool2d(xx.view(h_ // 2, w_ // 2, -1).permute(2, 0, 1)[None].float(), 2, 2, ceil_mode=True)[0]
                        parts.append(y.permute(1, 2, 0).reshape(-1, y.shape[0]).half().cpu().numpy()); meta.append([int(y.shape[1]), int(y.shape[2])])
                    assert pos == toks.shape[0]
                    arr = np.concatenate(parts, 0); tokf.write(arr.tobytes()); index.append([sid, off, int(arr.shape[0]), meta]); off += int(arr.shape[0])
        i0 += B; del out; tl = time.time(); tm['feat'] += tl - tb
        if (bi + 1) % 50 == 0: print(f"{i0}/{len(ids)} {time.time()-t0:.0f}s data {tm['data']:.0f}s fwd {tm['fwd']:.0f}s feat {tm['feat']:.0f}s", flush=True)
if tokf:
    tokf.close(); json.dump(dict(dim=int(hs[0].shape[-1]), dtype="float16", rows=index), open(os.path.join(a.tokens, f"index_shard{si}.json"), "w"))
kw = dict(sample_id=np.array(ids), layers=np.array(LAYERS), logp_no=LOGP, slots=np.array([s for s, _ in SENT]))
if FL is not None: kw.update(F_last=FL, F_img=FI)
np.savez(a.out, **kw); print("finished", len(ids), round(time.time() - t0), flush=True)
