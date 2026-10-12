#!/usr/bin/env python3
"""LoRA fine-tuning of Qwen3-VL-8B-Instruct for the mammography report (paper 4, V1). Language-model linear layers only (vision tower frozen).
usage: PYTHONPATH=$PWD python3 analysis_embed/vlm_train.py --out runs_vlm/v1_s0 --seed 0 [--max_steps 20]"""
import argparse, json, math, os, sys, time, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model
from cbmammo import data as D
import vlm_common as C
from vlm_text import build_segments
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--model", default=C.MODEL)
ap.add_argument("--bs", type=int, default=4); ap.add_argument("--accum", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--workers", type=int, default=6); ap.add_argument("--val_every", type=int, default=500); ap.add_argument("--n_val", type=int, default=1000); ap.add_argument("--train_split", default="train"); ap.add_argument("--val_split", default="val")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
M = ["manifests/cbis_ddsm.jsonl", "manifests/vindr.jsonl", "manifests/embed.jsonl"]
train = D.load_manifest(M, splits=[a.train_split]); val = D.load_manifest(M, splits=[a.val_split])
random.Random(0).shuffle(val)
idx = {json.loads(l)["sample_id"]: 1 for l in open(os.path.join(C.CACHE, "index.jsonl"))} if os.path.exists(os.path.join(C.CACHE, "index.jsonl")) else None
def has(r): return all(os.path.exists(os.path.join(C.CACHE, C.fname(r["sample_id"], v))) for v in C.views_of(r)) and len(C.views_of(r)) > 0
if idx is None: train = [r for r in train if has(r)]; print("index missing: using rows whose cache files exist", len(train), flush=True)
val = [r for r in val if has(r)][: a.n_val]
proc = AutoProcessor.from_pretrained(a.model)
class DS(torch.utils.data.Dataset):
    def __init__(s, recs): s.recs = recs
    def __len__(s): return len(s.recs)
    def __getitem__(s, i):
        r = s.recs[i]; return C.encode_sample(proc, r, build_segments(r["values"], r["side"]))
pad = proc.tokenizer.pad_token_id
g = torch.Generator(); g.manual_seed(a.seed)
dl = torch.utils.data.DataLoader(DS(train), batch_size=a.bs, shuffle=True, generator=g, num_workers=a.workers, collate_fn=lambda b: C.collate(b, pad), drop_last=True, persistent_workers=True, prefetch_factor=4)
vdl = torch.utils.data.DataLoader(DS(val), batch_size=8, shuffle=False, num_workers=2, collate_fn=lambda b: C.collate(b, pad))
model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
params = [p for p in model.parameters() if p.requires_grad]; print("trainable", sum(p.numel() for p in params), flush=True)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
total = (len(dl) // a.accum) if not a.max_steps else a.max_steps; warm = max(1, int(0.03 * total))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
def to_cuda(b): return {k: v.cuda(non_blocking=True) for k, v in b.items()}
@torch.no_grad()
def val_loss():
    model.eval(); tot, n = 0.0, 0
    for b in vdl:
        o = model(**to_cuda(b)); k = int((b["labels"][:, 1:] != -100).sum()); tot += o.loss.item() * k; n += k
    model.train(); return tot / max(n, 1)
log = []; model.train(); t0 = time.time(); step = 0; run = 0.0
for it, b in enumerate(dl):
    o = model(**to_cuda(b)); (o.loss / a.accum).backward(); run += o.loss.item() / a.accum
    if (it + 1) % a.accum == 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 25 == 0: print(f"step {step}/{total} loss {run/25:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True); log.append(dict(step=step, loss=run / 25)); run = 0.0
        if step % a.val_every == 0 or step == total:
            vl = val_loss(); print(f"VAL step {step} loss {vl:.4f}", flush=True); log.append(dict(step=step, val_loss=vl)); model.save_pretrained(a.out)
        if step >= total: break
model.save_pretrained(a.out)
json.dump(dict(log=log, steps=step, seed=a.seed, model=a.model, lr=a.lr, bs=a.bs, accum=a.accum, rows=len(train), seconds=round(time.time() - t0)), open(os.path.join(a.out, "train_log.json"), "w"))
print("finished", step, round(time.time() - t0), flush=True)
