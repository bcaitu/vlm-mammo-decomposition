#!/usr/bin/env python3
"""s6lc (paper 4, question 1): LoRA fine-tuning of Qwen3-VL-8B-Instruct on a subset of the pooled training manifests with a FIXED number of optimiser steps.
Recipe identical to vlm_train.py (v1): LoRA r=16 alpha=32 on the linear layers of the language model, vision tower frozen, AdamW lr 1e-4 (no weight decay), micro-batch 4 x accumulation 4,
3% linear warm-up + cosine decay over the step budget, gradient checkpointing, bf16. The subset is iterated repeatedly (new shuffle each epoch) until the step budget is reached.
Adapters are saved at the steps listed in --ckpt (ckpt_<step>/) so that the checkpoint can be chosen on the validation sample.
usage: PYTHONPATH=$PWD:$PWD/analysis_embed python3 analysis_embed/lc_train.py --rows runs_lc/sets/subset_s0_f16.json --out runs_lc/V/f16_s0 --seed 0 --steps 2295 --ckpt 765,1530,2295"""
import argparse, json, math, os, sys, time, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model
from cbmammo import data as D
import vlm_common as C
from vlm_text import build_segments
ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=2295); ap.add_argument("--ckpt", default=""); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--workers", type=int, default=6); ap.add_argument("--max_rows", type=int, default=0); ap.add_argument("--model", default=C.MODEL)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
if os.path.exists(os.path.join(a.out, "train_log.json")): print("[skip] finished:", a.out); sys.exit(0)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
M = ["manifests/cbis_ddsm.jsonl", "manifests/vindr.jsonl", "manifests/embed.jsonl"]
want = set(json.load(open(a.rows)))
train = [r for r in D.load_manifest(M, splits=["train"]) if r["sample_id"] in want]
assert len(train) == len(want), (len(train), len(want))
if a.max_rows: train = train[: a.max_rows]
proc = AutoProcessor.from_pretrained(a.model)
class DS(torch.utils.data.Dataset):
    def __init__(s, recs): s.recs = recs
    def __len__(s): return len(s.recs)
    def __getitem__(s, i):
        r = s.recs[i]; return C.encode_sample(proc, r, build_segments(r["values"], r["side"]))
pad = proc.tokenizer.pad_token_id
g = torch.Generator(); g.manual_seed(a.seed)
dl = torch.utils.data.DataLoader(DS(train), batch_size=a.bs, shuffle=True, generator=g, num_workers=a.workers, collate_fn=lambda b: C.collate(b, pad), drop_last=True, persistent_workers=True, prefetch_factor=4)
model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
params = [p for p in model.parameters() if p.requires_grad]; print("trainable", sum(p.numel() for p in params), "rows", len(train), "micro-batches/epoch", len(dl), flush=True)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
total = a.steps; warm = max(1, int(0.03 * total)); ck = {int(x) for x in a.ckpt.split(",") if x}; ck.add(total)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
def to_cuda(b): return {k: v.cuda(non_blocking=True) for k, v in b.items()}
log = []; model.train(); t0 = time.time(); step = 0; micro = 0; run = 0.0; epoch = 0; done = False
while not done:
    for b in dl:
        o = model(**to_cuda(b)); (o.loss / a.accum).backward(); run += o.loss.item() / a.accum; micro += 1
        if micro % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 25 == 0:
                print(f"step {step}/{total} ep {epoch} loss {run/25:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True); log.append(dict(step=step, loss=run / 25, epoch=epoch)); run = 0.0
            if step in ck: model.save_pretrained(os.path.join(a.out, f"ckpt_{step}")); print("saved ckpt", step, flush=True)
            if step >= total: done = True; break
    epoch += 1
json.dump(dict(log=log, steps=step, seed=a.seed, model=a.model, lr=a.lr, bs=a.bs, accum=a.accum, rows=len(train), epochs=epoch, seconds=round(time.time() - t0)), open(os.path.join(a.out, "train_log.json"), "w"))
print("finished", step, round(time.time() - t0), flush=True)
