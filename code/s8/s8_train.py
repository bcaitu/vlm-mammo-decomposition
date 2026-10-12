#!/usr/bin/env python3
"""s8 (paper 4): LoRA fine-tuning of Qwen3-VL-8B-Instruct with a FIXED number of optimiser steps; same recipe as lc_train.py (s6lc, v1) plus two options:
--tower_lora adds LoRA (same r/alpha) to the linear layers of the vision tower (blocks.*.attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2) and of the mergers (merger.*, deepstack_merger_list.*), --lr_vis sets their learning rate (default: same as --lr).
Without --tower_lora the script is equivalent to lc_train.py (used for the 2x step-budget arm).
Recipe identical to vlm_train.py (v1): LoRA r=16 alpha=32 on the linear layers of the language model, vision tower frozen, AdamW lr 1e-4 (no weight decay), micro-batch 4 x accumulation 4,
3% linear warm-up + cosine decay over the step budget, gradient checkpointing, bf16. The subset is iterated repeatedly (new shuffle each epoch) until the step budget is reached.
Adapters are saved at the steps listed in --ckpt (ckpt_<step>/) so that the checkpoint can be chosen on the validation sample.
usage: PYTHONPATH=$PWD:$PWD/analysis_embed python3 analysis_embed/s8_train.py --rows runs_lc/sets/subset_s0_f1.json --out runs_s8/VC/f1_s0 --seed 0 --steps 4591 --tower_lora --lr_vis 5e-5"""
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
ap.add_argument("--tower_lora", action="store_true"); ap.add_argument("--lr_vis", type=float, default=0.0, help="learning rate of the vision-tower LoRA parameters (0 = same as --lr)")
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
LANG = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
VIS = r".*visual.*\.(qkv|proj|linear_fc1|linear_fc2)"
targets = f"({LANG})|({VIS})" if a.tower_lora else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM"))
named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]; params = [p for _, p in named]
vis_p = [p for n, p in named if ".visual." in n]; lang_p = [p for n, p in named if ".visual." not in n]
print("trainable", sum(p.numel() for p in params), "of which vision", sum(p.numel() for p in vis_p), "rows", len(train), "micro-batches/epoch", len(dl), flush=True)
assert (len(vis_p) > 0) == a.tower_lora, ("vision LoRA parameters", len(vis_p), a.tower_lora)
groups = [dict(params=lang_p, lr=a.lr)] + ([dict(params=vis_p, lr=a.lr_vis or a.lr)] if vis_p else [])
opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.0)
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
json.dump(dict(log=log, steps=step, seed=a.seed, model=a.model, lr=a.lr, lr_vis=(a.lr_vis or a.lr) if a.tower_lora else None, tower_lora=a.tower_lora, n_trainable=sum(p.numel() for p in params), n_vision=sum(p.numel() for p in vis_p), bs=a.bs, accum=a.accum, rows=len(train), epochs=epoch, seconds=round(time.time() - t0)), open(os.path.join(a.out, "train_log.json"), "w"))
print("finished", step, round(time.time() - t0), flush=True)
