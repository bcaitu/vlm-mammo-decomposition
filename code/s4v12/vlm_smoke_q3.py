#!/usr/bin/env python3
"""Throughput smoke test of LoRA fine-tuning of a Qwen-VL model on two mammogram views (measures tokens per sample and seconds per step; no IDs printed)."""
import json, sys, time, os, numpy as np, torch
from PIL import Image
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model
name = sys.argv[1]; bs = int(sys.argv[2]); side = int(sys.argv[3])      # e.g. Qwen/Qwen2-VL-2B-Instruct 4 896
rows = []
for l in open("manifests/embed.jsonl"):
    d = json.loads(l)
    if d["split"] == "train" and d["views"].get("CC") and d["views"].get("MLO"): rows.append(d)
    if len(rows) >= 64: break
from cbmammo.data import read_image
def load(p):
    a = np.asarray(read_image(p)).astype(np.float32)
    if a.ndim == 3: a = a[..., 0]
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5); a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")
    w, h = im.size; s = side / max(w, h); w2, h2 = max(32, int(w * s) // 32 * 32), max(32, int(h * s) // 32 * 32)
    return im.resize((w2, h2))
proc = AutoProcessor.from_pretrained(name)
from transformers import AutoModelForImageTextToText as M
model = M.from_pretrained(name, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.0, task_type="CAUSAL_LM"))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
txt = "Breast density: C. Mass: present, oval shape, circumscribed margin. Calcifications: absent. Asymmetry: absent. Architectural distortion: absent. BI-RADS category: 4."
def batch(rs):
    pre = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>Write the mammography report.<|im_end|>\n<|im_start|>assistant\n"
    prompts = [pre + txt + "<|im_end|>" for _ in rs]
    imgs = [[load(r["views"]["CC"]), load(r["views"]["MLO"])] for r in rs]
    flat = [i for pair in imgs for i in pair]
    b = proc(text=prompts, images=flat, return_tensors="pt", padding=True)
    b["labels"] = b["input_ids"].clone(); b["labels"][b["attention_mask"] == 0] = -100
    return {k: v.cuda() for k, v in b.items()}
model.train(); times = []
sys.stderr = sys.stdout
for step in range(6):
    t0 = time.time(); b = batch(rows[step * bs:(step + 1) * bs]); t1 = time.time()
    out = model(**b); out.loss.backward(); opt.step(); opt.zero_grad(); torch.cuda.synchronize(); t2 = time.time()
    times.append((t1 - t0, t2 - t1)); ntok = int(b["attention_mask"].sum())
    print(f"step {step} load+proc {t1-t0:.2f}s fwd/bwd {t2-t1:.2f}s tokens/batch {ntok} tokens/sample {ntok/bs:.0f} loss {out.loss.item():.3f} mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
print("median fwd/bwd s", float(np.median([t[1] for t in times[1:]])), "median load s", float(np.median([t[0] for t in times[1:]])))
