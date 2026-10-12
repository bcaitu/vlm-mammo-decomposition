"""Shared pieces of the VLM study (paper 4): cache paths, prompt, sample building, collate."""
import hashlib, json, os, random
import numpy as np, torch
from PIL import Image
CACHE = "/raid/home/user0/cbmammo_cache/vlm896"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SYS = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
def fname(sid, v): return hashlib.sha1(f"{sid}|{v}".encode()).hexdigest()[:20] + ".png"
def load_img(sid, v):
    return Image.open(os.path.join(CACHE, fname(sid, v))).convert("RGB")
def views_of(rec): return [v for v in ("CC", "MLO") if rec["views"].get(v)]
def prompt_text(rec, task="train"):
    side = {"L": "left", "R": "right"}.get(rec.get("side") or "", "")
    p = SYS + "<|im_start|>user\n"
    for v in views_of(rec): p += f"{v} view: <|vision_start|><|image_pad|><|vision_end|>\n"
    if task == "zeroshot":
        p += ("Write a mammography report for this " + (side + " " if side else "") + "breast. Describe the breast density (ACR A-D), any mass (shape and margins), calcifications (morphology and distribution), "
              "asymmetry and architectural distortion, and finish with the BI-RADS category (1-5).")
    else:
        p += "Write the mammography report for the " + (side + " " if side else "") + "breast."
    return p + "<|im_end|>\n<|im_start|>assistant\n"
def blank_img(like):
    return Image.new("RGB", like.size, (0, 0, 0))
def encode_sample(proc, rec, segs=None, imgs=None, task="train"):
    """-> dict of tensors; segs = [(text, supervised)] appended as the response."""
    imgs = imgs if imgs is not None else [load_img(rec["sample_id"], v) for v in views_of(rec)]
    b = proc(text=[prompt_text(rec, task)], images=imgs, return_tensors="pt")
    out = {k: v for k, v in b.items()}
    if segs is None: return out
    tok = proc.tokenizer; ids, lab = [], []
    for t, s in segs:
        x = tok(t, add_special_tokens=False)["input_ids"]; ids += x; lab += (x if s else [-100] * len(x))
    end = tok("<|im_end|>", add_special_tokens=False)["input_ids"]; ids += end; lab += end
    n0 = out["input_ids"].shape[1]
    out["input_ids"] = torch.cat([out["input_ids"], torch.tensor([ids])], 1)
    out["attention_mask"] = torch.ones_like(out["input_ids"])
    out["labels"] = torch.cat([torch.full((1, n0), -100), torch.tensor([lab])], 1)
    if "mm_token_type_ids" in out: out["mm_token_type_ids"] = torch.cat([out["mm_token_type_ids"], torch.zeros((1, len(ids)), dtype=out["mm_token_type_ids"].dtype)], 1)
    return out
def collate(items, pad_id, left=False):
    L = max(i["input_ids"].shape[1] for i in items); out = {}
    for k in items[0]:
        if k in ("pixel_values", "image_grid_thw"): out[k] = torch.cat([i[k] for i in items], 0)
        elif items[0][k].dim() == 2 and items[0][k].shape[1] == items[0]["input_ids"].shape[1]:
            fill = pad_id if k == "input_ids" else (-100 if k == "labels" else 0); rows = []
            for i in items:
                pad = torch.full((1, L - i[k].shape[1]), fill, dtype=i[k].dtype)
                rows.append(torch.cat([pad, i[k]], 1) if left else torch.cat([i[k], pad], 1))
            out[k] = torch.cat(rows, 0)
        else: out[k] = torch.cat([i[k] for i in items], 0)
    return out
