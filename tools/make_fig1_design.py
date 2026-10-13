#!/usr/bin/env python3
"""Figure 1 (study design), revision of 2026-10-13: minus sign in the token score, A_probe read from language-model hidden states, A_tf from layer-0 tokens,
vision-side LoRA (tower, merger, DeepStack merger) for VC, 'concept-bottleneck pipeline' instead of a paper label. usage: python3 tools/make_fig1_design.py [out.png]"""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': 'sans-serif', 'mathtext.default': 'regular'})
OR, BL, GR, SK, PINK = '#D55E00', '#0072B2', '#009E73', '#56B4E9', '#CC79A7'
GREY, LORA = '#E4E4E4', '#F7D9C6'
fig, ax = plt.subplots(figsize=(7.1, 4.45)); ax.set_xlim(0, 100); ax.set_ylim(0, 71); ax.axis('off')
def box(x0, y0, x1, y1, text, fc='white', ec='#333333', lw=1.0, ls='-', fs=7.0, fw='normal', tc='black'):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0.0,rounding_size=1.2', fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha='center', va='center', fontsize=fs, fontweight=fw, color=tc, zorder=4, linespacing=1.15)
def arrow(x0, y0, x1, y1, color='#333333', lw=1.1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, color=color, lw=lw, zorder=3, shrinkA=0, shrinkB=0))
Y = {1: (52, 62), 2: (36, 46), 3: (20, 30), 4: (2, 12)}
# lane 1: VLM route
box(1, Y[1][0], 15, Y[1][1], 'Mammogram\n2 views, 896 px')
box(21, Y[1][0], 38, Y[1][1], 'Vision tower\n(Qwen3-VL ViT)', fc=GREY)
box(42, Y[1][0], 54, Y[1][1], 'Merger +\nDeepStack', fc=GREY)
box(58, Y[1][0], 76, Y[1][1], 'Language model\n(8B) + LoRA', fc=LORA, ec=OR)
box(82, Y[1][0], 99, Y[1][1], 'V: token score\n\u2212log P(\u201cNo\u201d)', ec=OR, lw=1.6, fw='bold')
for xa, xb in [(15, 21), (38, 42), (54, 58), (76, 82)]: arrow(xa, 57, xb, 57)
ax.add_patch(FancyBboxPatch((20.2, 51.2), 34.6, 11.6, boxstyle='round,pad=0.0,rounding_size=1.4', fc='none', ec=OR, lw=1.1, ls=(0, (3, 2)), zorder=3))
ax.text(37.5, 66.4, 'VC: LoRA also in the vision tower,\nmerger and DeepStack merger', ha='center', va='center', fontsize=6.6, color=OR, linespacing=1.15)
ax.text(70, 66.4, 'V: 4,591 steps; V2x: 9,182 steps;\nV$_{ctl}$: 2,295-step schedule', ha='center', va='center', fontsize=6.6, color=OR, linespacing=1.15)
# lane 2: probe on language-model hidden states
box(52, Y[2][0], 76, Y[2][1], 'Linear probe on\nlanguage-model hidden states\n(VP, VCP: fine-tuned VLM;\nA$_{probe}$: base model)', ec=BL, fs=6.5)
box(82, Y[2][0], 99, Y[2][1], 'VP, VCP\nA$_{probe}$', ec=BL, lw=1.6, fw='bold')
arrow(67, 52, 67, 46, color=BL); arrow(76, 41, 82, 41, color=BL)
# lane 3: transformer head on frozen-tower tokens (base model)
box(40, Y[3][0], 76, Y[3][1], 'Transformer head on layer-0 tokens\n(tower + merger output, base model)', ec=SK, fs=6.6)
box(82, Y[3][0], 99, Y[3][1], 'A$_{tf}$', ec=GR, lw=1.6, fw='bold')
ax.plot([48, 48], [52, 30.4], color=SK, lw=1.1, zorder=1); arrow(48, 31, 48, 30, color=SK)
arrow(76, 25, 82, 25, color=GR)
# lane 4: specialist route
box(1, Y[4][0], 15, Y[4][1], 'Mammogram\n(same breast)')
box(21, Y[4][0], 39, Y[4][1], 'DINOv2 ViT-B/14\n(frozen)', fc=GREY)
box(58, Y[4][0], 76, Y[4][1], 'Stage-1 head\n(concept-bottleneck\npipeline)', ec=PINK, fs=6.8)
box(82, Y[4][0], 99, Y[4][1], 'D: specialist\nhead', ec=PINK, lw=1.6, fw='bold')
for xa, xb in [(15, 21), (39, 58), (76, 82)]: arrow(xa, 7, xb, 7, color=PINK)
for yc, lab in [(41, 'probe on language-\nmodel states'), (25, 'read-out of\nvision-tower tokens')]:
    ax.text(1, yc, lab, ha='left', va='center', fontsize=7.0, color='#444444', style='italic')
ax.text(50, 14.9, 'concept route, specialist encoder', ha='center', va='center', fontsize=6.9, color=PINK)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
import os
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures', 'fig1_design.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300)
