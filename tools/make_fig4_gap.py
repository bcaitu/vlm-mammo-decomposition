"""Figure 4 (distance to D; revision of 2026-10-13: labels). Run from the repository root: python3 tools/make_fig4_gap.py"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

OR, BL, GR, SK = '#D55E00', '#0072B2', '#009E73', '#56B4E9'

G = pd.read_csv(os.path.join(ROOT, "results", "s8", "s8_posthoc2_gap_to_D.csv"))

rows_def = [
    ('A_probe', 'Linear probe', SK, 'o', 'Base model (no fine-tuning)'),
    ('A_tf', 'Transformer head', GR, 'o', 'Base model (no fine-tuning)'),
    ('VP', 'LoRA language side', BL, 'o', 'Probe on VLM hidden states'),
    ('VCP', 'LoRA language + vision side', BL, 'D', 'Probe on VLM hidden states'),
    ('V_ctl', '2,295-step schedule', OR, 'o', 'VLM token score'),
    ('V', '4,591 steps (reference)', OR, 'o', 'VLM token score'),
    ('V2x', '9,182 steps', OR, 'o', 'VLM token score'),
    ('VC', '4,591 steps + vision-side LoRA', OR, 'D', 'VLM token score'),
]
panels = [('calc', 'Calcifications', True), ('asymmetry', 'Asymmetry', True), ('mass', 'Mass (control)', False)]

ypos, y = {}, 0.0
prev = None
group_y = {}
for arm, lab, col, mk, grp in rows_def:
    if prev is not None and grp != prev:
        y -= 0.7
    ypos[arm] = y
    group_y.setdefault(grp, []).append(y)
    y -= 1.0
    prev = grp

fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.35), sharey=True)
for ax, (fnd, title, conf) in zip(axes, panels):
    for arm, lab, col, mk, grp in rows_def:
        r = G[(G.arm == arm) & (G.finding == fnd)].iloc[0]
        yy = ypos[arm]
        ax.plot([r.lo, r.hi], [yy, yy], color=col, lw=1.5, solid_capstyle='butt', zorder=2)
        ax.plot(r.delta_vs_D, yy, marker=mk, ms=5.6, color=col, mfc=(col if conf else 'white'), mec=col, mew=1.3, zorder=3, ls='none')
    ax.axvline(0, color='#555555', lw=0.9, zorder=1)
    ax.set_xlim(-0.115, 0.045)
    ax.set_xticks([-0.10, -0.05, 0.0])
    ax.set_xticklabels(['\u22120.10', '\u22120.05', '0'])
    ax.set_title(title, loc='left', fontsize=8.5, pad=4)
    ax.tick_params(axis='y', length=0)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
axes[0].set_yticks([ypos[a] for a, *_ in rows_def])
axes[0].set_yticklabels([l for _, l, *_ in rows_def], fontsize=7.4)
axes[0].set_ylim(y + 0.4, 1.35)
for grp, ys in group_y.items():
    axes[0].text(-0.60, max(ys) + 0.62, grp, transform=axes[0].get_yaxis_transform(), ha='left', va='center', fontsize=7.6, fontweight='bold', clip_on=False)
for ax, L in zip(axes, 'abc'):
    ax.text(-0.02, 1.075, L, transform=ax.transAxes, fontsize=10, fontweight='bold', ha='right', va='bottom')
fig.supxlabel('Arm minus D, difference in test AUC (0 = parity); paired 95% interval over test patients', fontsize=8, y=0.095)
h = [Line2D([], [], marker='o', ls='none', ms=5.2, color='#444444', label='primary finding'),
     Line2D([], [], marker='o', ls='none', ms=5.2, color='#444444', mfc='white', mew=1.2, label='control, descriptive'),
     Line2D([], [], marker='D', ls='none', ms=5.0, color='#444444', label='vision-side LoRA')]
fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=7.4, handletextpad=0.3, columnspacing=1.6, bbox_to_anchor=(0.63, -0.005))
fig.subplots_adjust(left=0.30, right=0.985, top=0.87, bottom=0.235, wspace=0.10)
os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)
fig.savefig(os.path.join(ROOT, 'figures', 'fig4_gap_to_D.png'), dpi=300)