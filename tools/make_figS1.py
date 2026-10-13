#!/usr/bin/env python3
"""Derived from s8/code/s8_figure.py (hash-frozen original left unchanged); only the two read-out labels differ: A_probe is a linear probe on hidden states of the base model, A_tf a transformer head on its vision-tower tokens.
s8 figures: (1) AUC at the top size by arm and finding; (2) confirmatory contrasts S1-S4 (+ S2 gap).
Reads s8_auc.csv, s8_contrasts.csv and (optional) s8_auc_by_run.csv written by s8_analysis.py.
usage (python tool, after apply_figure_style()): fig = make_fig1(auc_csv, byrun_csv); fig = make_fig2(contrasts_csv)"""
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
# top to bottom; colours are threaded with the s6lc figures (V vermillion, VP blue, A_probe light blue, A_tf green, D pink); the new focal arm VC is black
ARMS = [("D", "DINOv2 head", "#CC79A7", True), ("A_tf", "base model: transformer head, tower tokens", "#009E73", True), ("A_probe", "base model: linear probe, hidden states", "#56B4E9", False),
        ("VP", "probe on LoRA-VLM states", "#0072B2", True), ("VCP", "probe on vision-side-LoRA states", "#1b2a6b", True),
        ("V", "LoRA VLM (language side)", "#D55E00", True), ("V2x", "LoRA VLM (language side), 2x steps", "#E69F00", True), ("VC", "LoRA VLM (language + vision side)", "#111111", True)]
FIND = [("mass", "Mass (control)"), ("calc", "Calcifications"), ("asymmetry", "Asymmetry")]
def panel_letter(ax, letter, dx=-0.18, dy=1.02, fontsize=None):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold", fontsize=fontsize or 10, va="bottom", ha="left")
def make_fig1(auc_csv, byrun_csv=None, span=0.24):
    C = pd.read_csv(auc_csv); B = pd.read_csv(byrun_csv) if byrun_csv else None; n = len(ARMS)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True, gridspec_kw=dict(wspace=0.07, left=0.385, right=0.985, bottom=0.2, top=0.9))
    for k, (ax, (f, ttl)) in enumerate(zip(axs, FIND)):
        d = C[C.finding == f].set_index("arm"); present = [a for a in ARMS if a[0] in d.index]
        for bnd in (4.5, 2.5): ax.axhline(bnd, color="#d0d0d0", lw=0.7, zorder=0)
        for i, (arm, lab, col, seeds) in enumerate(ARMS):
            if arm not in d.index: continue
            y = n - 1 - i; r = d.loc[arm]
            ax.hlines(y, r.lo, r.hi, color=col, lw=1.6, zorder=3)
            if B is not None and seeds:
                b = B[(B.arm == arm) & (B.finding == f)].sort_values("seed")
                ax.scatter(b.auc, y + np.linspace(-0.2, 0.2, len(b)) if len(b) > 1 else [y], s=7, facecolors="none", edgecolors=col, linewidths=0.6, alpha=0.8, zorder=4)
            ax.plot(r.auc, y, "o", color=col, ms=7.0 if arm == "VC" else 5.2, zorder=5)
        lo, hi = d.lo.min(), d.hi.max(); c = float(d.auc.mean()); x0, x1 = c - span / 2, c + span / 2
        if lo < x0 + 0.01: x0 = lo - 0.01
        if hi > x1 - 0.01: x1 = hi + 0.01
        ax.set_xlim(x0, x1); ax.set_ylim(-0.7, n - 0.3); ax.set_title(ttl, loc="left")
        ax.set_yticks(range(n)); ax.set_yticklabels([a[1] for a in ARMS][::-1]) if k == 0 else ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    fig.text(0.67, 0.065, "AUC on EMBED test (bars: 95% interval; small circles: single seeds)", ha="center", va="center")
    return fig
def make_fig2(contrasts_csv):
    C = pd.read_csv(contrasts_csv)
    groups = [("S1", "S1  Gain from tuning the vision side: vision-side-LoRA VLM minus language-side VLM", True), ("S3", "S3  Gain from doubling the step budget: 2x-step VLM minus VLM", True),
              ("S4", "S4  Linearly accessible information: probe on vision-side-LoRA states minus probe on VLM states", True), ("S2", "S2  Gap to the DINOv2 head: vision-side-LoRA VLM minus DINOv2 head", False)]
    fig, ax = plt.subplots(figsize=(7.2, 4.1), gridspec_kw=dict(left=0.135, right=0.985, bottom=0.2, top=0.97))
    order = [("calc", "Calcifications", True), ("asymmetry", "Asymmetry", True), ("mass", "Mass (control)", False)]
    yt, yl = [], []; lo_all, hi_all = [0.0], [0.06]
    for g, (key, lab, gain) in enumerate(groups):
        y0 = g * 4.4
        ax.fill_betweenx([y0 + 0.55, y0 + 3.45], -0.03, 0.03, color="#e3e3e3", lw=0, zorder=0)
        if gain: ax.plot([0.05, 0.05], [y0 + 0.55, y0 + 3.45], color="#555555", lw=0.9, ls=(0, (1.5, 1.5)), zorder=1)
        ax.text(0.004, y0, lab, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7, fontweight="bold")
        for r, (f, ttl, conf) in enumerate(order):
            d = C[C.contrast.str.startswith(key) & (C.finding == f)]; yy = y0 + 1 + r; yt.append(yy); yl.append(ttl)
            if d.empty: ax.text(0.0, yy, "not run", color="#8c8c8c", fontsize=6.4, va="center", ha="center"); continue
            d = d.iloc[0]; lo_all.append(float(d.lo)); hi_all.append(float(d.hi))
            ax.errorbar(float(d.delta), yy, xerr=[[float(d.delta - d.lo)], [float(d.hi - d.delta)]], fmt="o", color="#222222", mfc="#222222" if conf else "white", ms=4.8, lw=1.3, capsize=2, zorder=3)
    ax.axvline(0, color="#8c8c8c", lw=0.8, zorder=1)
    ax.set_yticks(yt); ax.set_yticklabels(yl); ax.set_ylim(len(groups) * 4.4 - 0.7, -0.8); ax.tick_params(axis="y", length=0)
    ax.set_xlim(min(lo_all) - 0.02, max(hi_all) + 0.012); ax.set_xlabel("Difference in test AUC (95% interval over test patients)")
    h = [Line2D([], [], marker="o", ls="none", mfc="#222222", mec="#222222", ms=4.8, label="confirmatory"), Line2D([], [], marker="o", ls="none", mfc="white", mec="#222222", ms=4.8, label="descriptive (control)"),
         Patch(facecolor="#e3e3e3", label="within \u00b10.03"), Line2D([], [], color="#555555", ls=(0, (1.5, 1.5)), lw=0.9, label="0.05, practically significant (point estimate)")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.55, -0.01), handletextpad=0.4, columnspacing=1.2)
    return fig


if __name__ == "__main__":
    import os, sys
    mpl.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "figure.dpi": 150})  # sizes of the figure-style used for the other figures
    A = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "s8")
    os.makedirs(os.path.join(os.path.dirname(A), "..", "figures"), exist_ok=True)
    fig = make_fig1(os.path.join(A, "s8_auc.csv"), os.path.join(A, "s8_auc_by_run.csv"))
    for ax in fig.axes: ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.savefig(os.path.join(os.path.dirname(A), "..", "figures", "figS1_s8_auc.png"), dpi=300, bbox_inches="tight")
    fig2 = make_fig2(os.path.join(A, "s8_contrasts.csv"))
    fig2.savefig(os.path.join(os.path.dirname(A), "..", "figures", "figS2_s8_contrasts.png"), dpi=300, bbox_inches="tight")
