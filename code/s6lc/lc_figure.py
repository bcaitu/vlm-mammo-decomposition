#!/usr/bin/env python3
"""s6lc figures: (1) learning curves (test AUC vs number of positive training examples), (2) confirmatory contrasts at the top size.
Reads runs_lc/analysis/lc_curve.csv, lc_contrasts.csv and (optional) lc_auc_by_run.csv written by lc_analysis.py. Colours are bound to arms across the figure.
usage (python tool, after apply_figure_style()): fig = make_fig1(curve_csv, byrun_csv=None); fig = make_fig2(contrasts_csv)"""
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
COL = dict(V="#D55E00", VP="#0072B2", A_probe="#56B4E9", A_tf="#009E73", D="#CC79A7", V_ctl="#D55E00", D_ref="#CC79A7", V0="#6b6b6b", V_blank="#000000")
LAB = dict(V="LoRA VLM", VP="probe on VLM states", A_probe="linear probe, frozen tower", A_tf="transformer head, frozen tower", D="DINOv2 head")
FIND = [("mass", "Mass (control)"), ("calc", "Calcifications"), ("asymmetry", "Asymmetry")]
ORDER = ["f32", "f16", "f8", "f4", "f2", "f1"]
def panel_letter(ax, letter, dx=-0.18, dy=1.02, fontsize=None):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold", fontsize=fontsize or 10, va="bottom", ha="left")
def _kfmt(x, _=None): return f"{x/1000:g}k" if x >= 1000 else f"{x:g}"
def _repel(items, gap, lo=0.02, hi=0.98):
    """items: list of (key, y) in axes fraction -> dict key -> y with at least `gap` between neighbours (order kept)."""
    it = sorted(items, key=lambda t: t[1]); ys = [y for _, y in it]
    for _ in range(200):
        moved = False
        for i in range(1, len(ys)):
            d = ys[i] - ys[i - 1]
            if d < gap - 1e-9: s = (gap - d) / 2; ys[i - 1] -= s; ys[i] += s; moved = True
        ys = [min(max(y, lo), hi) for y in ys]
        if not moved: break
    return {k: y for (k, _), y in zip(it, ys)}
def make_fig1(curve_csv, byrun_csv=None, arms=("D", "A_probe", "A_tf", "VP", "V"), title=None):
    C = pd.read_csv(curve_csv); B = pd.read_csv(byrun_csv) if byrun_csv else None
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.05), sharey=True, gridspec_kw=dict(wspace=0.07, left=0.075, right=0.745, bottom=0.30, top=0.90 if title is None else 0.84))
    lo_all = np.nanmin(C["lo"]) if len(C) else 0.45; ylo = max(0.0, np.floor((min(lo_all, 0.5) - 0.02) * 20) / 20); yhi = min(1.0, np.ceil((np.nanmax(C["hi"]) + 0.01) * 20) / 20) if len(C) else 1.0
    for k, (ax, (f, ttl)) in enumerate(zip(axs, FIND)):
        ax.axhline(0.5, color="#bdbdbd", lw=0.8, ls=":", zorder=0)
        xmap = {s: v for s, v in zip(C[(C.finding == f) & (C.arm == "V")]["size"], C[(C.finding == f) & (C.arm == "V")]["n_pos_train"])}
        if B is not None and xmap:   # individual seeds of the focal arm, behind the mean line
            b = B[(B.arm == "V") & (B.finding == f)].copy(); b["x"] = b["size"].map(xmap); b = b.dropna(subset=["x"])
            ax.scatter(b["x"], b["auc"], s=7, facecolors="none", edgecolors=COL["V"], linewidths=0.6, alpha=0.55, zorder=2)
        ends = []
        for arm in arms:
            d = C[(C.arm == arm) & (C.finding == f)].copy()
            if d.empty: continue
            d["o"] = d["size"].map({s: i for i, s in enumerate(ORDER)}); d = d.sort_values("o")
            lw = 2.2 if arm == "V" else 1.5 if arm in ("VP", "D") else 1.1
            ax.fill_between(d.n_pos_train, d.lo, d.hi, color=COL[arm], alpha=0.14 if arm in ("V", "VP", "D") else 0.08, lw=0, zorder=1)
            ms_ = 3.4 if arm == "V" else 2.6; z_ = 4 if arm == "V" else 3
            if arm in ("V", "VP"):   # the top size comes from the 4,591-step v1 adapters, the other sizes from 2,295 steps: last segment dashed
                d1 = d[d["size"] != "f1"]; d2 = d[d["size"].isin(["f2", "f1"])]
                ax.plot(d1.n_pos_train, d1.auc, color=COL[arm], lw=lw, marker="o", ms=ms_, zorder=z_)
                ax.plot(d2.n_pos_train, d2.auc, color=COL[arm], lw=lw, ls=(0, (2.2, 1.6)), zorder=z_)
                ax.plot(d2.n_pos_train.iloc[-1:], d2.auc.iloc[-1:], color=COL[arm], marker="o", ms=ms_, ls="none", zorder=z_)
            else:
                ax.plot(d.n_pos_train, d.auc, color=COL[arm], lw=lw, marker="o", ms=ms_, zorder=z_)
            ends.append((arm, float(d.auc.iloc[-1]), float(d.n_pos_train.iloc[-1])))
        top = C[(C["size"] == "f1") & (C.finding == f)]
        for arm, mk, fc in (("D_ref", "D", "none"), ("V_ctl", "s", "none"), ("V0", "x", None), ("V_blank", "+", None)):
            d = top[top.arm == arm]
            if d.empty: continue
            ax.errorbar(d.n_pos_train, d.auc, yerr=[d.auc - d.lo, d.hi - d.auc], color=COL[arm], lw=0.8, capsize=0, marker=mk, ms=5.0, mec=COL[arm], mfc=fc if fc else COL[arm], mew=1.2, ls="none", zorder=5)
        ax.set_xscale("log"); ax.set_title(ttl, loc="left"); ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_kfmt)); ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        xs = C[C.finding == f].n_pos_train
        if len(xs):
            ax.set_xlim(xs.min() * 0.78, xs.max() * 1.3); ax.set_xticks([t for t in (100, 300, 1000, 3000) if xs.min() * 0.78 <= t <= xs.max() * 1.3])
        ax.set_ylim(ylo, yhi)
        if k == 0: ax.set_ylabel("AUC on EMBED test")
        if k == 2 and ends:   # identity labels once per row, at the right end of the last panel, with leaders
            yy = {a: (y - ylo) / (yhi - ylo) for a, y, _ in ends}; pos = _repel(list(yy.items()), 0.095)
            for a, y, x in ends:
                ax.annotate(LAB[a], xy=(x, y), xycoords="data", xytext=(1.06, pos[a]), textcoords="axes fraction", color=COL[a], va="center", ha="left", fontsize=6.6,
                            arrowprops=dict(arrowstyle="-", lw=0.5, color=COL[a], shrinkA=0, shrinkB=2, relpos=(0, 0.5)), annotation_clip=False, fontweight="bold" if a == "V" else "normal")
    fig.text(0.41, 0.205, "Positive training examples (known present; mean over three seeds)", ha="center", va="center")
    h = [Line2D([], [], marker="D", ls="none", mfc="none", mec=COL["D_ref"], mew=1.2, ms=5, label="DINOv2 head, original recipe (top size)"),
         Line2D([], [], marker="s", ls="none", mfc="none", mec=COL["V_ctl"], mew=1.2, ms=5, label="LoRA VLM, 2,295 steps (top size)"),
         Line2D([], [], marker="x", ls="none", mec=COL["V0"], mew=1.2, ms=5, label="base VLM, no fine-tuning"),
         Line2D([], [], marker="+", ls="none", mec=COL["V_blank"], mew=1.2, ms=6, label="LoRA VLM, black images"),
         Line2D([], [], marker="o", ls="none", mfc="none", mec=COL["V"], mew=0.8, ms=3.5, alpha=0.7, label="single seeds of LoRA VLM"),
         Patch(facecolor="#9a9a9a", alpha=0.25, label="95% interval over test patients")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.42, -0.01), handletextpad=0.3, columnspacing=1.2)
    if title: fig.text(0.075, 0.95, title, ha="left", va="center")
    return fig
def make_fig2(contrasts_csv):
    C = pd.read_csv(contrasts_csv)
    groups = [("K1", lambda c: c.str.startswith("K1"), "VLM-state probe \u2212 best frozen-tower read-out", 0.03),
              ("K2", lambda c: c.str.startswith("K2"), "VLM own score \u2212 VLM-state probe", 0.03),
              ("K3", lambda c: c.str.startswith("K3 V - D"), "VLM own score \u2212 DINOv2 head", 0.05)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw=dict(width_ratios=[3.0, 1.35], wspace=0.38, left=0.115, right=0.985, bottom=0.27, top=0.93))
    order = [("calc", "Calcifications", True), ("asymmetry", "Asymmetry", True), ("mass", "Mass (control)", False)]
    yt, yl = [], []; lo_all, hi_all = [0.0], [0.05]
    for g, (kname, sel, lab, marg) in enumerate(groups):
        y0 = g * 4.4
        ax.fill_betweenx([y0 + 0.55, y0 + 3.45], -marg, marg, color="#e3e3e3", lw=0, zorder=0)
        ax.text(0.012, y0, f"{kname}  {lab}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7, fontweight="bold")
        for r, (f, ttl, conf) in enumerate(order):
            d = C[sel(C.contrast) & (C.finding == f) & (C["size"] == "f1")]
            yy = y0 + 1 + r; yt.append(yy); yl.append(ttl)
            if d.empty: ax.text(0.0, yy, "not run", color="#8c8c8c", fontsize=6.4, va="center", ha="center"); continue
            d = d.iloc[0]; lo_all.append(float(d.lo)); hi_all.append(float(d.hi))
            ax.errorbar(float(d.delta), yy, xerr=[[float(d.delta - d.lo)], [float(d.hi - d.delta)]], fmt="o", color="#222222", mfc="#222222" if conf else "white", ms=4.8, lw=1.3, capsize=2, zorder=3)
    ax.axvline(0, color="#8c8c8c", lw=0.8, zorder=1)
    ax.set_yticks(yt); ax.set_yticklabels(yl); ax.set_ylim(len(groups) * 4.4 - 0.7, -0.8); ax.tick_params(axis="y", length=0)
    ax.set_xlim(min(lo_all) - 0.02, max(hi_all) + 0.012); ax.set_xlabel("Difference in test AUC at the top size (95% interval)")
    d4 = C[C.contrast.str.startswith("K4")]; lo4, hi4 = [0.0, 0.03], []
    for r, (f, ttl, conf) in enumerate(order):
        d = d4[d4.finding == f]
        if d.empty: ax2.text(0.0, r, "not run", color="#8c8c8c", fontsize=6.4, va="center", ha="center"); continue
        d = d.iloc[0]; lo4.append(float(d.lo)); hi4.append(float(d.hi))
        ax2.errorbar(float(d.delta), r, xerr=[[float(d.delta - d.lo)], [float(d.hi - d.delta)]], fmt="o", color=COL["V"], mfc=COL["V"] if conf else "white", ms=4.8, lw=1.3, capsize=2, zorder=3)
    ax2.axvline(0.02, color="#555555", lw=0.9, ls="--", zorder=1); ax2.axvline(0, color="#8c8c8c", lw=0.8, zorder=1)
    ax2.set_xlim(min(lo4) - 0.01, max(hi4 + [0.03]) + 0.01)
    ax2.set_yticks(range(3)); ax2.set_yticklabels([o[1] for o in order]); ax2.set_ylim(2.6, -0.6); ax2.tick_params(axis="y", length=0)
    ax2.set_xlabel("AUC gain per doubling"); ax2.set_title("K4  LoRA VLM gain", loc="left", fontsize=7, fontweight="bold")
    h = [Line2D([], [], marker="o", ls="none", mfc="#222222", mec="#222222", ms=4.8, label="confirmatory"), Line2D([], [], marker="o", ls="none", mfc="white", mec="#222222", ms=4.8, label="descriptive (control)"),
         Patch(facecolor="#e3e3e3", label="equivalence margin"), Line2D([], [], color="#555555", ls="--", lw=0.9, label="rise threshold, 0.02 per doubling")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.52, -0.01), handletextpad=0.4, columnspacing=1.2)
    for a, L in ((ax, "a"), (ax2, "b")): panel_letter(a, L, dx=-0.16 if L == "a" else -0.30)
    return fig
