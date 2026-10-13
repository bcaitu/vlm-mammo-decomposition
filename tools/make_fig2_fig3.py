#!/usr/bin/env python3
"""Figures 2 and 3 from results/s6lc/lc_curve.csv, lc_auc_by_run.csv and lc_contrasts.csv with the revised labels (tools/lc_figure_rev.py). Run from the repository root: python3 tools/make_fig2_fig3.py"""
import os, warnings; warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mpl.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "figure.dpi": 150})
exec(open(os.path.join(ROOT, "tools", "lc_figure_rev.py")).read())
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
S = os.path.join(ROOT, "results", "s6lc")
make_fig1(os.path.join(S, "lc_curve.csv"), os.path.join(S, "lc_auc_by_run.csv")).savefig(os.path.join(ROOT, "figures", "fig2_learning_curves.png"), dpi=300)
make_fig2(os.path.join(S, "lc_contrasts.csv")).savefig(os.path.join(ROOT, "figures", "fig3_contrasts.png"), dpi=300)
print("figures 2 and 3 written")
