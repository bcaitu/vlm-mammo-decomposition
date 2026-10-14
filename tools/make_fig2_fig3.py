#!/usr/bin/env python3
"""Figures 2 and 3 from data/lc_curve.csv, data/lc_auc_by_run.csv and data/lc_contrasts.csv with the revised labels. Run from the package root: python3 build/make_fig2_fig3.py"""
import warnings; warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
mpl.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "figure.dpi": 150})
exec(open("build/lc_figure_rev.py").read())
make_fig1("data/lc_curve.csv", "data/lc_auc_by_run.csv").savefig("figures/fig2_learning_curves.png", dpi=300)
make_fig2("data/lc_contrasts.csv").savefig("figures/fig4_contrasts.png", dpi=300)
print("figures 2 and 3 written")
