"""Figure 4 (s10db: the DINOv2 head at the update budget of the VLM) and Figure S4 (examination-type strata).
Run from the package root: python3 build/make_fig4_budget.py"""
import warnings; warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "figure.dpi": 150})
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.lines import Line2D
D = 'results/s10/'
COL = dict(V="#D55E00", D="#CC79A7", D_bV="#0072B2", V2x="#D55E00", D_b2x="#0072B2")
SIZES = ['f32', 'f16', 'f8', 'f4', 'f2', 'f1']; X = np.arange(6); XL = ['2.3k', '4.6k', '9.2k', '18k', '37k', '73k']
F = [('calc', 'Calcifications'), ('asymmetry', 'Asymmetry'), ('mass', 'Mass (control)')]
cu = pd.read_csv(D + 's10db_tables/s10db_curve.csv'); c = pd.read_csv(D + 's10db_tables/s10db_contrasts.csv')

fig = plt.figure(figsize=(7.1, 4.6))
gs = fig.add_gridspec(1, 3, wspace=0.32, left=0.07, right=0.99, top=0.86, bottom=0.50)
gsb = fig.add_gridspec(1, 3, wspace=0.22, left=0.33, right=0.99, top=0.33, bottom=0.10)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
for ax, (f, title) in zip(axes, F):
    for arm, lab, ls in [('D', 'DINOv2 head, original recipe (D)', '--'), ('D_bV', 'DINOv2 head at the updates of V (D$_{bV}$)', '-'), ('V', 'LoRA VLM token score (V)', '-')]:
        t = cu[(cu.arm == arm) & (cu.finding == f)].set_index('size').loc[SIZES]
        ax.fill_between(X, t.lo, t.hi, color=COL[arm], alpha=0.10, lw=0)
        ax.plot(X, t.auc, color=COL[arm], ls=ls, lw=1.3, marker='o', ms=3.4, label=lab, zorder=3)
    for arm, dx in [('V2x', 0.22), ('D_b2x', 0.40)]:
        t = cu[(cu.arm == arm) & (cu.finding == f) & (cu['size'] == 'f1')].iloc[0]
        ax.plot([5 + dx] * 2, [t.lo, t.hi], color=COL[arm], lw=1.0)
        ax.plot(5 + dx, t.auc, marker='o', ms=3.6, mfc='white', mec=COL[arm], mew=1.2, ls='none', zorder=4)
    ax.set_xticks(X); ax.set_xticklabels(XL); ax.set_xlim(-0.35, 5.65)
    ax.set_title(title, loc='left', pad=3)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
axes[0].set_ylabel('AUC, EMBED test set (seed mean)')
axes[1].set_xlabel('Training breasts (nested subsets of the pool)')
h = [Line2D([], [], color=COL['V'], marker='o', ms=3.4, lw=1.3, label='LoRA VLM token score (V)'),
     Line2D([], [], color=COL['D_bV'], marker='o', ms=3.4, lw=1.3, label='DINOv2 head at the updates of V (D$_{\\mathrm{bV}}$)'),
     Line2D([], [], color=COL['D'], marker='o', ms=3.4, lw=1.3, ls='--', label='DINOv2 head, original recipe (D)'),
     Line2D([], [], color='#444444', marker='o', ms=3.6, mfc='white', mew=1.2, ls='none', label='open markers: whole pool, twice the updates (V2x, D$_{\\mathrm{b2x}}$)')]
fig.legend(handles=h, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.53, 0.995), handlelength=2.2, columnspacing=1.6)
for ax, L in zip(axes, 'abc'):
    ax.text(-0.08, 1.06, L, transform=ax.transAxes, fontsize=10, fontweight='bold', ha='right', va='bottom')
# bottom: contrasts forest
rows = [('desc K3 V - D (original, whole pool)', 'K3: V $-$ D (original recipes)', COL['V'], False),
        ('C1 V - D_bV (gap at matched updates, whole pool)', 'C1: V $-$ D$_{\\mathrm{bV}}$ (equal updates)', COL['V'], True),
        ('C3 V2x - D_b2x (gap at the updates of V2x, whole pool)', 'C3: V2x $-$ D$_{\\mathrm{b2x}}$ (twice the updates)', COL['V'], True),
        ('C2 D_bV - D (cost of the update budget of V to D, whole pool)', 'C2: D$_{\\mathrm{bV}}$ $-$ D (cost to D of fewer updates)', COL['D_bV'], True),
        ("desc D_b2x - D_bV (value of doubling D's updates)", 'D$_{\\mathrm{b2x}}$ $-$ D$_{\\mathrm{bV}}$ (doubling the updates of D)', COL['D_bV'], False)]
axb = [fig.add_subplot(gsb[0, i]) for i in range(3)]
for ax, (f, title) in zip(axb, F):
    ax.axvspan(-0.03, 0.03, color='#e6e6e6', lw=0, zorder=0)
    ax.axvline(0, color='#555555', lw=0.9, zorder=1)
    for i, (k, lab, col, conf) in enumerate(rows):
        r = c[(c.contrast == k) & (c.finding == f)].iloc[0]; y = -i
        ax.plot([r.lo, r.hi], [y, y], color=col, lw=1.5, solid_capstyle='butt', zorder=2)
        ax.plot(r.delta, y, marker='o', ms=5.0, color=col, mfc=(col if (conf and f != 'mass') else 'white'), mec=col, mew=1.2, ls='none', zorder=3)
    ax.set_ylim(-len(rows) + 0.4, 0.6); ax.set_xlim(-0.085, 0.045)
    ax.set_xticks([-0.08, -0.04, 0.0, 0.04]); ax.set_xticklabels(['\u22120.08', '\u22120.04', '0', '+0.04'])
    ax.set_yticks([-i for i in range(len(rows))]); ax.set_yticklabels([lab for _, lab, _, _ in rows] if ax is axb[0] else [])
    ax.tick_params(axis='y', length=0)
    for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
axb[1].set_xlabel('Difference in test AUC at the whole pool (paired 95% interval over test patients; grey: $\\pm$0.03)')
for ax, L in zip(axb, 'def'):
    ax.text(-0.08 if ax is not axb[0] else -1.25, 1.06, L, transform=ax.transAxes, fontsize=10, fontweight='bold', ha='right', va='bottom')
for ax, (f, title) in zip(axb, F):
    ax.set_title(title, loc='left', pad=3)
fig.savefig('figures/fig4_budget.png', dpi=300)

# ---------- Figure S4: strata ----------
p2 = pd.read_csv(D + 's10an_posthoc_tables/ph2_auc_by_stratum.csv'); da = pd.read_csv(D + 's10db_posthoc_tables/dbph_auc_by_stratum.csv')
p3 = pd.read_csv(D + 's10an_posthoc_tables/ph3_contrasts_by_stratum.csv'); dc = pd.read_csv(D + 's10db_posthoc_tables/dbph_contrasts_by_stratum.csv'); p1 = pd.read_csv(D + 's10an_posthoc_tables/ph1_examtype_indicator.csv')
ST = [('all rows', 'All'), ('screening only', 'Screening'), ('diagnostic only', 'Diagnostic')]
ARMS = [('V', p2, 'V (token score)', COL['V'], 'o'), ('VP', p2, 'VP (probe)', '#0072B2', 's'), ('D_bV', da, 'D$_{\\mathrm{bV}}$', '#56B4E9', 'D'), ('D', p2, 'D', COL['D'], 'o')]
fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.4), gridspec_kw=dict(hspace=0.55, wspace=0.3, left=0.08, right=0.99, top=0.88, bottom=0.09))
for ax, (f, title) in zip(axes[0], F):
    for j, (arm, src, lab, col, mk) in enumerate(ARMS):
        for i, (s, sl) in enumerate(ST):
            r = src[(src.arm == arm) & (src.stratum == s) & (src.finding == f)].iloc[0]
            x = i + (j - 1.5) * 0.17
            ax.plot([x, x], [r.lo, r.hi], color=col, lw=1.0); ax.plot(x, r.auc, marker=mk, ms=3.8, color=col, ls='none', label=lab if (i == 0) else None, zorder=3)
    r = p1[p1.finding == f].iloc[0]
    ax.plot([-0.4, 0.4], [r.auc_exam_type_indicator] * 2, color='#444444', lw=0.9, ls=':', label='exam type as score' )
    ax.axhline(0.5, color='#bbbbbb', lw=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels([sl for _, sl in ST]); ax.set_title(title, loc='left', pad=3); ax.set_xlim(-0.6, 2.6)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
axes[0][0].set_ylabel('AUC (seed mean)')
axes[0][0].legend(loc='upper center', bbox_to_anchor=(1.7, 1.42), ncol=5, frameon=False, handlelength=1.6, columnspacing=1.2)
CS = [('K3 V - D', p3, 'K3: V $-$ D', COL['V']), ('C1 V - D_bV', dc, 'C1: V $-$ D$_{\\mathrm{bV}}$', '#CC79A7'), ('C2 D_bV - D', dc, 'C2: D$_{\\mathrm{bV}}$ $-$ D', '#0072B2'),
      ('K2 V - VP', p3, 'K2: V $-$ VP', '#009E73'), ('S3 V2x - V', p3, 'S3: V2x $-$ V', '#E69F00')]
for ax, (f, title) in zip(axes[1], F):
    ax.axhspan(-0.03, 0.03, color='#e6e6e6', lw=0, zorder=0); ax.axhline(0, color='#555555', lw=0.8)
    for j, (k, src, lab, col) in enumerate(CS):
        for i, (s, sl) in enumerate(ST):
            r = src[(src.contrast == k) & (src.stratum == s) & (src.finding == f)].iloc[0]
            x = i + (j - 2) * 0.15
            ax.plot([x, x], [r.lo, r.hi], color=col, lw=1.0); ax.plot(x, r.delta, marker='o', ms=3.6, color=col, ls='none', label=lab if i == 0 else None, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels([sl for _, sl in ST]); ax.set_xlim(-0.6, 2.6); ax.set_ylim(-0.26, 0.08)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
axes[1][0].set_ylabel('Difference in AUC (paired 95% interval)')
axes[1][0].legend(loc='upper center', bbox_to_anchor=(1.7, 1.33), ncol=5, frameon=False, handlelength=1.6, columnspacing=1.2)
for ax, L in zip(list(axes[0]) + list(axes[1]), 'abcdef'):
    ax.text(-0.1, 1.05, L, transform=ax.transAxes, fontsize=10, fontweight='bold', ha='right', va='bottom')
fig.savefig('figures/figS4_strata.png', dpi=300)
print('ok')
