import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import zipfile
import io
import os

# Apply figure style (global state)
def apply_figure_style():
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'font.family': 'sans-serif',
    })

try:
    apply_figure_style()  # provided by the figure-style skill of the Claude Science kernel; plain matplotlib defaults otherwise
except NameError:
    pass

# Load s4res data
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
resdf = pd.read_csv(os.path.join(ROOT, 'results', 's4res', 'res_main.csv'))

# Load s7zoom data
zz = json.load(open(os.path.join(ROOT, 'results', 's7zoom', 's7zoom_probe_results.json')))

# Build the figure
style = {'calc': dict(c='#222222', mk='o', ls='-', mfc='#222222', lab='Calcifications'),
         'asymmetry': dict(c='#222222', mk='s', ls='--', mfc='#222222', lab='Asymmetry'),
         'mass': dict(c='#8A8A8A', mk='o', ls=':', mfc='white', lab='Mass (control)')}
off = {'calc': -0.09, 'asymmetry': 0.0, 'mass': 0.09}
xs = [0, 1, 2]
labs = ['896', '1344', '1792']

lev = zz['levels']
ci = zz['ci']
d = zz['diff']['mean_ds1-mean_ds4']
keys = ['ds1', 'ds2', 'ds4']
pitch = ['85\n(native)', '170\n(1,800-px view)', '340\n(900-px view)']
xb = [0, 1, 2]

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.1, 3.1))

for slot in ['mass', 'asymmetry', 'calc']:
    r = resdf[resdf.slot == slot].iloc[0]
    st = style[slot]
    ys = [r.auc_896, r.auc_1344, r.auc_1792]
    lo = [r.auc_896_lo, r.auc_1344_lo, r.auc_1792_lo]
    hi = [r.auc_896_hi, r.auc_1344_hi, r.auc_1792_hi]
    xx = [x + off[slot] for x in xs]
    axa.errorbar(xx, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)],
                 color=st['c'], marker=st['mk'], ls=st['ls'], lw=1.2, ms=4.8,
                 mfc=st['mfc'], mec=st['c'], capsize=2, elinewidth=0.9,
                 label=st['lab'], zorder=3)

axa.set_xticks(xs)
axa.set_xticklabels(labs)
axa.set_xlim(-0.4, 2.4)
axa.set_ylim(0.45, 0.86)
axa.set_xlabel('Input size of the whole mammogram pair (px)', fontsize=8)
axa.set_ylabel('AUC, linear probe on base-model hidden states', fontsize=8)
axa.axhline(0.5, color='#BBBBBB', lw=0.8, ls=':', zorder=1)
hh, ll = axa.get_legend_handles_labels()
order = [2, 1, 0]
axa.legend([hh[i] for i in order], [ll[i] for i in order],
           frameon=False, fontsize=6.9, loc='upper left', ncol=3,
           handlelength=1.7, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.2)

for kind, c, lw, lab, dx in [('mean', '#222222', 1.6, 'mean over tokens (primary)', -0.06),
                               ('max', '#8A8A8A', 1.1, 'max over tokens', 0.06)]:
    ys = [lev[f'{kind}_{k}']['auc'] for k in keys]
    lo = [ci[f'{kind}_{k}'][0] for k in keys]
    hi = [ci[f'{kind}_{k}'][1] for k in keys]
    axb.errorbar([x + dx for x in xb], ys,
                 yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)],
                 color=c, marker='o', ms=4.6, lw=lw, capsize=2, elinewidth=0.9,
                 label=lab, zorder=3)

axb.axhline(zz['shortcut_auc'], color='#888888', lw=0.9, ls='--', zorder=1)
axb.text(2.38, zz['shortcut_auc'] + 0.012, 'brightness and density only',
         ha='right', va='bottom', fontsize=6.8, color='#555555')
axb.axhline(0.5, color='#BBBBBB', lw=0.8, ls=':', zorder=1)
axb.text(2.38, 0.505, 'chance', ha='right', va='bottom', fontsize=6.8, color='#777777')
axb.set_xticks(xb)
axb.set_xticklabels(pitch, fontsize=7.2)
axb.set_xlim(-0.4, 2.4)
axb.set_ylim(0.45, 1.0)
axb.set_xlabel('Pixel pitch of the calcification crop (\u00b5m)', fontsize=8)
axb.set_ylabel('AUC, calcification crop vs normal tissue', fontsize=8)
axb.legend(frameon=False, fontsize=7.0, loc='upper left', handlelength=2.0,
           bbox_to_anchor=(0.01, 0.76))
axb.text(0.02, 0.43,
         f"native minus 900-px view: +{d['delta']:.3f}\n95% interval +{d['lo']:.3f} to +{d['hi']:.3f}",
         transform=axb.transAxes, ha='left', va='center', fontsize=6.9,
         color='#333333', linespacing=1.25)

for ax, L in zip((axa, axb), 'ab'):
    ax.text(-0.02, 1.015, L, transform=ax.transAxes, fontsize=10,
            fontweight='bold', ha='right', va='bottom')

fig.subplots_adjust(left=0.085, right=0.99, top=0.91, bottom=0.20, wspace=0.27)
os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)
fig.savefig(os.path.join(ROOT, 'figures', 'fig5_resolution_place.png'), dpi=300)