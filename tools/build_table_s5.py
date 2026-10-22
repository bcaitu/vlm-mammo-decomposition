#!/usr/bin/env python3
"""Table S5 (seed-to-seed variation at the full training size) from the per-run AUC files results/s6lc/lc_auc_by_run.csv and results/s8/s8_auc_by_run.csv. Run from the repository root: python3 tools/build_table_s5.py"""
import os, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lc = pd.read_csv(os.path.join(ROOT, 'results', 's6lc', 'lc_auc_by_run.csv')); s8 = pd.read_csv(os.path.join(ROOT, 'results', 's8', 's8_auc_by_run.csv'))
f1 = lc[lc['size'] == 'f1']
s10 = pd.read_csv(os.path.join(ROOT, 'results', 's10', 's10db_tables', 's10db_auc_by_run.csv')); s10 = s10[s10['size'] == 'f1']   # s10db: per-run AUCs at the whole pool
def get(arm):
    if arm in ('D_bV', 'D_b2x'): return s10[s10.arm == arm]
    d = (s8 if arm in ('VC', 'VCP', 'V2x') else f1); return d[d.arm == arm]
def arm10(arm, f): return s10[(s10.arm == arm) & (s10.finding == f)].sort_values('seed_index').auc.values
groups = [('VLM token score', [('V0', 'Base model, no fine-tuning'), ('V_blank', 'LoRA VLM, black images (control)'), ('V_ctl', 'LoRA VLM, 2,295-step schedule'), ('V', 'LoRA VLM, 4,591 steps (reference)'), ('V2x', 'LoRA VLM, 9,182 steps'), ('VC', 'LoRA VLM + vision-side LoRA, 4,591 steps')]),
          ('Probe on VLM hidden states', [('VP', 'language-side LoRA'), ('VCP', 'language-side and vision-side LoRA')]),
          ('Read-outs of the base model (no fine-tuning)', [('A_probe', 'linear probe on hidden states'), ('A_tf', 'transformer head on tower tokens')]),
          ('Specialist encoder (stage 1)', [('D', 'DINOv2 head, cached features'), ('D_ref', 'DINOv2 head, original runs with augmentation')]),
          ('Specialist head trained for the updates of the VLM (s10db; final weights)', [('D_bV', r'D$_{\mathrm{bV}}$, updates of V at the whole pool (4,591)'), ('D_b2x', r'D$_{\mathrm{b2x}}$, updates of V2x (9,182)')])]
CONTR = [(r'C1: V $-$ D$_{\mathrm{bV}}$', 'V', 'D_bV'), (r'C2: D$_{\mathrm{bV}}$ $-$ D', 'D_bV', 'D'), (r'C3: V2x $-$ D$_{\mathrm{b2x}}$', 'V2x', 'D_b2x')]
sgn = lambda x: f'{x:+.3f}'.replace('-', '$-$')
def cell(arm, finding):
    v = get(arm); v = v[v.finding == finding]['auc']
    if len(v) == 1: return f'{v.mean():.3f} (one run)'
    sd = v.std(ddof=1)
    if v.nunique() == 1: return f'{v.mean():.3f} (det.)'
    return f'{v.mean():.3f} (' + ('$<$0.001' if sd < 0.0005 else f'{sd:.3f}') + ')'
L = [r'\begin{table}[h]', r'\centering\scriptsize\setlength{\tabcolsep}{3pt}',
     r'\caption{Seed-to-seed variation at the full training size (EMBED test set). Mean AUC over three seeds and, in brackets, the sample standard deviation ($n=3$). Entries marked det. are deterministic estimates without a training seed (the base model and the linear probe on its states); the black-image control is a single run on a sample of 3,000 test breasts. The 2,295-step schedule used a checkpoint chosen on validation (2,295, 1,530 and 765 steps for seeds 0, 1 and 2), which contributes to its seed variation. The patient-bootstrap intervals of these means are in Table 1 of the main text; they do not contain the seed-to-seed variation. D$_{\mathrm{bV}}$ and D$_{\mathrm{b2x}}$ are the heads of study s10db (cosine schedule compressed to the updates of V and V2x, final weights; three seeds). The last block gives the contrasts C1--C3 per seed (the arm of seed $i$ minus the other arm of seed $i$), their mean and sample standard deviation; these are arithmetic on the saved per-run AUCs of the full test set, not a new analysis.}\label{tab:seeds}',
     r'\begin{tabular}{@{}lccc@{}}', r'\toprule', r'Arm & Calcifications & Asymmetry & Mass (control) \\', r'\midrule']
for gname, arms in groups:
    L.append(r'\multicolumn{4}{l}{\emph{' + gname + r'}}\\')
    for arm, lab in arms:
        L.append(r'\quad ' + lab + ' & ' + ' & '.join(cell(arm, f) for f in ('calc', 'asymmetry', 'mass')) + r' \\')
L.append(r'\midrule'); L.append(r'\multicolumn{4}{l}{\emph{Per-seed contrasts of s10db at the whole pool (AUC difference, same seed on both sides)}}\\')
for lab, a, b in CONTR:
    x = {f: arm10(a, f) - arm10(b, f) for f in ('calc', 'asymmetry', 'mass')}
    L.append(r'\quad ' + lab + ', mean (SD) & ' + ' & '.join(f'{sgn(x[f].mean())} ({x[f].std(ddof=1):.3f})' for f in x) + r' \\')
    L.append(r'\quad\quad seeds 0/1/2 & ' + ' & '.join('/'.join(sgn(v) for v in x[f]) for f in x) + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
os.makedirs(os.path.join(ROOT, 'tables'), exist_ok=True)
open(os.path.join(ROOT, 'tables', 'tableS5_seeds.tex'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('Table S5 written')
