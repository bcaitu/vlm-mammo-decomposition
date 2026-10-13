#!/usr/bin/env python3
"""Tables S6 (recipes of all arms) and S7 (training updates, selected checkpoints and passes over the training subset, per size and seed).
Reads results/s6lc/lc_training_budget.csv (aggregate metadata read from the training logs of the compute server by build/extract_training_budget.py) and
(re)writes the BUDGET_* rows of data/paper4_numbers.csv. Run from the repository root: python3 tools/build_tables_s6_s7.py
Definitions: V arms, updates = selected optimiser step (16 breasts per step); D, updates = selected epoch x updates per epoch (batch 16); passes = updates x 16 / breasts in the training subset."""
import os, csv
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
b = pd.read_csv(os.path.join(ROOT, 'results', 's6lc', 'lc_training_budget.csv'))
BS = 16
v = b[b.arm.isin(['V', 'V_ctl', 'VC', 'V2x'])].copy(); v['upd'] = v.selected_step; v['pass_'] = v.upd * BS / v.subset_rows
d = b[b.arm == 'D'].copy(); d['upd'] = d.selected_epoch * d.epochs_or_updates_per_epoch; d['pass_'] = d.upd * BS / d.subset_rows
frac = [('f32', '1/32'), ('f16', '1/16'), ('f8', '1/8'), ('f4', '1/4'), ('f2', '1/2'), ('f1', 'whole pool')]
fi = lambda x: f'{int(round(x)):,}'
slash = lambda s: '/'.join(fi(x) for x in s)
# ---------- Table S7
L = [r'\begin{table}[h]', r'\centering\scriptsize\setlength{\tabcolsep}{3pt}',
     r'\caption{Training length of the VLM (V) and of the specialist head (D) at every training size. Breasts: mean size of the nested subset over three seeds. Updates are optimiser updates at 16 breasts each: for V the selected step; for D the selected epoch times the updates per epoch (one D epoch covers the positives and a 1:1 redrawn sample of EMBED negatives, hence fewer rows than the subset). Passes: updates $\times$ 16 divided by the number of breasts of the subset, mean over seeds. Seeds 0/1/2 are separated by slashes. V at the sizes 1/32 to 1/2 was trained on one cosine schedule of 2,295 steps and the checkpoint after 765, 1,530 or 2,295 steps was chosen by validation AUC of the token score, so checkpoints before the last were taken before the learning rate had decayed; V at the whole pool (4,591 steps) was not selected. D was trained for up to 20 epochs with patience 4, and the epoch with the highest dataset-mean macro F1 on the validation split was kept. Last block: other arms at the whole pool.}\label{tab:budget}',
     r'\begin{tabular}{@{}lrrcrccrr@{}}', r'\toprule',
     r' & & \multicolumn{3}{c}{V: token score of the LoRA VLM} & \multicolumn{4}{c}{D: specialist head} \\',
     r'\cmidrule(lr){3-5}\cmidrule(lr){6-9}',
     r'Size & Breasts & Schedule & Selected step & Passes & Selected epoch & Updates & Passes & D/V updates \\', r'\midrule']
num = {}
for k, lab in frac:
    vv = v[(v.arm == 'V') & (v['size'] == k)].sort_values('seed'); dd = d[d['size'] == k].sort_values('seed')
    ratio = dd.upd.mean() / vv.upd.mean(); num[k] = ratio
    L.append(f"{lab} & {fi(vv.subset_rows.mean())} & {fi(vv.steps_scheduled.iloc[0])} & {slash(vv.upd)} & {vv.pass_.mean():.2f} & " +
             '/'.join(str(int(x)) for x in dd.selected_epoch) + f" & {fi(dd.upd.mean())} & {dd.pass_.mean():.1f} & {ratio:.1f} " + r'\\')
L += [r'\midrule', r'\multicolumn{9}{@{}l}{\textit{Other arms at the whole pool (73,470 breasts)}}\\']
for arm, lab in [('V_ctl', r'V$_{\mathrm{ctl}}$ (2,295-step schedule)'), ('VC', 'VC (vision-side LoRA)'), ('V2x', 'V2x (9,182 steps)')]:
    vv = v[v.arm == arm].sort_values('seed')
    L.append(lab + f" & 73,470 & {fi(vv.steps_scheduled.iloc[0])} & {slash(vv.upd)} & {vv.pass_.mean():.2f} & \\multicolumn{{4}}{{c}}{{}} " + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
(os.makedirs(os.path.join(ROOT, 'tables'), exist_ok=True), open(os.path.join(ROOT, 'tables', 'tableS7_budget.tex'), 'w', encoding='utf-8'))[1].write('\n'.join(L) + '\n')
# ---------- Table S6
def row(*c): return ' & '.join(c) + r' \\' + '\n'
H6 = r"""\begin{table}[p]
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{Recipes of the compared arms as run. The contrast between two arms is a contrast between these complete recipes (input, labels, loss, sampling, number of updates and selection), not between architectures alone. Unknown labels are masked in every loss and excluded from every AUC. Training sizes: nested subsets of the pool of 73,470 breasts; passes and updates per size and seed are in Table~\ref{tab:budget}.}\label{tab:recipes}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{1.7cm}>{\raggedright\arraybackslash}p{2.9cm}>{\raggedright\arraybackslash}p{3.3cm}>{\raggedright\arraybackslash}p{2.5cm}>{\raggedright\arraybackslash}p{3.1cm}>{\raggedright\arraybackslash}p{2.0cm}@{}}
\toprule
Arm & Input & Target and loss & Sampling & Updates and selection & Learning rate \\
\midrule
"""
rows6 = [
 ('V, V$_{\\mathrm{ctl}}$, V2x', 'Both views at 896~px and a query naming the side; Qwen3-VL-8B with language-side LoRA (rank 16)',
  'Templated report from the reference labels (density, mass, calcifications, asymmetry, distortion, BI-RADS); next-token cross-entropy on supervised report tokens',
  'Rows of the pool or subset, reshuffled each pass; no class weighting; 16 breasts per step', 'V: 4,591 steps, last checkpoint. V$_{\\mathrm{ctl}}$ and sizes 1/32 to 1/2: 2,295-step schedule, checkpoint (765, 1,530 or 2,295) by validation AUC of the token score. V2x: 9,182 steps',
  r'$10^{-4}$, AdamW without weight decay, 3\% warm-up, cosine decay over the scheduled steps'),
 ('VC', 'As V; LoRA also in the vision tower, merger and DeepStack merger', 'As V', 'As V', '4,591 steps, last checkpoint', r'$10^{-4}$ (language side), $2\times10^{-5}$ (vision side)'),
 ('VP, VCP', 'Frozen hidden states of V or VC (layers 0 to 36; mean over image tokens or last prompt token)',
  'Known label of one finding; class-balanced L2 logistic regression (L-BFGS, 150 iterations)', 'All rows of the training subset with a known label', 'No gradient update of the VLM; layer, pooling and $C\\in\\{0.001,0.01,0.1\\}$ per finding by validation AUC', r'not applicable'),
 ('A$_{\\mathrm{probe}}$, A$_{\\mathrm{lin}}$', 'Same, on hidden states of the base model without fine-tuning', 'As VP', 'As VP', 'As VP (A$_{\\mathrm{lin}}$: layer 0 only)', 'not applicable'),
 ('A$_{\\mathrm{tf}}$', 'Layer-0 tokens (vision tower and merger output) of both views of the base model, $2\\times2$ mean-pooled',
  'Known label of one finding; class-weighted binary cross-entropy; two-layer transformer head', 'Rows of the training subset with a known label', 'Up to 30 epochs, patience 6 on validation AUC', r'$3\times10^{-4}$, AdamW, weight decay 0.05'),
 ('D', 'Cached $2\\times2$-pooled tokens of frozen DINOv2 ViT-B/14, both views, no augmentation',
  'All concept heads and the BI-RADS head of the stage-1 model; masked, class-weighted cross-entropy (class weight power 0.5; head weights of the stage-1 configuration)',
  'EMBED negatives redrawn at 1:1 each epoch; batch 16', 'Up to 20 epochs, patience 4; epoch with the highest dataset-mean macro F1 over the validation split', r'$3\times10^{-4}$, AdamW, weight decay 0.05, 5\% warm-up, cosine decay over 20 epochs'),
 ('D$_{\\mathrm{ref}}$', 'Original stage-1 runs on images with augmentation (whole pool only)', 'As D', 'As D', 'As the stage-1 configuration', 'Stage-1 configuration'),
]
body6 = ''.join(row(*r) for r in rows6)
open(os.path.join(ROOT, 'tables', 'tableS6_recipes.tex'), 'w', encoding='utf-8').write(H6 + body6 + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n' + r'\end{table}' + '\n')
# ---------- BUDGET_* rows of the numbers table
pth = os.path.join(ROOT, 'data', 'paper4_numbers.csv')
rows = [r for r in csv.DictReader(open(pth, encoding='utf-8', newline='')) if not r['id'].startswith('BUDGET_')]
vall = pd.concat([v[v.arm.isin(['V', 'V_ctl'])], v[v.arm.isin(['V2x'])]])
src = 'computed from data/lc_training_budget.csv by build/build_tables_s6_s7.py (passes = selected updates x 16 / breasts in the subset)'
new = [('BUDGET_V_pass_min', vall.pass_.min()), ('BUDGET_V_pass_max', vall.pass_.max()), ('BUDGET_D_pass_min', d.pass_.min()), ('BUDGET_D_pass_max', d.pass_.max()),
       ('BUDGET_ratio_f1', num['f1']), ('BUDGET_ratio_f2', num['f2']), ('BUDGET_V_pass_f1', v[(v.arm == 'V') & (v['size'] == 'f1')].pass_.mean()), ('BUDGET_D_pass_f1', d[d['size'] == 'f1'].pass_.mean()),
       ('BUDGET_V_pass_f2', v[(v.arm == 'V') & (v['size'] == 'f2')].pass_.mean()), ('BUDGET_D_pass_f2', d[d['size'] == 'f2'].pass_.mean())]
for i, x in new: rows.append(dict(id=i, finding='', value=repr(round(float(x), 6)), lo='', hi='', source=src, status='computed'))
with open(pth, 'w', encoding='utf-8', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['id', 'finding', 'value', 'lo', 'hi', 'source', 'status'], lineterminator='\n'); w.writeheader(); w.writerows(rows)
print('Tables S6, S7 written;', {i: round(float(x), 2) for i, x in new})
