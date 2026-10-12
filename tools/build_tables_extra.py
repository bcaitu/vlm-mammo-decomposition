#!/usr/bin/env python3
"""Table S4 (test sets of the earlier studies) from the output tables of s4v12, s4v3eval, s4diag, s4res, s7zoom. Run from the manuscript folder."""
import pandas as pd, zipfile, io, os, json
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def rd(z, n): return pd.read_csv(os.path.join(D, 'results', n))
pp = rd(None, 's4v12/posthoc_prior.csv').set_index('split')
fm = rd(None, 's4v3eval/fresh/tables/fresh_main.csv')
fr = {r.split: (int(r.n), int(r.patients)) for _, r in fm.drop_duplicates('split').iterrows()}
names = [('test_embed', 'EMBED Hologic'), ('test_embed_fuji', 'EMBED Fuji holdout'), ('test_embed_ge', 'EMBED GE holdout'), ('test_vindr', 'VinDr-Mammo'), ('test_cbis_ddsm', 'CBIS-DDSM'), ('test_cdd_cesm', 'CDD-CESM'), ('test_inbreast', 'INbreast'), ('test_cmmd', 'CMMD')]
f = lambda x: f'{x:,}'
rows = []
for k, lab in names:
    a = f(int(pp.loc[k, 'n']))
    b = f'{f(fr[k][0])} ({f(fr[k][1])})' if k in fr else 'not used'
    rows.append(f'{lab} & {a} & {b} \\\\')
dm = rd(None, 's4diag/diag_main.csv').set_index('slot')
zz = json.load(open(f'{D}/results/s7zoom/s7zoom_probe_results.json'))
tex = [r'\begin{table}[h]', r'\centering\footnotesize\setlength{\tabcolsep}{4pt}',
 r'\caption{Test sets of the earlier studies. s4v12: breasts per set in the first comparison (total ' + f(int(pp.n.sum())) + r'). s4v3eval: fresh breasts (patients) per set; sets with all rows used in s4v12 had no fresh rows. Probe studies (s4diag and s4res): test part of the 2,000-breast EMBED validation sample, positives and negatives for mass, calcifications and asymmetry: ' +
 ', '.join(f"{int(dm.loc[s,'n_pos_test'])}/{int(dm.loc[s,'n_neg_test'])}" for s in ['mass', 'calc', 'asymmetry']) +
 r'. Zoom study (s7zoom): ' + f(zz['n']) + ' windows (' + f(zz['n_pos']) + ' positive) from ' + f(zz['n_groups']) + r' studies of the VinDr-Mammo training split.}\label{tab:sets}',
 r'\begin{tabular}{@{}lrr@{}}', r'\toprule', r'Set & s4v12 & s4v3eval, fresh rows \\', r'\midrule'] + rows + [r'\bottomrule', r'\end{tabular}', r'\end{table}']
(os.makedirs(os.path.join(D,'tables'), exist_ok=True), open(os.path.join(D,'tables','tableS4_sets.tex'), 'w', encoding='utf-8'))[1].write('\n'.join(tex) + '\n')
print('\n'.join(tex)[:900])
