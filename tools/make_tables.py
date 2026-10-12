#!/usr/bin/env python3
"""Build Tables 1, 2, S1, S2, S3 from data/paper4_numbers.csv (run from the manuscript folder: python3 build/make_tables.py).
Table S4 is built by build/build_tables_extra.py from the output tables of the earlier studies.
Static captions and column layouts are stored below; every number is read from the canonical table."""
import csv, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(ROOT, 'data', 'paper4_numbers.csv'), encoding='utf-8', newline='')))
N = {(r['id'], r['finding']): r for r in rows}
FIND = ['calc', 'asymmetry', 'mass']
def g(i, f=''):
    return N[(i, f)]
def sg(x):
    r = round(float(x), 3)
    s = f'{abs(r):.3f}'
    return ('$-$' + s) if r < 0 else ('+' + s)
def un(x):
    return f'{round(float(x), 3):.3f}'
def ax_(arm, f):
    r = g(f'AUC_{arm}', f); return f"{un(r['value'])} [{un(r['lo'])}, {un(r['hi'])}]"
def cx(k, f):
    r = g(k, f); return f"{sg(r['value'])} [{sg(r['lo'])}, {sg(r['hi'])}]"
def pr(x): return f'{x:,}'
def wr(name, s):
    os.makedirs(os.path.join(ROOT, 'tables'), exist_ok=True)
    open(os.path.join(ROOT, 'tables', name), 'w', encoding='utf-8').write(s)

# ---- Table 1
H1 = r"""\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\caption{AUC (95\% interval over test patients) at the full training size (73,470 breasts) on the EMBED test set (13,178 breasts, 2,743 patients). Means over three seeds; the linear probe on base-model states is deterministic (one run). $^{\dagger}$Layer and pooling chosen on the validation sample (layer 8, a language-model layer, for all three findings).}\label{tab:auc}
\begin{tabular}{@{}lccc@{}}
\toprule
Arm & Calcifications & Asymmetry & Mass (control) \\
\midrule
"""
F1 = r"""\bottomrule
\end{tabular}
\end{table}
"""
ARMROWS = [
 ('VLM own answer', [('V0', 'Base model, no fine-tuning'), ('V_blank', 'LoRA VLM, black images (control)'), ('V_ctl', 'LoRA VLM, 2,295 steps'), ('V', 'LoRA VLM, 4,591 steps (reference)'), ('V2x', 'LoRA VLM, 9,182 steps'), ('VC', 'LoRA VLM with tower LoRA, 4,591 steps')]),
 ('Probe on VLM hidden states', [('VP', 'language-side LoRA'), ('VCP', 'language-side and tower LoRA')]),
 ('Read-outs of the base model (no fine-tuning)', [('A_probe', r'linear probe on hidden states$^{\dagger}$'), ('A_tf', 'transformer head on tower tokens')]),
 ('Specialist encoder (stage 1)', [('D', 'DINOv2 head, cached features'), ('D_ref', 'DINOv2 head, original runs with augmentation')])]
body = ''
for grp, arms in ARMROWS:
    body += r'\multicolumn{4}{@{}l}{\textit{' + grp + r'}}\\' + '\n'
    for arm, lab in arms:
        body += r'\hspace{1em}' + lab + ' & ' + ' & '.join(ax_(arm, f) for f in FIND) + r' \\' + '\n'
wr('table1_auc.tex', H1 + body + F1)

# ---- Table 2
H2 = r"""\begin{table}[tp]
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\caption{Pre-registered (K1--K4, S1--S5), pre-specified descriptive (V$-$V$_{\mathrm{ctl}}$) and post hoc (P2, P6) contrasts in test AUC at the full training size, with 95\% intervals over test patients. Calcifications and asymmetry are the confirmatory findings; mass is a descriptive control. A difference of 0.05 AUC or more was pre-specified as practically significant. A$^{*}$ (K1): the better of the linear probe on hidden states and the transformer head on tower tokens of the base model, chosen on validation (the transformer head at the full size). K4: the top size was trained for 4,591 steps and the lower sizes for 2,295; threshold 0.02.}\label{tab:contrasts}
\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{4.1cm}ccc@{}}
\toprule
ID & Contrast: question & Calcifications & Asymmetry & Mass (control) \\
\midrule
"""
F2 = r"""\bottomrule
\end{tabular}
\end{table}
"""
CROWS = [('K1', r'VP $-$ A$^{*}$', 'Does language-side LoRA add information readable by a probe beyond the base model?'),
 ('K2', r'V $-$ VP', "Is information lost between hidden states and the model's own answer?"),
 ('K3', r'V $-$ D', 'Gap between the VLM answer and the specialist head'),
 ('K4', r'V gain per doubling', 'Gain of the VLM curve per doubling of training breasts, last two doublings'),
 ('DESC_V-Vctl', r'V $-$ V$_{\mathrm{ctl}}$', 'Doubling of steps at the full size, 2,295 to 4,591'),
 ('S1', r'VC $-$ V', 'Tower LoRA added to language-side LoRA'),
 ('S3', r'V2x $-$ V', 'Doubling of steps, 4,591 to 9,182'),
 ('S4', r'VCP $-$ VP', 'Tower LoRA, information readable by a linear probe'),
 ('S2', r'VC $-$ D', 'Gap to the specialist head after tower LoRA'),
 ('S5', r'VC $-$ VCP', 'Loss between hidden states and answer after tower LoRA'),
 ('P2', r'VCP $-$ D', 'Post hoc: probe on tower-tuned states against the specialist head'),
 ('P6', r'(V2x$-$V)$-$(V$-$V$_{\mathrm{ctl}}$)', 'Post hoc: second doubling minus first doubling')]
body = ''
for k, lab, q in CROWS:
    body += ('desc.' if k == 'DESC_V-Vctl' else k) + ' & ' + lab + ': ' + q + ' & ' + ' & '.join(cx(k, f) for f in FIND) + r' \\' + '\n'
    if k in ('DESC_V-Vctl', 'S5'): body += r'\addlinespace' + '\n'
wr('table2_contrasts.tex', H2 + body + F2)

# ---- Table S1
HS1 = r"""\begin{table}[h]
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\caption{Data used in the full-size comparison. Counts are breasts (rows); positives are breasts with the finding known to be present; breasts with an unknown label for a finding are excluded from that finding. Test negatives: mass 9,781, calcifications 9,784, asymmetry 9,781; test breasts with unknown label: 2,877, 2,824 and 2,359.}\label{tab:data}
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Role & Set & Breasts & Patients & Mass & Calc. & Asym. \\
\midrule
"""
FS1 = r"""\bottomrule
\end{tabular}
\end{table}
"""
body = ''
for role, nm, lab in [('Training pool', 'pool_cbis_ddsm', 'CBIS-DDSM'), ('', 'pool_vindr', 'VinDr-Mammo'), ('', 'pool_embed', 'EMBED'), ('', 'pool_total', r'\textit{Pool total}'), ('Validation', 'val_sample_embed', 'EMBED validation sample'), ('Test', 'test_embed', 'EMBED test')]:
    v = lambda k, f='': int(round(float(g(f'COUNT_{nm}_{k}', f)['value'])))
    body += f"{role} & {lab} & {pr(v('rows'))} & {pr(v('patients'))} & {pr(v('positives', 'mass'))} & {pr(v('positives', 'calc'))} & {pr(v('positives', 'asymmetry'))} " + r'\\' + '\n'
wr('tableS1_data.tex', HS1 + body + FS1)

# ---- Table S2 (chain of studies; literal facts from the study reports and the server hash records)
HS2 = r"""\begin{table}[h]
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{Chain of studies with protocols frozen by file hashes before the runs (``pre-registered''; see Methods). Times are UTC+5 and refer to the recording of file hashes on the server before the corresponding runs; hash counts are files that matched after the runs.}\label{tab:chain}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{1.6cm}>{\raggedright\arraybackslash}p{3.6cm}>{\raggedright\arraybackslash}p{2.0cm}>{\raggedright\arraybackslash}p{1.8cm}>{\raggedright\arraybackslash}p{2.4cm}>{\raggedright\arraybackslash}p{3.0cm}@{}}
\toprule
Study & Question & Frozen & Hashes & Deviations & Outcome by the rule \\
\midrule
"""
FS2 = r"""\bottomrule
\end{tabular}
\end{table}
"""
chain = [
 (r'''s4v12 (V1, V2)''', r'''Detection and image dependence of the VLM report against the concept route, one epoch''', r'''2026-10-04 13:07''', r'''9 of 9 (13:07 and 21:07)''', r'''none after freeze; post hoc diagnostics labelled''', r'''V1-1, V1-2: no stable difference; controls V1-3 and V2-1 not met'''),
 (r'''s4v3eval''', r'''Fresh rows; metrics without a constant-report floor''', r'''2026-10-05 17:00''', r'''25 files''', r'''none in the analysis''', r'''VLM worse in detection and BI-RADS category (3 of 4 sets) and in CMMD malignancy AUC'''),
 (r'''s4diag''', r'''Perception or verbalization (2,000 EMBED validation breasts)''', r'''2026-10-05 21:36''', r'''13 of 13''', r'''none''', r'''intermediate'''),
 (r'''s4res''', r'''Whole-image resolution 896, 1344, 1792 px, probe on base-model hidden states''', r'''2026-10-05 23:09''', r'''8 of 8''', r'''none''', r'''unclear'''),
 (r'''s6lc''', r'''Learning curves and decomposition (V, VP, A, D)''', r'''2026-10-06 21:58; phase 2 later''', r'''68 of 68 and 6 of 6''', r'''explanatory notes only''', r'''no row of the outcome table fully met'''),
 (r'''s7zoom''', r'''Native-resolution crops with known location''', r'''2026-10-07 19:38''', r'''5 of 5''', r'''none''', r'''unclear'''),
 (r'''s8''', r'''Vision-tower LoRA and doubled steps''', r'''2026-10-08 15:06; phase 2 20:07''', r'''24 of 25 (one expected mismatch), 5 of 5, 1 of 1''', r'''deviation 1: verdict order aligned with the protocol text before any test metric''', r'''calcifications: S1 equivalent, S3 adds; asymmetry: S1 adds, S2 below'''),
]
body = ''.join(' & '.join(row) + r' \\' + '\n' for row in chain)
wr('tableS2_chain.tex', HS2 + body + FS2)

# ---- Table S3
HS3 = r"""\begin{table}[h]
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\caption{Post hoc contrasts (not pre-registered; written and run after the main analysis). Differences in test AUC with 95\% intervals over test patients. P3 was also listed as a pre-specified descriptive contrast in the learning-curve study (Table~2 of the main text).}\label{tab:posthoc}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{5.2cm}ccc@{}}
\toprule
Contrast & Calcifications & Asymmetry & Mass (control) \\
\midrule
"""
FS3 = r"""\bottomrule
\end{tabular}
\end{table}
"""
PHN = [('P1', r'P1: VP $-$ D'), ('P2', r'P2: VCP $-$ D'), ('P3', r'P3: V $-$ V$_{\mathrm{ctl}}$ (first doubling)'), ('P4', r'P4: V2x $-$ V (second doubling)'), ('P5', r'P5: V2x $-$ V$_{\mathrm{ctl}}$ (two doublings)'), ('P6', r'P6: second minus first doubling')]
body = ''.join(lab + ' & ' + ' & '.join(cx(k, f) for f in FIND) + r' \\' + '\n' for k, lab in PHN)
body += r'\midrule' + '\n' + r'\multicolumn{4}{@{}l}{\textit{Distance to the specialist head D (arm $-$ D)}}\\' + '\n'
for arm, lab in [('A_probe', r'A$_{\mathrm{probe}}$'), ('A_tf', r'A$_{\mathrm{tf}}$'), ('VP', 'VP'), ('VCP', 'VCP'), ('V_ctl', r'V$_{\mathrm{ctl}}$'), ('V', 'V'), ('V2x', 'V2x'), ('VC', 'VC')]:
    body += lab + ' $-$ D & ' + ' & '.join(cx(f'GAP_{arm}-D', f) for f in FIND) + r' \\' + '\n'
wr('tableS3_posthoc.tex', HS3 + body + FS3)
print('tables written: table1_auc, table2_contrasts, tableS1_data, tableS2_chain, tableS3_posthoc')
