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
def wr(name, s): os.makedirs(os.path.join(ROOT, 'tables'), exist_ok=True); open(os.path.join(ROOT, 'tables', name), 'w', encoding='utf-8').write(s)

# ---- Table 1
H1 = r"""\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\caption{AUC (95\% interval over test patients) at the full training size (73,470 breasts) on the EMBED test set (13,178 breasts, 2,743 patients). Each finding is scored on the test breasts with a known label (positives and negatives: calcifications 570 and 9,784; asymmetry 1,038 and 9,781; mass 520 and 9,781); all arms use the same breasts except the black-image control$^{\S}$. VLM rows are teacher-forced negative-token scores. Means over three seeds; the base model and the linear probe on its states are deterministic (one run). VC: LoRA also in the vision tower, merger and DeepStack merger. $^{\dagger}$Layer and pooling chosen on the validation sample (layer 8, a language-model layer, for all three findings). $^{\ddagger}$Schedule of 2,295 steps; the checkpoint (2,295, 1,530 and 765 steps for seeds 0, 1 and 2) was chosen on the validation sample. $^{\S}$Sample of 3,000 test breasts, single run.}\label{tab:auc}
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
 ('VLM token score (teacher-forced negative-token score)', [('V0', 'Base model, no fine-tuning'), ('V_blank', r'LoRA VLM, black images (control)$^{\S}$'), ('V_ctl', r'LoRA VLM, 2,295-step schedule$^{\ddagger}$'), ('V', 'LoRA VLM, 4,591 steps (reference)'), ('V2x', 'LoRA VLM, 9,182 steps'), ('VC', 'LoRA VLM + vision-side LoRA, 4,591 steps')]),
 ('Probe on VLM hidden states', [('VP', 'language-side LoRA'), ('VCP', 'language-side and vision-side LoRA')]),
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
\caption{Protocol-frozen contrasts (K1--K4, S1--S5; the confirmatory family for calcifications and asymmetry), a descriptive contrast fixed in the protocol (V$-$V$_{\mathrm{ctl}}$), the paired contrast VP$-$A$_{\mathrm{probe}}$ added in the last study and post hoc contrasts (P2, P6) in test AUC at the full training size, with 95\% intervals over test patients (conditional on the trained models; seed-to-seed variation is in Table S5). Mass is a descriptive control. A difference of 0.05 AUC or more was fixed in the protocols as practically significant; an interval within $\pm$0.03 (K3: $\pm$0.05) was labelled equivalent. A$^{*}$ (K1): the base-model read-out with the higher validation AUC, chosen per training size over the three findings (the transformer head on tower tokens at the full size, although the linear probe has the higher test AUC for calcifications; VP$-$A$_{\mathrm{probe}}$ is given separately). K4: the top size was trained for 4,591 steps, the lower sizes on a 2,295-step schedule with a validation-selected checkpoint (Table S7), so scheduled and selected steps differ; threshold 0.02. V$_{\mathrm{ctl}}$: 2,295-step schedule with the checkpoint chosen on validation (765 to 2,295 steps). Vision side: tower, merger and DeepStack merger.}\label{tab:contrasts}
\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{4.1cm}ccc@{}}
\toprule
ID & Contrast: question & Calcifications & Asymmetry & Mass (control) \\
\midrule
"""
F2 = r"""\bottomrule
\end{tabular}
\end{table}
"""
CROWS = [('K1', r'VP $-$ A$^{*}$', 'Does language-side LoRA add information readable by a probe beyond the base-model read-out A$^{*}$?'),
 ('DESC_VP-A_probe', r'VP $-$ A$_{\mathrm{probe}}$', 'Probe on VLM states against the linear probe on base-model states (paired, last study)'),
 ('K2', r'V $-$ VP', "Does a probe on the hidden states rank better than the model's token score?"),
 ('K3', r'V $-$ D', 'Gap between the VLM token score and the specialist head'),
 ('K4', r'V gain per doubling', 'Gain of the VLM curve per doubling of training breasts, last two doublings'),
 ('DESC_V-Vctl', r'V $-$ V$_{\mathrm{ctl}}$', '4,591 steps against the 2,295-step schedule with validation-selected checkpoint'),
 ('S1', r'VC $-$ V', 'Vision-side LoRA added to language-side LoRA'),
 ('S3', r'V2x $-$ V', 'Doubling of steps, 4,591 to 9,182'),
 ('S4', r'VCP $-$ VP', 'Vision-side LoRA, information readable by a linear probe'),
 ('S2', r'VC $-$ D', 'Gap to the specialist head after vision-side LoRA'),
 ('S5', r'VC $-$ VCP', 'Token score minus a probe on the same states after vision-side LoRA'),
 ('P2', r'VCP $-$ D', 'Post hoc: probe on tower-tuned states against the specialist head'),
 ('P6', r'(V2x$-$V)$-$(V$-$V$_{\mathrm{ctl}}$)', 'Post hoc: second doubling minus first doubling')]
body = ''
for k, lab, q in CROWS:
    cells = [cx('S10_K1b', f) for f in FIND] if k == 'DESC_VP-A_probe' else [cx(k, f) for f in FIND]
    body += ('desc.' if k.startswith('DESC_') else k) + ' & ' + lab + ': ' + q + ' & ' + ' & '.join(cells) + r' \\' + '\n'
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
HS2 = r"""{\scriptsize\setlength{\tabcolsep}{2.5pt}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{1.35cm}>{\raggedright\arraybackslash}p{2.7cm}>{\raggedright\arraybackslash}p{1.6cm}>{\raggedright\arraybackslash}p{1.6cm}>{\raggedright\arraybackslash}p{2.0cm}>{\raggedright\arraybackslash}p{2.6cm}>{\raggedright\arraybackslash}p{2.7cm}@{}}
\caption{Chain of studies with protocols frozen by file hashes before the runs (``protocol-frozen''; see Methods). Times are UTC+5 and refer to the recording of file hashes on the server before the corresponding runs; hash counts are files that matched after the runs. The last column states what was known to the authors when the protocol was frozen.}\label{tab:chain}\\
\toprule
Study & Question & Frozen & Hashes & Deviations & Outcome by the rule & Known when frozen \\
\midrule
\endfirsthead
\multicolumn{7}{@{}l}{\textit{Table S2 (continued)}}\\
\toprule
Study & Question & Frozen & Hashes & Deviations & Outcome by the rule & Known when frozen \\
\midrule
\endhead
\bottomrule
\endfoot
"""
FS2 = r"""\end{longtable}}
"""
chain = [
 (r'''s4v12 (V1, V2)''', r'''Detection and image dependence of the VLM report against the concept route, one epoch''', r'''2026-10-04 13:07''', r'''9 of 9 (13:07 and 21:07)''', r'''none after freeze; post hoc diagnostics labelled''', r'''V1-1, V1-2: no stable difference; controls V1-3 and V2-1 not met''', r'''No VLM result; results of the concept route known'''),
 (r'''s4v3eval''', r'''Fresh rows; metrics without a constant-report floor''', r'''2026-10-05 17:00''', r'''25 files''', r'''none in the analysis''', r'''VLM worse in detection and BI-RADS category (3 of 4 sets) and in CMMD malignancy AUC''', r'''s4v12 results, including the uninformative slot metric'''),
 (r'''s4diag''', r'''Perception or verbalization (2,000 EMBED validation breasts)''', r'''2026-10-05 21:36''', r'''13 of 13''', r'''none''', r'''intermediate''', r'''s4v12 and s4v3eval results; designed as exploratory'''),
 (r'''s4res''', r'''Whole-image resolution 896, 1344, 1792 px, probe on base-model hidden states''', r'''2026-10-05 23:09''', r'''8 of 8''', r'''none''', r'''unclear''', r'''s4diag results; designed as exploratory'''),
 (r'''s6lc''', r'''Learning curves and decomposition (V, VP, A, D)''', r'''2026-10-06 21:58; phase 2 later''', r'''68 of 68 and 6 of 6''', r'''explanatory notes only''', r'''no row of the outcome table fully met''', r'''Earlier studies of the same adapters in aggregate; no s6lc test metric'''),
 (r'''s7zoom''', r'''Native-resolution crops with known location''', r'''2026-10-07 19:38''', r'''5 of 5''', r'''none''', r'''unclear''', r'''s6lc results; designed as exploratory'''),
 (r'''s8''', r'''Vision-tower LoRA and doubled steps''', r'''2026-10-08 15:06; phase 2 20:07''', r'''24 of 25 (one expected mismatch), 5 of 5, 1 of 1''', r'''deviation 1: verdict order aligned with the protocol text before any test metric''', r'''calcifications: S1 equivalent, S3 adds; asymmetry: S1 adds, S2 below''', r'''All s6lc test metrics, including V minus D; no metric of VC, VCP or V2x'''),
 (r'''s10an''', r'''Analyses on the saved test scores of s6lc and s8: VP minus A$_{\mathrm{probe}}$ with an interval (B1), one examination per patient (B2), own against black images on the same breasts (B3), label counts and composition (B4, B5)''', r'''2026-10-13 23:08 (with s10db)''', r'''37 of 37 (checked before each analysis and after both)''', r'''verdict column of B1 written with the s6lc rule instead of the ``gain'' rule of the protocol (same numbers); analysis job restarted once after a hash-check command error, no code change''', r'''B1: equivalent (calcifications, asymmetry); B2: labels of K2, K3, S3 and V minus V$_{\mathrm{ctl}}$ change on the earliest examination; B3: image dependence shown''', r'''All s6lc and s8 test metrics and the second external review'''),
 (r'''s10db''', r'''DINOv2 head trained for the number of updates of V (D$_{\mathrm{bV}}$) and of V2x (D$_{\mathrm{b2x}}$); contrasts C1--C3''', r'''2026-10-13 23:08 (with s10an)''', r'''37 of 37; 21 runs, exit code 0, updates as specified; K3 reproduced to $10^{-16}$''', r'''none in the analysis; environment variable and dry-run flag added before the freeze''', r'''asymmetry: C1 not below, C2 reduces (read by the protocol rule as explained by the longer training of D; D$_{\mathrm{bV}}$ also differs in schedule and checkpoint rule); calcifications: C1 unclear, C2 reduces (``other''); C3 unclear and below''', r'''All s6lc and s8 test metrics; the expectation written in section 7 of the protocol (``the asymmetry gap remains'') was not confirmed'''),
 (r'''post hoc''', r'''Examination-type strata (screening, diagnostic) for all arms and contrasts''', r'''not frozen; scripts written after B2 and B5 had been seen; hashes of the main outputs recorded before the run''', r'''--''', r'''--''', r'''V within diagnostic examinations near chance; gaps larger within strata than pooled''', r'''All s10an and s10db results'''),
]
body = ''.join(' & '.join(row) + r' \\' + '\n' for row in chain)
wr('tableS2_chain.tex', HS2 + body + FS2)

# ---- Table S3
HS3 = r"""\begin{table}[h]
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\caption{Post hoc contrasts (not frozen before the main analysis; written and run after it). Differences in test AUC with 95\% intervals over test patients. P3 was also listed as a pre-specified descriptive contrast in the learning-curve study (Table~2 of the main text).}\label{tab:posthoc}
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
