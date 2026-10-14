#!/usr/bin/env python3
"""Tables S8-S11 from the s10 tables (data/s10/). Run from the package root."""
import pandas as pd, numpy as np
D = 'results/s10/'
F = [('calc', 'Calcifications'), ('asymmetry', 'Asymmetry'), ('mass', 'Mass (control)')]
def sg(x, d=3):
    r = round(float(x), d); s = f"{abs(r):.{d}f}"
    return ('$-$' + s) if r < 0 else ('+' + s if r > 0 else s)
def us(x, d=3): return f"{float(x):.{d}f}"
def ci(r, signed=True, d=3):
    f = sg if signed else us
    return f"{f(r.delta if 'delta' in r else r.auc, d)} [{f(r.lo, d)}, {f(r.hi, d)}]"
def n(x): return f"{int(round(float(x))):,}"
ARM = {'V': 'LoRA VLM, token score (V)', 'D_bV': 'DINOv2 head at the updates of V (D$_{\\mathrm{bV}}$)', 'D': 'DINOv2 head, original recipe (D)',
       'V2x': 'LoRA VLM, 9,182 steps (V2x)', 'D_b2x': 'DINOv2 head at the updates of V2x (D$_{\\mathrm{b2x}}$)', 'VP': 'Probe on VLM states (VP)',
       'VC': 'VLM + vision-side LoRA (VC)', 'VCP': 'Probe on VC states (VCP)', 'V_ctl': 'LoRA VLM, 2,295-step schedule (V$_{\\mathrm{ctl}}$)',
       'A_probe': 'Base model, linear probe (A$_{\\mathrm{probe}}$)', 'A_tf': 'Base model, transformer head (A$_{\\mathrm{tf}}$)'}
SIZES = [('f32', '1/32'), ('f16', '1/16'), ('f8', '1/8'), ('f4', '1/4'), ('f2', '1/2'), ('f1', 'whole pool')]

# ---------- S8: D at the update budget of V ----------
cu = pd.read_csv(D + 's10db_tables/s10db_curve.csv'); c = pd.read_csv(D + 's10db_tables/s10db_contrasts.csv')
upd = {'f32': '765', 'f16': '765/1,530/765', 'f8': '1,530/2,295/2,295', 'f4': '1,530/2,295/2,295', 'f2': '2,295/765/765', 'f1': '4,591'}
L = [r"\begin{table}[h]", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}",
     r"\caption{Study s10db: the DINOv2 head trained for the number of optimiser updates of the VLM. D$_{\mathrm{bV}}$ is the head D of Table~S6 with its cosine schedule compressed to the number of updates that the token-score arm V received at the same training size and seed (Table~S7; the learning rate was not re-tuned); D$_{\mathrm{b2x}}$ receives the 9,182 updates of V2x (whole pool only). AUC on the EMBED test set (13,178 breasts), mean over three seeds, 95\% interval over test patients; the sampling of D (EMBED negatives redrawn 1:1 each epoch) was kept. Contrasts are differences of seed-mean AUCs within the same bootstrap resamples (paired by patient). C1--C3 are the protocol-frozen contrasts of s10db for calcifications and asymmetry; the other rows are descriptive. The protocol was frozen on 13 October 2026 after all s8 results were known.}\label{tab:s10db}",
     r"\begin{tabular}{@{}llccc@{}}", r"\toprule", r"Size (updates of V, seeds 0/1/2) & Arm & Calcifications & Asymmetry & Mass (control) \\", r"\midrule"]
for sz, lab in SIZES:
    arms = ['V', 'D_bV', 'D'] + (['V2x', 'D_b2x'] if sz == 'f1' else [])
    for i, a in enumerate(arms):
        cells = []
        for f, _ in F:
            r = cu[(cu.arm == a) & (cu['size'] == sz) & (cu.finding == f)].iloc[0]
            cells.append(ci(r, signed=False))
        first = f"{lab} ({upd[sz]})" if i == 0 else ''
        L.append(f"{first} & {ARM[a]} & " + ' & '.join(cells) + r" \\")
    if sz != 'f1': L.append(r"\addlinespace[2pt]")
L += [r"\midrule", r"\multicolumn{5}{@{}l}{\textit{Contrasts at the whole pool (difference in AUC; rule of the protocol in brackets)}}\\"]
rows = [('C1 V - D_bV (gap at matched updates, whole pool)', 'C1: V $-$ D$_{\\mathrm{bV}}$ (gap at the updates of V)'),
        ('C2 D_bV - D (cost of the update budget of V to D, whole pool)', 'C2: D$_{\\mathrm{bV}}$ $-$ D (cost to D of the update budget of V)'),
        ('C3 V2x - D_b2x (gap at the updates of V2x, whole pool)', 'C3: V2x $-$ D$_{\\mathrm{b2x}}$ (gap at the updates of V2x)'),
        ('desc K3 V - D (original, whole pool)', 'K3: V $-$ D (original recipes; s6lc)'),
        ("desc D_b2x - D_bV (value of doubling D's updates)", 'D$_{\\mathrm{b2x}}$ $-$ D$_{\\mathrm{bV}}$ (doubling the updates of D)'),
        ('desc D_b2x - D', 'D$_{\\mathrm{b2x}}$ $-$ D')]
VERD = {'not below (lower bound > -0.03)': 'not below', 'unclear': 'unclear', 'below (upper bound < 0)': 'below', 'reduces (upper bound < 0)': 'reduces',
        'adds (lower bound > 0)': 'adds', 'equivalent within +-0.03': 'equivalent'}
for k, lab in rows:
    cells = []
    for f, _ in F:
        r = c[(c.contrast == k) & (c.finding == f)].iloc[0]
        v = VERD.get(r.verdict, '')
        cells.append(ci(r) + (f" ({v})" if v and r.kind == 'confirmatory' and f != 'mass' else ''))
    L.append(r"\multicolumn{2}{@{}l}{" + lab + "} & " + ' & '.join(cells) + r" \\")
L += [r"\addlinespace[2pt]", r"\multicolumn{5}{@{}l}{\textit{V $-$ D$_{\mathrm{bV}}$ by training size (descriptive)}}\\"]
for sz, lab in SIZES:
    cells = [ci(c[(c.contrast == f'desc V - D_bV at {sz}') & (c.finding == f)].iloc[0]) for f, _ in F]
    L.append(r"\multicolumn{2}{@{}l}{" + lab + "} & " + ' & '.join(cells) + r" \\")
L += [r"\addlinespace[2pt]", r"\multicolumn{5}{@{}l}{\textit{Gain per doubling of the training breasts over the last two doublings (1/4 $\to$ whole pool; descriptive)}}\\"]
for a in ['V', 'D_bV', 'D']:
    cells = [ci(c[(c.contrast == f'desc gain per doubling over the last two doublings, {a}') & (c.finding == f)].iloc[0]) for f, _ in F]
    L.append(r"\multicolumn{2}{@{}l}{" + ARM[a] + "} & " + ' & '.join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open('tables/tableS8_s10db.tex', 'w').write('\n'.join(L) + '\n')

# ---------- S9: analyses on saved scores ----------
b1 = pd.read_csv(D + 's10an_tables/b1_paired.csv'); b2 = pd.read_csv(D + 's10an_tables/b2_exam_primary.csv')
bc = pd.read_csv(D + 's10an_tables/b2_exam_counts.csv'); bd = pd.read_csv(D + 's10an_tables/b2_exam_draws.csv'); b3 = pd.read_csv(D + 's10an_tables/b3_blank.csv')
L = [r"\begin{table}[h]", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}",
     r"\caption{Study s10an: analyses on the saved test scores of s6lc and s8 (no new training; protocol frozen on 13 October 2026). (a) Paired contrasts of the probe on VLM states with the two base-model read-outs, in the same bootstrap resamples. (b) The protocol-frozen contrasts of s6lc and s8 recomputed on the earliest examination of each patient (one examination per patient; labelled breasts: calcifications 3,247, asymmetry 3,593, mass 3,225), with the labels of the decision rules; the last row gives the mean and the 2.5th and 97.5th percentiles of K3 over 200 random draws of one examination per patient (a spread over draws, not a confidence interval). (c) The LoRA VLM (seed mean) and the same model given black images on the same 3,000 test breasts (positives: 138, 240 and 127). Differences in AUC with 95\% intervals over test patients.}\label{tab:s10an}",
     r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Contrast & Calcifications & Asymmetry & Mass (control) \\", r"\midrule",
     r"\multicolumn{4}{@{}l}{\textit{(a) Probe on VLM states against base-model read-outs, full test set}}\\"]
for k, lab in [('K1 VP - A_tf (A* at the full size)', 'K1: VP $-$ A$_{\\mathrm{tf}}$ (A$^{*}$ at the full size)'), ('K1b VP - A_probe', 'VP $-$ A$_{\\mathrm{probe}}$ (matched probe-selection rules)')]:
    L.append(lab + ' & ' + ' & '.join(ci(b1[(b1.contrast == k) & (b1.finding == f)].iloc[0]) + (f" ({VERD.get(b1[(b1.contrast == k) & (b1.finding == f)].iloc[0].verdict, '')})" if f != 'mass' else '') for f, _ in F) + r" \\")
pt = pd.read_csv(D + 's10an_tables/b1_probe_vs_tf.csv')
L.append('A$_{\\mathrm{probe}}$ $-$ A$_{\\mathrm{tf}}$ (descriptive) & ' + ' & '.join(ci(pt[pt.finding == f].iloc[0]) for f, _ in F) + r" \\")
L += [r"\addlinespace[2pt]", r"\multicolumn{4}{@{}l}{\textit{(b) Earliest examination per patient (sensitivity analysis); label of the rule in brackets}}\\"]
NM = [('K1 VP - A_tf (A* at the full size)', 'K1: VP $-$ A$^{*}$'), ('K2 V - VP', 'K2: V $-$ VP'), ('K3 V - D', 'K3: V $-$ D'), ('S1 VC - V', 'S1: VC $-$ V'),
      ('S2 VC - D', 'S2: VC $-$ D'), ('S3 V2x - V', 'S3: V2x $-$ V'), ('S4 VCP - VP', 'S4: VCP $-$ VP'), ('S5 VC - VCP', 'S5: VC $-$ VCP'), ('desc V - V_ctl', 'V $-$ V$_{\\mathrm{ctl}}$ (descriptive)')]
VE = {'not above by more than 0.05': 'not below by 0.05', 'below by more than 0.05': 'below by $>$0.05', 'not below (lower bound > -0.03)': 'not below', 'below (upper bound < 0)': 'below',
      'unclear': 'unclear', 'adds (lower bound > 0)': 'adds', 'equivalent within +-0.03': 'equivalent', 'reduces (upper bound < 0)': 'reduces', 'inconclusive': 'inconclusive'}
e = b2[b2.set == 'earliest examination per patient']
for k, lab in NM:
    cells = []
    for f, _ in F:
        r = e[(e.contrast == k) & (e.finding == f)].iloc[0]
        cells.append(ci(r) + (f" ({VE.get(r.verdict, r.verdict)})" if f != 'mass' and not k.startswith('desc') else ''))
    L.append(lab + ' & ' + ' & '.join(cells) + r" \\")
cells = [f"{sg(bd[(bd.contrast == 'K3 V - D') & (bd.finding == f)].iloc[0]['mean'])} [{sg(bd[(bd.contrast == 'K3 V - D') & (bd.finding == f)].iloc[0].p2_5)}, {sg(bd[(bd.contrast == 'K3 V - D') & (bd.finding == f)].iloc[0].p97_5)}]" for f, _ in F]
L.append('K3 over 200 draws of one examination per patient & ' + ' & '.join(cells) + r" \\")
L += [r"\addlinespace[2pt]", r"\multicolumn{4}{@{}l}{\textit{(c) Own images against black images, same 3,000 test breasts}}\\"]
L.append('LoRA VLM, own images (AUC, seed mean) & ' + ' & '.join(us(b3[b3.finding == f].iloc[0].auc_V_seedmean) for f, _ in F) + r" \\")
L.append('LoRA VLM, black images (AUC) & ' + ' & '.join(f"{us(b3[b3.finding == f].iloc[0].auc_blank)} [{us(b3[b3.finding == f].iloc[0].blank_lo)}, {us(b3[b3.finding == f].iloc[0].blank_hi)}]" for f, _ in F) + r" \\")
L.append('Own minus black images & ' + ' & '.join(f"{sg(b3[b3.finding == f].iloc[0].diffmean_V_blank)} [{sg(b3[b3.finding == f].iloc[0].dm_lo)}, {sg(b3[b3.finding == f].iloc[0].dm_hi)}]" for f, _ in F) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open('tables/tableS9_s10an.tex', 'w').write('\n'.join(L) + '\n')

# ---------- S10: labels and composition ----------
b4 = pd.read_csv(D + 's10an_tables/b4_label_counts.csv'); comp = pd.read_csv(D + 's10an_tables/b5_composition.csv'); b5 = pd.read_csv(D + 's10an_tables/b5_test_by_examtype.csv')
PART = [('cbis_ddsm', 'train (pool)', 'CBIS-DDSM, training pool'), ('vindr', 'train (pool)', 'VinDr-Mammo, training pool'), ('embed', 'train (pool)', 'EMBED, training pool'),
        ('embed', 'validation sample (2,000)', 'EMBED, validation sample'), ('embed', 'test split', 'EMBED, test set')]
FD = [('mass', 'Mass'), ('calc', 'Calcifications'), ('asymmetry', 'Asymmetry'), ('distortion', 'Distortion')]
L = [r"\begin{table}[h]", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}",
     r"\caption{Labels and composition. (a) Breasts with a present, absent or unknown label for each finding in the training pool, the validation sample and the EMBED test set (from the stage-1 manifests). CBIS-DDSM has no absent labels (its cases are lesion cases; a finding not annotated is unknown), and VinDr-Mammo has no unknown labels (every finding not annotated is absent). (b) EMBED partitions: examinations, examinations per patient, age, breast density (BI-RADS a--d), manufacturer, examination type and race as coded in the EMBED release (categories with fewer than 10 breasts are suppressed). (c) Known labels of the EMBED test set by examination type.}\label{tab:composition}",
     r"\begin{tabular}{@{}lrrrrrrrrrrrr@{}}", r"\toprule",
     r" & \multicolumn{3}{c}{Mass} & \multicolumn{3}{c}{Calcifications} & \multicolumn{3}{c}{Asymmetry} & \multicolumn{3}{c}{Distortion} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
     r"(a) Partition (breasts, patients) & pres. & abs. & unk. & pres. & abs. & unk. & pres. & abs. & unk. & pres. & abs. & unk. \\", r"\midrule"]
for ds, pa, lab in PART:
    t = b4[(b4.dataset == ds) & (b4.partition == pa)]
    pb, pp = t.partition_breasts.iloc[0], t.partition_patients.iloc[0]
    cells = []
    for f, _ in FD:
        for s in ['present', 'absent', 'unknown']:
            cells.append(n(t[(t.finding == f) & (t.state == s)].breasts.iloc[0]))
    L.append(f"{lab} ({n(pb)}, {n(pp)}) & " + ' & '.join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\vspace{6pt}", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
      r"(b) EMBED partition & Training pool & Validation sample & Test set \\", r"\midrule"]
P = ['EMBED training pool', 'EMBED validation sample', 'EMBED test']
S = {p: comp[(comp.partition == p) & (comp.variable == 'summary')].iloc[0] for p in P}
L.append('Breasts / patients / examinations & ' + ' & '.join(f"{n(S[p].breasts)} / {n(S[p].patients)} / {n(S[p].examinations)}" for p in P) + r" \\")
L.append('Examinations per patient, mean (median) & ' + ' & '.join(f"{S[p].exams_per_patient_mean:.2f} ({S[p].exams_per_patient_median:.0f})" for p in P) + r" \\")
L.append('Age, median (IQR), years & ' + ' & '.join(f"{S[p].age_median:.1f} ({S[p].age_q1:.1f}--{S[p].age_q3:.1f})" for p in P) + r" \\")
def cat(p, var, c):
    t = comp[(comp.partition == p) & (comp.variable == var) & (comp.category == c)]
    if len(t) == 0: return '0'
    v = str(t.n.iloc[0]); return v if v == '<10' else n(v)
for c_ in ['A', 'B', 'C', 'D', 'missing']:
    L.append(f"Density {c_ if c_ != 'missing' else '(missing)'} & " + ' & '.join(cat(p, 'density (tissueden)', c_) for p in P) + r" \\")
L.append('Manufacturer Hologic & ' + ' & '.join(cat(p, 'vendor', 'hologic') for p in P) + r" \\")
for c_ in ['screening', 'diagnostic']:
    L.append(f"Examination type: {c_} & " + ' & '.join(cat(p, 'examination type', c_) for p in P) + r" \\")
RC = [('African American  or Black', 'African American or Black'), ('Caucasian or White', 'Caucasian or White'), ('Asian', 'Asian'),
      ('Unknown, Unavailable or Unreported', 'Unknown, unavailable or unreported'), ('Native Hawaiian or Other Pacific Islander', 'Native Hawaiian or other Pacific Islander'),
      ('American Indian or Alaskan Native', 'American Indian or Alaskan Native'), ('Multiple', 'Multiple'), ('missing', 'Missing')]
for k, lab in RC:
    L.append(f"Race: {lab} & " + ' & '.join(cat(p, 'race (RACE_DESC)', k).replace('<10', '$<$10') for p in P) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\vspace{6pt}", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
      r"(c) EMBED test set & \multicolumn{3}{c}{Screening (11,431 breasts)} & \multicolumn{3}{c}{Diagnostic (1,747 breasts)} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
      r"Finding & present & absent & unknown & present & absent & unknown \\", r"\midrule"]
for f, lab in F:
    cells = [n(b5[(b5.finding == f) & (b5.exam_type == et) & (b5.state == s)].breasts.iloc[0]) for et in ['screening', 'diagnostic'] for s in ['present', 'absent', 'unknown']]
    L.append(lab.replace(' (control)', '') + ' & ' + ' & '.join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open('tables/tableS10_composition.tex', 'w').write('\n'.join(L) + '\n')

# ---------- S11: strata ----------
p2 = pd.read_csv(D + 's10an_posthoc_tables/ph2_auc_by_stratum.csv'); p3 = pd.read_csv(D + 's10an_posthoc_tables/ph3_contrasts_by_stratum.csv')
da = pd.read_csv(D + 's10db_posthoc_tables/dbph_auc_by_stratum.csv'); dc = pd.read_csv(D + 's10db_posthoc_tables/dbph_contrasts_by_stratum.csv'); p1 = pd.read_csv(D + 's10an_posthoc_tables/ph1_examtype_indicator.csv')
ST = [('all rows', 'All test breasts'), ('screening only', 'Screening examinations'), ('diagnostic only', 'Diagnostic examinations')]
L = [r"\begin{table}[h]", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}",
     r"\caption{Post hoc stratification of the EMBED test set by examination type (screening or diagnostic), written after the composition analysis (Table~S10c) had been seen and run on saved scores only; not part of any protocol. Known labels per stratum: screening 315 / 727 / 263 positive and 9,395 negative breasts, diagnostic 255 / 311 / 257 positive and 389 / 386 / 386 negative breasts (calcifications / asymmetry / mass). First row: AUC of the examination-type indicator itself (diagnostic = 1) as a score. AUC and paired differences with 95\% intervals over test patients; means over three seeds. The pooled values of V, D and the contrasts of s6lc and s8 are those of Tables~1, 2 and S8.}\label{tab:strata}",
     r"\begin{tabular}{@{}llccc@{}}", r"\toprule", r"Stratum & Arm or contrast & Calcifications & Asymmetry & Mass (control) \\", r"\midrule"]
L.append(r"All test breasts & Examination type as a score (diagnostic = 1) & " + ' & '.join(f"{us(p1[p1.finding == f].iloc[0].auc_exam_type_indicator)} [{us(p1[p1.finding == f].iloc[0].lo)}, {us(p1[p1.finding == f].iloc[0].hi)}]" for f, _ in F) + r" \\")
L.append(r"\midrule")
for s, slab in ST:
    arms = [('V', p2), ('V2x', p2), ('VC', p2), ('VP', p2), ('A_probe', p2), ('A_tf', p2), ('D_bV', da), ('D_b2x', da), ('D', p2)]
    for i, (a, src) in enumerate(arms):
        cells = [ci(src[(src.arm == a) & (src.stratum == s) & (src.finding == f)].iloc[0], signed=False) for f, _ in F]
        L.append((slab if i == 0 else '') + ' & ' + ARM[a] + ' & ' + ' & '.join(cells) + r" \\")
    CS = [('K3 V - D', p3, 'K3: V $-$ D'), ('K2 V - VP', p3, 'K2: V $-$ VP'), ('K1b VP - A_probe', p3, 'VP $-$ A$_{\\mathrm{probe}}$'), ('S1 VC - V', p3, 'S1: VC $-$ V'), ('S3 V2x - V', p3, 'S3: V2x $-$ V'),
          ('desc V - V_ctl', p3, 'V $-$ V$_{\\mathrm{ctl}}$'), ('C1 V - D_bV', dc, 'C1: V $-$ D$_{\\mathrm{bV}}$'), ('C2 D_bV - D', dc, 'C2: D$_{\\mathrm{bV}}$ $-$ D'), ('C3 V2x - D_b2x', dc, 'C3: V2x $-$ D$_{\\mathrm{b2x}}$')]
    for k, src, lab in CS:
        cells = [ci(src[(src.contrast == k) & (src.stratum == s) & (src.finding == f)].iloc[0]) for f, _ in F]
        L.append(' & ' + lab + ' & ' + ' & '.join(cells) + r" \\")
    if s != 'diagnostic only': L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open('tables/tableS11_strata.tex', 'w').write('\n'.join(L) + '\n')
print('tables S8-S11 written')
