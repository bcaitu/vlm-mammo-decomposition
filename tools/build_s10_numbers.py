#!/usr/bin/env python3
"""Append the s10 numbers (s10db: D at the update budget of V; s10an: analyses on saved scores;
post hoc: examination-type strata) to data/paper4_numbers.csv.  Idempotent: rows whose id starts
with S10_ are removed first.  Run from the package root: python3 build/build_s10_numbers.py"""
import pandas as pd, numpy as np
D = 'results/s10/'
num = pd.read_csv('data/paper4_numbers.csv')
num = num[~num.id.str.startswith('S10_')].copy()
rows = []
def add(id_, finding, value, lo=None, hi=None, source='', status=''):
    rows.append(dict(id=id_, finding=finding, value=value, lo='' if lo is None or pd.isna(lo) else lo,
                     hi='' if hi is None or pd.isna(hi) else hi, source=source, status=status))
F = ['mass', 'calc', 'asymmetry']
# --- s10db contrasts (whole pool, pooled test) ---
c = pd.read_csv(D + 's10db_tables/s10db_contrasts.csv')
cmap = {'C1 V - D_bV (gap at matched updates, whole pool)': 'C1',
        'C2 D_bV - D (cost of the update budget of V to D, whole pool)': 'C2',
        'C3 V2x - D_b2x (gap at the updates of V2x, whole pool)': 'C3',
        "desc D_b2x - D_bV (value of doubling D's updates)": 'Db2x-DbV', 'desc D_b2x - D': 'Db2x-D'}
for k, v in cmap.items():
    for _, r in c[c.contrast == k].iterrows():
        add(f'S10_{v}', r.finding, r.delta, r.lo, r.hi, 's10db s10db_contrasts.csv', f'{r.kind}; verdict: {r.verdict}')
for sz in ['f32', 'f16', 'f8', 'f4', 'f2', 'f1']:
    for _, r in c[c.contrast == f'desc V - D_bV at {sz}'].iterrows():
        add(f'S10_VDbV_{sz}', r.finding, r.delta, r.lo, r.hi, 's10db s10db_contrasts.csv', 'descriptive by size')
for arm in ['V', 'D_bV', 'D']:
    for _, r in c[c.contrast == f'desc gain per doubling over the last two doublings, {arm}'].iterrows():
        add(f'S10_GAIN_{arm}', r.finding, r.delta, r.lo, r.hi, 's10db s10db_contrasts.csv', 'descriptive')
rd = pd.read_csv(D + 's10db_tables/s10db_reading.csv')
for _, r in rd.iterrows():
    add('S10_SHARE_C2_K3', r.finding, r.share_of_K3_due_to_D_training_length, source='s10db s10db_reading.csv', status='(D_bV - D)/(V - D), point estimate')
cu = pd.read_csv(D + 's10db_tables/s10db_curve.csv')
for _, r in cu.iterrows():
    add(f'S10_AUC_{r.arm}_{r["size"]}', r.finding, r.auc, r.lo, r.hi, 's10db s10db_curve.csv', f'seed mean over {r.n_seeds} seeds; sd_seeds={r.sd_seeds:.4f}')
    if r['size'] == 'f1':
        add(f'S10_AUC_{r.arm}', r.finding, r.auc, r.lo, r.hi, 's10db s10db_curve.csv', f'whole pool; seed mean; sd_seeds={r.sd_seeds:.4f}')
        add(f'S10_SD_{r.arm}', r.finding, r.sd_seeds, source='s10db s10db_curve.csv', status='between-seed sd, whole pool')
# --- s10an B1 ---
b1 = pd.read_csv(D + 's10an_tables/b1_paired.csv')
for _, r in b1[b1.contrast == 'K1b VP - A_probe'].iterrows():
    add('S10_K1b', r.finding, r.delta, r.lo, r.hi, 's10an b1_paired.csv', 'VP - A_probe, paired; protocol rule "gain", margin 0.03')
pt = pd.read_csv(D + 's10an_tables/b1_probe_vs_tf.csv')
for _, r in pt.iterrows():
    add('S10_Aprobe-Atf', r.finding, r.delta, r.lo, r.hi, 's10an b1_probe_vs_tf.csv', 'descriptive')
# --- B2 earliest examination per patient ---
b2 = pd.read_csv(D + 's10an_tables/b2_exam_primary.csv')
nm = {'K1 VP - A_tf (A* at the full size)': 'K1', 'K1b VP - A_probe': 'K1b', 'K2 V - VP': 'K2', 'K3 V - D': 'K3', 'S1 VC - V': 'S1',
      'S2 VC - D': 'S2', 'S3 V2x - V': 'S3', 'S4 VCP - VP': 'S4', 'S5 VC - VCP': 'S5', 'desc V - V_ctl': 'V-Vctl'}
for _, r in b2[b2.set == 'earliest examination per patient'].iterrows():
    add(f'S10_EARLY_{nm[r.contrast]}', r.finding, r.delta, r.lo, r.hi, 's10an b2_exam_primary.csv', f'earliest examination per patient; verdict: {r.verdict}')
bc = pd.read_csv(D + 's10an_tables/b2_exam_counts.csv')
for _, r in bc.iterrows():
    add('S10_EARLY_rows', r.finding, r.rows, source='s10an b2_exam_counts.csv', status='labelled breasts, earliest examination')
    add('S10_EARLY_npos', r.finding, r.n_pos, source='s10an b2_exam_counts.csv'); add('S10_EARLY_nneg', r.finding, r.n_neg, source='s10an b2_exam_counts.csv')
bd = pd.read_csv(D + 's10an_tables/b2_exam_draws.csv')
for _, r in bd[bd.contrast == 'K3 V - D'].iterrows():
    add('S10_DRAWS_K3', r.finding, r['mean'], r.p2_5, r.p97_5, 's10an b2_exam_draws.csv', '200 random one-examination-per-patient draws; percentiles over draws (not a CI)')
# --- B3 blank ---
b3 = pd.read_csv(D + 's10an_tables/b3_blank.csv')
for _, r in b3.iterrows():
    add('S10_BLANK_rows', r.finding, r.rows, source='s10an b3_blank.csv'); add('S10_BLANK_npos', r.finding, r.n_pos, source='s10an b3_blank.csv')
    add('S10_BLANK_V_seed0', r.finding, r.auc_V_seed0, source='s10an b3_blank.csv', status='V seed 0 on the 3,000-row sample')
    add('S10_BLANK_Vmean', r.finding, r.auc_V_seedmean, source='s10an b3_blank.csv', status='V seed mean on the 3,000-row sample')
    add('S10_BLANK_auc', r.finding, r.auc_blank, r.blank_lo, r.blank_hi, 's10an b3_blank.csv')
    add('S10_V-blank', r.finding, r.diffmean_V_blank, r.dm_lo, r.dm_hi, 's10an b3_blank.csv', 'seed-mean V minus V_blank, same 3,000 rows, paired')
# --- B5 exam type ---
b5 = pd.read_csv(D + 's10an_tables/b5_test_by_examtype.csv')
for _, r in b5.iterrows():
    add(f'S10_N_{r.exam_type}_{r.state}', r.finding, r.breasts, source='s10an b5_test_by_examtype.csv', status='EMBED test, labelled breasts by examination type')
comp = pd.read_csv(D + 's10an_tables/b5_composition.csv')
t = comp[(comp.partition == 'EMBED test')]
s = t[t.variable == 'summary'].iloc[0]
add('S10_TEST_exams', '', s.examinations, source='s10an b5_composition.csv'); add('S10_TEST_exams_per_patient_mean', '', s.exams_per_patient_mean, source='s10an b5_composition.csv')
add('S10_TEST_age_median', '', s.age_median, s.age_q1, s.age_q3, 's10an b5_composition.csv', 'median [IQR]')
for _, r in t[t.variable != 'summary'].iterrows():
    add(f'S10_TEST_{r.variable.split(" ")[0]}_{str(r.category).replace(" ", "_")}', '', r.n, source='s10an b5_composition.csv')
# --- post hoc: exam type ---
p1 = pd.read_csv(D + 's10an_posthoc_tables/ph1_examtype_indicator.csv')
for _, r in p1.iterrows():
    add('S10_PH_examtype_auc', r.finding, r.auc_exam_type_indicator, r.lo, r.hi, 'post hoc ph1_examtype_indicator.csv', 'AUC of the examination-type indicator (diagnostic = 1)')
    add('S10_PH_share_diag_pos', r.finding, r.share_diagnostic_among_positives, source='post hoc ph1_examtype_indicator.csv')
    add('S10_PH_share_diag_neg', r.finding, r.share_diagnostic_among_negatives, source='post hoc ph1_examtype_indicator.csv')
p2 = pd.read_csv(D + 's10an_posthoc_tables/ph2_auc_by_stratum.csv')
st = {'all rows': 'all', 'screening only': 'scr', 'diagnostic only': 'dia', 'earliest examination': 'early'}
for _, r in p2.iterrows():
    add(f'S10_PH_AUC_{r.arm}_{st[r.stratum]}', r.finding, r.auc, r.lo, r.hi, 'post hoc ph2_auc_by_stratum.csv', f'n_pos={r.n_pos}, n_neg={r.n_neg}')
p3 = pd.read_csv(D + 's10an_posthoc_tables/ph3_contrasts_by_stratum.csv')
for _, r in p3.iterrows():
    key = nm.get(r.contrast) or nm[r.contrast + ' (A* at the full size)']
    add(f'S10_PH_{key}_{st[r.stratum]}', r.finding, r.delta, r.lo, r.hi, 'post hoc ph3_contrasts_by_stratum.csv', f'verdict: {r.verdict}')
da = pd.read_csv(D + 's10db_posthoc_tables/dbph_auc_by_stratum.csv')
for _, r in da.iterrows():
    add(f'S10_DBPH_AUC_{r.arm}_{st[r.stratum]}', r.finding, r.auc, r.lo, r.hi, 'post hoc dbph_auc_by_stratum.csv')
dc = pd.read_csv(D + 's10db_posthoc_tables/dbph_contrasts_by_stratum.csv')
dm = {'C1 V - D_bV': 'C1', 'C2 D_bV - D': 'C2', 'C3 V2x - D_b2x': 'C3', 'desc K3 V - D': 'K3', 'desc D_b2x - D_bV': 'Db2x-DbV'}
for _, r in dc.iterrows():
    add(f'S10_DBPH_{dm[r.contrast]}_{st[r.stratum]}', r.finding, r.delta, r.lo, r.hi, 'post hoc dbph_contrasts_by_stratum.csv', f'verdict: {r.verdict}')
# shares of K3 explained by C2 within screening
for f in F:
    k3 = dc[(dc.contrast == 'desc K3 V - D') & (dc.stratum == 'screening only') & (dc.finding == f)].delta.iloc[0]
    c2 = dc[(dc.contrast == 'C2 D_bV - D') & (dc.stratum == 'screening only') & (dc.finding == f)].delta.iloc[0]
    add('S10_DBPH_SHARE_C2_K3_scr', f, c2 / k3, source='post hoc dbph_contrasts_by_stratum.csv', status='(D_bV - D)/(V - D) in the screening stratum, point estimate')
new = pd.DataFrame(rows)
out = pd.concat([num, new], ignore_index=True)
out.to_csv('data/paper4_numbers.csv', index=False)
print(len(num), 'kept;', len(new), 'S10 rows added;', len(out), 'total')
