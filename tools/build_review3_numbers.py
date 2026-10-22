#!/usr/bin/env python3
"""Rows added in the revised manuscript: share of unknown labels by examination type and the ratio of the random-draw gap
to the pooled gap. Idempotent: removes its own rows (ids S10_UNK_SHARE_*, S10_DRAWS_RATIO) and appends them again.
Run from the package root: python3 tools/build_review3_numbers.py"""
import os, csv
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pth = os.path.join(ROOT, 'data', 'paper4_numbers.csv')
rows = [r for r in csv.DictReader(open(pth, encoding='utf-8', newline='')) if not (r['id'].startswith('S10_UNK_SHARE_') or r['id'] == 'S10_DRAWS_RATIO')]
G = {(r['id'], r['finding']): r for r in rows}
b5 = pd.read_csv(os.path.join(ROOT, 'results', 's10', 's10an_tables', 'b5_test_by_examtype.csv'))
new = []
src = 'computed from results/s10/s10an_tables/b5_test_by_examtype.csv: unknown / (present + absent + unknown) breasts of the stratum, in percent'
for tag, et in (('scr', 'screening'), ('dia', 'diagnostic')):
    for f in ('calc', 'asymmetry', 'mass'):
        t = b5[(b5.finding == f) & (b5.exam_type == et)]
        share = 100.0 * t[t.state == 'unknown'].breasts.sum() / t.breasts.sum()
        new.append(dict(id=f'S10_UNK_SHARE_{tag}', finding=f, value=repr(round(float(share), 4)), lo='', hi='', source=src, status='computed'))
for f in ('calc', 'asymmetry', 'mass'):
    ratio = float(G[('S10_DRAWS_K3', f)]['value']) / float(G[('K3', f)]['value'])
    new.append(dict(id='S10_DRAWS_RATIO', finding=f, value=repr(round(ratio, 4)), lo='', hi='',
                    source='computed: S10_DRAWS_K3 (mean over 200 random one-examination draws) / K3 (pooled)', status='computed'))
rows += new
with open(pth, 'w', encoding='utf-8', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['id', 'finding', 'value', 'lo', 'hi', 'source', 'status'], lineterminator='\n'); w.writeheader(); w.writerows(rows)
print('added', len(new), 'rows;', {(r['id'], r['finding']): round(float(r['value']), 3) for r in new})
