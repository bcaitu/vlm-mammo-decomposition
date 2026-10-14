#!/usr/bin/env python3
"""Approximate word count of the generated main.tex by section (figure environments, table inputs and comments removed)."""
import re, sys
s = open(sys.argv[1] if len(sys.argv) > 1 else 'main.tex').read()
def clean(t):
    t = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', t, flags=re.S)
    t = re.sub(r'\\input\{[^}]*\}', '', t)
    t = t.replace(r'\%', 'PCT')
    t = re.sub(r'(?m)%.*$', '', t)
    t = re.sub(r'\\cite\{[^}]*\}', '', t)
    t = re.sub(r'\\(ref|label)\{[^}]*\}', 'X', t)
    t = re.sub(r'\\[a-zA-Z]+\*?', '', t)
    t = re.sub(r'[\$\{\}~]', ' ', t)
    return t
secs = ['Abstract', 'Introduction', 'Results', 'Discussion', 'Methods', 'Data availability']
tot = 0
for a, b in zip(secs, secs[1:]):
    i = s.index('\\section*{' + a + '}'); j = s.index('\\section*{' + b + '}')
    w = len(clean(s[i:j]).split()); print(f'{a:14s} {w}')
    if a in ('Introduction', 'Results', 'Discussion'): tot += w
print('Introduction + Results + Discussion:', tot, '(journal guideline 4,500; Methods, abstract, legends and references excluded)')
