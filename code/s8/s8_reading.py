#!/usr/bin/env python3
"""s8: applies the outcome table of the pre-registration (section 6) mechanically to s8_contrasts.csv; separately for calcifications and asymmetry; no other interpretation.
Rows of the table are evaluated independently; every row that is satisfied is listed; if none is satisfied the outcome is 'other' (described as is).
usage: python3 s8_reading.py --contrasts analysis/s8_contrasts.csv --out analysis/s8_reading.md"""
import argparse, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--contrasts", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
K = pd.read_csv(a.contrasts); NAME = {"calc": "кальцинаты (подтверждающая)", "asymmetry": "асимметрия (подтверждающая)", "mass": "масса (контроль, описательно)"}
def get(prefix, f):
    d = K[K.contrast.str.startswith(prefix) & (K.finding == f)]; return None if d.empty else d.iloc[0]
def cat(v):
    if v is None: return "нет данных"
    s = str(v.verdict)
    for k, c in (("adds", "adds"), ("equivalent", "equivalent"), ("reduces", "reduces"), ("below", "below"), ("not below", "notbelow"), ("unclear", "unclear")):
        if s.startswith(k): return c
    return "unclear"
# 'not below' starts with 'not', 'below' with 'below': order of the checks above is safe because 'not below' does not start with 'below'.
RU = {"adds": "добавляет", "equivalent": "эквивалентно", "reduces": "снижает", "below": "ниже", "notbelow": "не ниже", "unclear": "неясно", "нет данных": "нет данных"}
fmt = lambda r: "нет данных" if r is None else f"{r.delta:+.3f} [{r.lo:+.3f}; {r.hi:+.3f}]"
out = ["# s8: чтение исходов по таблице предрегистрации (раздел 6), механически\n"]
for f in ("calc", "asymmetry", "mass"):
    s1, s2, s3, s4, s5 = (get(p, f) for p in ("S1", "S2", "S3", "S4", "S5")); c1, c2, c3, c4 = cat(s1), cat(s2), cat(s3), cat(s4)
    out.append(f"## {NAME[f]}\n")
    out.append(f"- S1 (VC − V, вклад башни): {fmt(s1)}: {RU[c1]}\n- S2 (VC − D, разрыв с головой стадии 1): {fmt(s2)}: {RU[c2]}\n- S3 (V2x − V, вклад бюджета шагов): {fmt(s3)}: {RU[c3]}\n- S4 (VCP − VP, линейно доступная информация): {fmt(s4)}: {RU[c4]}\n- S5 (VC − VCP, потеря на интерфейсе, описательно): {fmt(s5)}\n")
    rows = []
    if c1 == "adds" and c2 == "notbelow": rows.append("S1 «добавляет», S2 «не ниже»: замороженная башня была главным ограничением рецепта; с дообученной башней VLM достигает головы стадии 1.")
    if c1 == "adds" and c2 == "below": rows.append("S1 «добавляет», S2 «ниже»: башня объясняет часть разрыва, но не весь; остаток за считыванием, входом или метками.")
    if c1 == "equivalent" and c3 == "adds": rows.append("S1 «эквивалентно», S3 «добавляет»: ограничивает бюджет обучения, а не замороженная башня.")
    if c1 == "equivalent" and c3 == "equivalent": rows.append("S1 «эквивалентно», S3 «эквивалентно»: ни башня, ни бюджет шагов в проверенных пределах; ограничение лежит во входе (896 px, метки уровня железы) или в классе рецепта.")
    if c4 == "adds" and c1 == "equivalent": rows.append("S4 «добавляет» при S1 «эквивалентно»: башня добавляет информацию в состояния, но ответ её не использует (потеря на интерфейсе).")
    if f == "mass": out.append("Для контроля (масса) вывод по таблице не делается; строки приведены описательно:\n" + ("\n".join(f"- {x}" for x in rows) if rows else "- ни одна строка не выполнена\n") + "\n")
    else: out.append("Строки таблицы, которые выполнены:\n" + ("\n".join(f"- {x}" for x in rows) if rows else "- ни одна строка не выполнена (иное: описывается как есть, без вывода по таблице)") + "\n")
open(a.out, "w", encoding="utf-8").write("\n".join(out)); print("wrote", a.out)
