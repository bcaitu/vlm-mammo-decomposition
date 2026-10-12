#!/usr/bin/env python3
"""s6lc: applies the outcome-reading table of the pre-registration (section 6) mechanically to lc_contrasts.csv; no other interpretation.
usage: python3 lc_reading.py --contrasts runs_lc/analysis/lc_contrasts.csv --out reading.md"""
import argparse, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--contrasts", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
K = pd.read_csv(a.contrasts); NAME = {"calc": "кальцинаты (подтверждающая)", "asymmetry": "асимметрия (подтверждающая)", "mass": "масса (контроль, описательно)"}
def get(prefix, f, sz="f1"):
    d = K[K.contrast.str.startswith(prefix) & (K.finding == f) & (K["size"] == sz)]
    return None if d.empty else d.iloc[0]
def cat(v):
    if v is None: return "нет данных"
    s = str(v.verdict)
    return "above" if s.startswith("above") else "below" if s.startswith("below") else "equivalent" if s.startswith("equivalent") else "notabove" if s.startswith("not above") else "rising" if s.startswith("rising") else "plateau" if s.startswith("plateau") else "inconclusive"
RU = {"above": "выше", "below": "ниже", "equivalent": "эквивалентно", "notabove": "не выше", "rising": "рост", "plateau": "плато", "inconclusive": "неопределённо", "нет данных": "нет данных"}
out = ["# Чтение исходов по таблице предрегистрации (раздел 6), механически\n"]
for f in ("calc", "asymmetry", "mass"):
    k1, k2, k3 = get("K1", f), get("K2", f), get("K3 V - D", f); enc = get("decomp: D - A_tf", f); k4 = get("K4", f, "f4-f1")
    c1, c2, c3, ce, c4 = cat(k1), cat(k2), cat(k3), cat(enc), cat(k4)
    fmt = lambda r: "нет данных" if r is None else f"{r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]"
    out.append(f"## {NAME[f]}\n")
    out.append(f"- K1 (VP − A*): {fmt(k1)}: {RU[c1]}\n- K2 (V − VP): {fmt(k2)}: {RU[c2]}\n- K3 (V − D): {fmt(k3)}: {RU[c3]}\n- D − A_tf (вклад энкодера): {fmt(enc)}: {RU[ce]}\n- K4 (прирост V за удвоение): {fmt(k4)}: {RU[c4]}\n")
    r = []
    if c1 in ("notabove", "equivalent") and c2 == "equivalent": r.append("LoRA на языковой части переформатирует то, что даёт замороженная башня; потолок задают башня и класс считывания.")
    elif c1 in ("notabove", "equivalent") and c2 == "below": r.append("Информация есть в скрытых состояниях дообученной модели, но собственная оценка использует её хуже (потеря на интерфейсе).")
    elif c1 == "above": r.append("Дообучение языковой части повышает линейно доступную информацию относительно лучшего считывания A* (нелинейная обработка тех же токенов; информации сверх токенов оно не добавляет).")
    else: r.append(f"Сочетание K1 = {RU[c1]}, K2 = {RU[c2]} вне таблицы предрегистрации; описывается как есть.")
    if c3 == "below" and ce == "above": r.append("Отставание от головы стадии 1 в основном объясняется энкодером (DINOv2 против башни Qwen).")
    elif c3 == "below" and ce == "equivalent": r.append("Отставание от головы стадии 1 объясняется не энкодером, а считыванием и обучением VLM.")
    elif c3 == "below": r.append(f"K3 «ниже», D − A_tf = {RU[ce]}: вне таблицы; описывается как есть.")
    else: r.append(f"K3 = {RU[c3]}: строка о разрыве с головой стадии 1 не применяется.")
    if c4 == "rising": r.append("Кривая V при этом рецепте ещё поднимается с числом примеров.")
    elif c4 == "plateau": r.append("Кривая V вышла на плато при этом рецепте.")
    else: r.append("K4 неопределённо: вывод о росте или плато не делается.")
    out.append("Вывод по таблице:\n" + "\n".join(f"- {x}" for x in r) + "\n")
open(a.out, "w", encoding="utf-8").write("\n".join(out)); print("wrote", a.out)
