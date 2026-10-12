#!/usr/bin/env python3
"""s6lc: descriptive / post hoc numbers for the report (no new inference; point estimates from lc_curve.csv, lc_contrasts.csv, probe grids and the queue log).
usage: python3 lc_posthoc.py --dir analysis --out analysis/posthoc.md
Files read: lc_curve.csv, lc_contrasts.csv, grids/{A_f1_s0,V_f1_s0,V_f1_s1,V_f1_s2}_grid.csv (validation sample VS only), queue2.log."""
import argparse, re, datetime as dt, collections
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
C = pd.read_csv(f"{a.dir}/lc_curve.csv"); K = pd.read_csv(f"{a.dir}/lc_contrasts.csv")
SZ = ["f32", "f16", "f8", "f4", "f2", "f1"]; FIND = [("calc", "кальцинаты"), ("asymmetry", "асимметрия"), ("mass", "масса (контроль)")]
def auc(arm, f, s):
    r = C[(C.arm == arm) & (C.finding == f) & (C["size"] == s)]; return float(r.auc.iloc[0]) if len(r) else float("nan")
out = ["# s6lc: описательные и post hoc числа (точечные оценки; интервалов нет, если не указано)\n"]
out.append("## 1. V − D по размерам (из lc_contrasts.csv, интервалы по пациентам теста; описательно)\n")
out.append("| находка | " + " | ".join(SZ) + " |"); out.append("|---|" + "---|" * len(SZ))
for f, ru in FIND:
    d = K[(K.finding == f) & K.contrast.eq("K3 V - D")].set_index("size")
    out.append(f"| {ru} | " + " | ".join(f"{d.loc[s].delta:+.3f} [{d.loc[s].lo:+.3f}, {d.loc[s].hi:+.3f}]" for s in SZ) + " |")
out.append("\n## 2. Средний прирост AUC за удвоение числа строк обучения (post hoc, точечно)\n")
out.append("| находка | рукав | f32→f4 (3 удвоения) | f4→f1 (2 удвоения) |"); out.append("|---|---|---|---|")
for f, ru in FIND:
    for arm in ["V", "D", "VP", "A_probe", "A_tf"]:
        v = [auc(arm, f, s) for s in SZ]; out.append(f"| {ru} | {arm} | {(v[3]-v[0])/3:+.3f} | {(v[5]-v[3])/2:+.3f} |")
    eq = (auc("V_ctl", f, "f1") - auc("V", f, "f4")) / 2
    out.append(f"| {ru} | V при равном бюджете 2 295 шагов (f4→f2→f1 с V_ctl) | — | {eq:+.3f} |")
out.append("\n## 3. Верхний размер (f1): разности относительно эталонов, точечно (post hoc)\n")
out.append("| находка | V − A_probe | VP − A_probe | D − A_probe | V − A_tf | D − A_tf | V − VP | V − D |"); out.append("|---|---|---|---|---|---|---|---|")
for f, ru in FIND:
    g = lambda x, y: auc(x, f, "f1") - auc(y, f, "f1")
    out.append(f"| {ru} | {g('V','A_probe'):+.3f} | {g('VP','A_probe'):+.3f} | {g('D','A_probe'):+.3f} | {g('V','A_tf'):+.3f} | {g('D','A_tf'):+.3f} | {g('V','VP'):+.3f} | {g('V','D'):+.3f} |")
out.append("\n## 4. Профиль по слоям: лучший AUC на VS (80 / 134 / 85 положительных: кальцинаты / асимметрия / масса; шум порядка ±0,03–0,04), лучший C, f1\n")
G = {n: pd.read_csv(f"{a.dir}/grids/{n}_grid.csv") for n in ["A_f1_s0", "V_f1_s0", "V_f1_s1", "V_f1_s2"]}
LAY = [0, 8, 16, 20, 24, 28, 32, 36]
for f, ru in FIND:
    for pool, pl in (("img", "среднее по токенам изображения"), ("last", "последний токен подсказки")):
        base = G["A_f1_s0"][(G["A_f1_s0"].finding == f) & (G["A_f1_s0"].pool == pool)].groupby("layer").auc_vs.max().reindex(LAY)
        ft = pd.concat([G[n][(G[n].finding == f) & (G[n].pool == pool)].groupby("layer").auc_vs.max().reindex(LAY) for n in ["V_f1_s0", "V_f1_s1", "V_f1_s2"]], axis=1).mean(axis=1)
        out.append(f"\n{ru}, пулинг: {pl}\n"); out.append("| модель | " + " | ".join(str(l) for l in LAY) + " |"); out.append("|---|" + "---|" * len(LAY))
        out.append("| базовая | " + " | ".join("—" if np.isnan(x) else f"{x:.3f}" for x in base) + " |")
        out.append("| дообученная V (среднее по 3 сидам) | " + " | ".join("—" if np.isnan(x) else f"{x:.3f}" for x in ft) + " |")
lines = open(f"{a.dir}/queue2.log").read().splitlines(); day = dt.date(2026, 10, 6); prev = None; start = {}; dur = {}
for ln in lines:
    m = re.match(r"(\d\d):(\d\d):(\d\d)\s+(START|DONE|FAILED)\s+(\S+)", ln)
    if not m: continue
    h, mi, s, kind, name = m.groups(); t = dt.time(int(h), int(mi), int(s))
    if prev is not None and t < prev: day += dt.timedelta(days=1)
    prev = t; ts = dt.datetime.combine(day, t)
    if kind == "START": start[name] = ts
    elif kind == "DONE" and name in start: dur[name] = (ts - start[name]).total_seconds() / 3600
cat = collections.defaultdict(float); cnt = collections.Counter()
for k, v in dur.items(): c = k.split("_")[0]; cat[c] += v; cnt[c] += 1
first = min(start.values()); last = max(start[k] + dt.timedelta(hours=dur[k]) for k in dur)
out.append(f"\n## 5. Время основной очереди (queue2.log; пилот не входит)\n\nЗадач с временем: {len(dur)}; сумма времени задач {sum(dur.values()):.1f} ч; календарное время {(last-first).total_seconds()/3600:.1f} ч ({first:%Y-%m-%d %H:%M} — {last:%Y-%m-%d %H:%M}, UTC+5).\n")
out.append("| группа задач | число | часы |"); out.append("|---|---|---|")
for k, v in sorted(cat.items(), key=lambda x: -x[1]): out.append(f"| {k} | {cnt[k]} | {v:.1f} |")
open(a.out, "w", encoding="utf-8").write("\n".join(out) + "\n"); print("wrote", a.out)
