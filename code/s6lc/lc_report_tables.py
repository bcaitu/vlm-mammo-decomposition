#!/usr/bin/env python3
"""s6lc: markdown tables for the report from runs_lc/analysis/{lc_curve,lc_contrasts,lc_auc_by_run}.csv (no interpretation; numbers are copied from the files).
usage: python3 lc_report_tables.py --dir runs_lc/analysis --out tables.md"""
import argparse, json, os
import pandas as pd, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
C = pd.read_csv(f"{a.dir}/lc_curve.csv"); K = pd.read_csv(f"{a.dir}/lc_contrasts.csv"); meta = json.load(open(f"{a.dir}/lc_meta.json")) if os.path.exists(f"{a.dir}/lc_meta.json") else {}
SZ = ["f32", "f16", "f8", "f4", "f2", "f1"]; FIND = [("mass", "масса (контроль)"), ("calc", "кальцинаты"), ("asymmetry", "асимметрия")]
ARM = [("V", "V: LoRA VLM, собственная оценка"), ("VP", "VP: проба на скрытых состояниях V"), ("A_lin", "A_lin: линейная проба, слой 0, база"), ("A_probe", "A_probe: лучшая линейная проба, база"), ("A_tf", "A_tf: трансформерная голова, токены башни"),
       ("D", "D: голова стадии 1 на кэше DINOv2"), ("D_ref", "D_ref: стадия 1, исходный рецепт"), ("V_ctl", "V_ctl: V, 2 295 шагов"), ("V0", "V0: базовая модель"), ("V_blank", "V_blank: V, чёрные изображения")]
f3 = lambda x: "—" if pd.isna(x) else f"{x:.3f}"
out = []
for f, ttl in FIND:
    nps = {s: C[(C.finding == f) & (C["size"] == s)].n_pos_train.mean() for s in SZ}
    out.append(f"\n### AUC на тесте EMBED, {ttl}: среднее по сидам [95% интервал по пациентам теста]\n")
    out.append("| рукав | " + " | ".join(f"{s} ({nps[s]:.0f})" if not np.isnan(nps[s]) else s for s in SZ) + " |"); out.append("|---|" + "---|" * len(SZ))
    for arm, lab in ARM:
        d = C[(C.arm == arm) & (C.finding == f)]
        if d.empty: continue
        cells = []
        for s in SZ:
            r = d[d["size"] == s]
            cells.append("—" if r.empty else f"{r.auc.iloc[0]:.3f} [{r.lo.iloc[0]:.3f}, {r.hi.iloc[0]:.3f}]" + ("" if r.n_seeds.iloc[0] == 3 or s == "f1" and arm in ("D_ref", "V0", "V_blank") else f" (сидов {int(r.n_seeds.iloc[0])})"))
        out.append(f"| {lab} | " + " | ".join(cells) + " |")
    out.append("")
out.append("\n### Стандартное отклонение AUC между сидами (обучение и подвыборка)\n"); out.append("| рукав, находка | " + " | ".join(SZ) + " |"); out.append("|---|" + "---|" * len(SZ))
for arm, lab in ARM[:6]:
    for f, ttl in FIND:
        d = C[(C.arm == arm) & (C.finding == f)]
        if d.empty: continue
        out.append(f"| {arm}, {f} | " + " | ".join(f3(d[d['size'] == s].sd_seeds.iloc[0]) if not d[d['size'] == s].empty else "—" for s in SZ) + " |")
out.append("\n### Подтверждающие контрасты на верхнем размере (f1): разность AUC [95% интервал], вывод по правилам предрегистрации\n")
out.append("| контраст | находка | разность | интервал | граница | вывод |"); out.append("|---|---|---|---|---|---|")
top = K[K["size"].isin(["f1", "f4-f1"])]
for _, r in top.iterrows(): out.append(f"| {r.contrast} | {r.finding} | {r.delta:+.3f} | [{r.lo:+.3f}, {r.hi:+.3f}] | {r.margin:g} | {r.verdict} |")
out.append("\n### Контрасты K1–K3 на всех размерах (описательно)\n"); out.append("| контраст | находка | размер | разность | интервал | вывод |"); out.append("|---|---|---|---|---|---|")
for _, r in K[K.contrast.str.match(r"K[123] ") & ~K["size"].isin(["f1", "f4-f1"])].iterrows(): out.append(f"| {r.contrast} | {r.finding} | {r['size']} | {r.delta:+.3f} | [{r.lo:+.3f}, {r.hi:+.3f}] | {r.verdict} |")
if meta: out.append(f"\nРукав A\\* (лучший из A_probe и A_tf по VS AUC) по размерам: {meta.get('astar')}. Повторов бутстрепа: {meta.get('n_boot')}. Найдены рукава: {', '.join(meta.get('arms_found', []))}.")
open(a.out, "w", encoding="utf-8").write("\n".join(out)); print("wrote", a.out, len(out), "lines")
