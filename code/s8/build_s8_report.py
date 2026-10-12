#!/usr/bin/env python3
"""s8: assembles paper4_s8_ru.md from the analysis outputs (numbers are read from the files, not typed).
Inputs in --dir: s8_auc.csv, s8_auc_by_run.csv, s8_contrasts.csv, s8_meta.json, run_meta.json, s8_reading.md; optional s8_interpretation_ru.md (written by hand after reading the results; inserted verbatim as section 4).
usage: python3 build_s8_report.py --dir analysis --fig1 <artifact_id> --fig2 <artifact_id> --out paper4_s8_ru.md"""
import argparse, json, os, re, datetime as dt
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--fig1", required=True); ap.add_argument("--fig2", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
D = a.dir; A = pd.read_csv(f"{D}/s8_auc.csv"); B = pd.read_csv(f"{D}/s8_auc_by_run.csv"); K = pd.read_csv(f"{D}/s8_contrasts.csv"); M = json.load(open(f"{D}/run_meta.json")); AM = json.load(open(f"{D}/s8_meta.json"))
READ = open(f"{D}/s8_reading.md", encoding="utf-8").read().split("\n", 1)[1]
INTERP = open(f"{D}/s8_interpretation_ru.md", encoding="utf-8").read() if os.path.exists(f"{D}/s8_interpretation_ru.md") else "ИНТЕРПРЕТАЦИЯ НЕ ДОБАВЛЕНА (файл s8_interpretation_ru.md отсутствует)."
Mfig = lambda aid: "{" + "{artifact:art_" + aid + "}" + "}"
def fa(x, d=3): return f"{x:.{d}f}".replace(".", ",")
def fs(x, d=3): return f"{x:+.{d}f}".replace(".", ",").replace("-", "\u2212")
def cur(arm, f): return A[(A.arm == arm) & (A.finding == f)].iloc[0]
def cell(arm, f): r = cur(arm, f); return f"{fa(r.auc)} [{fa(r.lo)}; {fa(r.hi)}]"
def con(prefix, f):
    d = K[K.contrast.str.startswith(prefix) & (K.finding == f)]; assert len(d) >= 1, (prefix, f); return d.iloc[0]
VERD = [(r"^adds \(lower bound > 0\) and point >= 0\.05$", "добавляет (нижняя граница > 0) и точечная оценка ≥ 0,05"), (r"^adds \(lower bound > 0\)$", "добавляет (нижняя граница > 0)"), (r"^equivalent within \+-([\d.]+)$", None),
        (r"^reduces \(upper bound < 0\)$", "снижает (верхняя граница < 0)"), (r"^unclear$", "неясно"), (r"^below \(upper bound < 0\)$", "ниже (верхняя граница < 0)"), (r"^not below \(lower bound > -([\d.]+)\)$", None)]
def ruv(v):
    v = str(v)
    for pat, rep in VERD:
        m = re.match(pat, v)
        if m:
            if rep: return rep
            if v.startswith("equivalent"): return f"эквивалентно в пределах ±{m.group(1).replace('.', ',')}"
            return f"не ниже (нижняя граница > −{m.group(1).replace('.', ',')})"
    return v
FR = {"calc": "кальцинаты", "asymmetry": "асимметрия", "mass": "масса (контроль)"}
LAB = {"V": "V: LoRA VLM, языковая часть (s6lc)", "VP": "VP: проба на состояниях V (s6lc)", "VC": "VC: LoRA VLM, языковая часть и башня", "VCP": "VCP: проба на состояниях VC", "V2x": "V2x: V при удвоенном бюджете шагов (9 182)",
       "D": "D: голова стадии 1 на DINOv2 (s6lc)", "A_probe": "A_probe: лучшая линейная проба на замороженной башне (s6lc)", "A_tf": "A_tf: трансформерная голова на токенах башни (s6lc)"}
SNAME = {"S1": "S1: VC − V (вклад башни)", "S2": "S2: VC − D (разрыв с головой стадии 1)", "S3": "S3: V2x − V (вклад бюджета шагов)", "S4": "S4: VCP − VP (линейно доступная информация)", "S5": "S5: VC − VCP (потеря на интерфейсе)",
         "VC - V2x": "VC − V2x", "VCP - A_probe": "VCP − A_probe", "VCP - A_tf": "VCP − A_tf", "V2x - D": "V2x − D"}
def sname(c):
    for k, v in SNAME.items():
        if c.startswith(k): return v
    return c
def comma(t): return re.sub(r"(?<=\d)\.(?=\d)", ",", t)
def fi(x): return f"{int(x):,}".replace(",", " ")
top = ["| рукав | " + " | ".join(FR[f] for f in ("calc", "asymmetry", "mass")) + " |", "|---|---|---|---|"]
for arm in ("VC", "V2x", "V", "VCP", "VP", "D", "A_tf", "A_probe"): top.append(f"| {LAB[arm]} | " + " | ".join(cell(arm, f) for f in ("calc", "asymmetry", "mass")) + " |")
conf = ["| контраст | находка | разность [95 % интервал] | вывод по правилам предрегистрации |", "|---|---|---|---|"]
for p in ("S1", "S3", "S4", "S2"):
    for f in ("calc", "asymmetry", "mass"):
        r = con(p, f); conf.append(f"| {sname(r.contrast)} | {FR[f]}{'' if r.kind == 'confirmatory' else ' (описательно)'} | {fs(r.delta)} [{fs(r.lo)}; {fs(r.hi)}] | {ruv(r.verdict)} |")
desc = ["| контраст | находка | разность [95 % интервал] | вывод |", "|---|---|---|---|"]
for p in ("S5", "VC - V2x", "VCP - A_probe", "VCP - A_tf", "V2x - D"):
    for f in ("calc", "asymmetry", "mass"):
        r = con(p, f); desc.append(f"| {sname(r.contrast)} | {FR[f]} | {fs(r.delta)} [{fs(r.lo)}; {fs(r.hi)}] | {ruv(r.verdict)} |")
import hashlib
def sha16(pth): return hashlib.sha256(open(pth, "rb").read()).hexdigest()[:16]
_gain = K[K.contrast.str.startswith(("S1", "S3", "S4", "VC - V2x", "VCP - A_probe", "VCP - A_tf"))]
_aff = _gain[(_gain.hi < 0) & (_gain.lo > -0.03)]
DEV1_TXT = (f"Из {len(_gain)} вердиктов по правилу прироста затронуты {len(_aff)}: " + "; ".join(f"{r.contrast.split(' (')[0]}, {FR[r.finding]}: {fs(r.delta)} [{fs(r.lo)}; {fs(r.hi)}], по предрегистрации «эквивалентно», по исходному коду было бы «снижает»" for r in _aff.itertuples()) + ".") if len(_aff) else "Ни один из вердиктов по правилу прироста не затронут."
PH2 = pd.read_csv(f"{D}/s8_posthoc2_gap_to_D.csv")
PH2_DET = all(sha16(f"{D}/{fn}") == sha16(f"{D}/posthoc2_dup/{fn}") for fn in ("s8_auc.csv", "s8_auc_by_run.csv"))
PH2_HASH = open(f"{D}/s8_hashes_posthoc2.txt", encoding="utf-8").read().splitlines()[1].split()[0][:16]
PH2_TIME = open(f"{D}/s8_hashes_posthoc2.txt", encoding="utf-8").read().splitlines()[0].split(" (")[0]
PH = pd.read_csv(f"{D}/s8_posthoc.csv"); PHV = pd.read_csv(f"{D}/s8_posthoc_vctl_auc.csv")
PH_DET = all(sha16(f"{D}/{fn}") == sha16(f"{D}/posthoc_dup/{fn}") for fn in ("s8_auc.csv", "s8_auc_by_run.csv"))
PH_HASH = [ln.split()[0][:16] for ln in open(f"{D}/s8_hashes_posthoc.txt", encoding="utf-8").read().splitlines()[1:3]]
PH_TIME = open(f"{D}/s8_hashes_posthoc.txt", encoding="utf-8").read().splitlines()[0].split(" (")[0]
PHN = {"P1": "P1: VP − D (проба на состояниях V против головы стадии 1)", "P2": "P2: VCP − D (проба на состояниях VC против головы стадии 1)",
       "P3": "P3: V − V_ctl (первое удвоение шагов, 2 295 → 4 591)", "P4": "P4: V2x − V (второе удвоение, 4 591 → 9 182; то же, что S3)",
       "P5": "P5: V2x − V_ctl (два удвоения)", "P6": "P6: (V2x − V) − (V − V_ctl) (второе удвоение минус первое)"}
ph_rows = ["| контраст | находка | разность [95 % интервал] | интервал |", "|---|---|---|---|"]
for k in ("P1", "P2", "P3", "P4", "P5", "P6"):
    for f in ("calc", "asymmetry", "mass"):
        r = PH[(PH.contrast.str.startswith(k + " ")) & (PH.finding == f)].iloc[0]
        ph_rows.append(f"| {PHN[k]} | {FR[f]} | {fs(r.delta)} [{fs(r.lo)}; {fs(r.hi)}] | {'не включает 0' if r.excludes_zero else 'включает 0'} |")
def ph(k, f):
    r = PH[(PH.contrast.str.startswith(k + " ")) & (PH.finding == f)].iloc[0]; return f"{fs(r.delta)} [{fs(r.lo)}; {fs(r.hi)}]"
ARMLAB = {"A_probe": "A_probe: линейная проба, замороженная башня", "A_tf": "A_tf: трансформерная голова, замороженная башня", "VP": "VP: проба на состояниях V", "VCP": "VCP: проба на состояниях VC", "V_ctl": "V_ctl: ответ, 2 295 шагов", "V": "V: ответ, 4 591 шаг", "V2x": "V2x: ответ, 9 182 шага", "VC": "VC: ответ, башня и язык, 4 591 шаг"}
g_rows = ["| рукав | кальцинаты | асимметрия | масса (контроль) |", "|---|---|---|---|"]
for arm_ in ("A_probe", "A_tf", "VP", "VCP", "V_ctl", "V", "V2x", "VC"):
    cells_ = []
    for f_ in ("calc", "asymmetry", "mass"):
        r_ = PH2[(PH2.arm == arm_) & (PH2.finding == f_)].iloc[0]; cells_.append(f"{fs(r_.delta_vs_D)} [{fs(r_.lo)}; {fs(r_.hi)}]")
    g_rows.append(f"| {ARMLAB[arm_]} | " + " | ".join(cells_) + " |")
vc_row = "; ".join(f"{FR[f]} {fa(r.auc)} [{fa(r.lo)}; {fa(r.hi)}]" for f in ("calc", "asymmetry", "mass") for r in [PHV[PHV.finding == f].iloc[0]])
seeds = ["| рукав | находка | сид 0 | сид 1 | сид 2 | SD между сидами |", "|---|---|---|---|---|---|"]
for arm in ("VC", "V2x", "V", "VCP", "VP", "D"):
    for f in ("calc", "asymmetry", "mass"):
        b = B[(B.arm == arm) & (B.finding == f)].sort_values("seed"); r = cur(arm, f)
        seeds.append(f"| {arm} | {FR[f]} | " + " | ".join(fa(x) for x in b.auc) + " | " * (3 - len(b)) + f" | {fa(r.sd_seeds)} |")
tr = M["train"]; tv = pd.DataFrame(tr)
def hrs(run_prefix): d = tv[tv.run.str.startswith(run_prefix)]; return d.seconds.sum() / 3600, len(d), d.seconds.mean() / 3600
vc_h, vc_n, vc_m = hrs("VC/"); v2_h, v2_n, v2_m = hrs("V2x/"); pil_h, pil_n, _ = hrs("pilot/")
hm, h2, h1 = M["hash_main"], M["hash_phase2"], M["hash_dev1"]; bad = [x.split(":")[0] for x in hm["failed"]]
ret = M["retries"]; pc = M["pilot"]["pilot"]
def cc_(p, f): r = con(p, f); return f"{p} {fs(r.delta)} [{fs(r.lo)}; {fs(r.hi)}]: {ruv(r.verdict)}"
def rows_for(f_ru):
    m = re.search(r"#### " + re.escape(f_ru) + r".*?Строки таблицы, которые выполнены:\n(.*?)(?=\n#### |\Z)", READ.replace("## ", "#### "), re.S)
    return [x.strip("- ").strip() for x in m.group(1).strip().splitlines() if x.strip()] if m else []
rd_calc = rows_for("кальцинаты (подтверждающая)"); rd_asym = rows_for("асимметрия (подтверждающая)")
max_gain = max(float(con(p, f).delta) for p in ("S1", "S3") for f in ("calc", "asymmetry"))
gaps = [float(con(p, f).delta) for p in ("S2", "V2x - D") for f in ("calc", "asymmetry")]
T = []
T.append(f"""# s8: дообучение зрительной башни и бюджет шагов (статья 4, последнее исследование). Итоговый отчёт

Дата отчёта: {dt.date.today().isoformat()}. Предрегистрация: `s8_prereg_ru.md` (хэши фазы 1 записаны до запуска основных прогонов, фазы 2 — до запуска VC; отклонение 1 описано в разделе 1). План, код и отчёт подготовлены с помощью Claude Science (claude-fable-5-1); учёт в `ai_use_log.csv`. Сравнение касается обнаружения находок (AUC по метке железы), не качества текста отчёта.

## Кратко

1. Расчёт завершён полностью: {M['queue']['n_done']} из {M['queue']['n_tasks']} задач, сбоев очереди {len(M['queue']['failed'])}; первых попыток, завершившихся ошибкой и перезапущенных: {len(ret)} (исполнительное замечание, раздел 1). Хэши: основной файл {hm['ok']} OK и {len(hm['failed'])} ожидаемое расхождение ({', '.join(bad) if bad else '—'}; отклонение 1), фаза 2 {h2['ok']} OK, исправленный скрипт {h1['ok']} OK. Итоговый анализ запущен один раз; до него метрики теста не вычислялись и не просматривались.
2. AUC на полном объёме (тест EMBED, среднее по сидам [95 % интервал по пациентам теста]), кальцинаты / асимметрия: VC {cell('VC', 'calc')} / {cell('VC', 'asymmetry')}; V {cell('V', 'calc')} / {cell('V', 'asymmetry')}; V2x {cell('V2x', 'calc')} / {cell('V2x', 'asymmetry')}; голова D {cell('D', 'calc')} / {cell('D', 'asymmetry')}.
3. Подтверждающие контрасты. Кальцинаты: {cc_('S1', 'calc')} (вклад башни); {cc_('S2', 'calc')} (разрыв с D); {cc_('S3', 'calc')} (бюджет шагов); {cc_('S4', 'calc')}. Асимметрия: {cc_('S1', 'asymmetry')}; {cc_('S2', 'asymmetry')}; {cc_('S3', 'asymmetry')}; {cc_('S4', 'asymmetry')}. Точечные оценки S1 и S3 по обеим находкам не достигают порога «практически значимо» 0,05 (наибольшая {fs(max_gain)}). Разрыв с головой D остаётся после каждой из правок по отдельности: у VC (только башня) и у V2x (только удвоенные шаги) от {fs(min(gaps))} до {fs(max(gaps))} AUC (кальцинаты и асимметрия, S2 и описательный контраст V2x − D, раздел 3.4); рукав с обеими правками не запускался.
4. Чтение по таблице исходов предрегистрации (раздел 6), механически (`s8_reading.py`, раздел 3.3). Кальцинаты: {' '.join(rd_calc) if rd_calc else 'ни одна строка не выполнена'} Асимметрия: {' '.join(rd_asym) if rd_asym else 'ни одна строка не выполнена'}
5. Постхок (раздел 3.6; не предрегистрировано). Проба на состояниях VC против головы D (P2): кальцинаты {ph('P2', 'calc')}, асимметрия {ph('P2', 'asymmetry')}. Отдача от шагов, второе удвоение минус первое (P6): кальцинаты {ph('P6', 'calc')}, асимметрия {ph('P6', 'asymmetry')}, масса {ph('P6', 'mass')}.
6. Интерпретация автора анализа и ограничения — раздел 4.

## 1. Целостность прогона

- **Очередь:** {M['queue']['n_done']} из {M['queue']['n_tasks']} задач; сбоев {len(M['queue']['failed'])}. Старт основной очереди: {(M.get('queue_first_start') or '')[:8]} (2026-10-08, UTC+5); последняя запись очереди: `{M.get('queue_last')}`.
- **Время обучений** (по журналам `train_log.json`): VC {vc_n} прогона по {fa(vc_m, 1)} ч в среднем (сумма {fa(vc_h, 1)} ч); V2x {v2_n} прогона по {fa(v2_m, 1)} ч (сумма {fa(v2_h, 1)} ч); пилот {pil_n} прогона (сумма {fa(pil_h, 1)} ч). Все прогоны прошли заданное число шагов (4 591 у VC, 9 182 у V2x, 2 295 у пилота); обучаемых параметров у VC {fi(tv[tv.run.str.startswith('VC/')].n_trainable.iloc[0])} (из них в башне {fi(tv[tv.run.str.startswith('VC/')].n_vision.iloc[0])}), у V2x {fi(tv[tv.run.str.startswith('V2x/')].n_trainable.iloc[0])}.
- **Пилот скорости обучения башни** (f16, сид 0, VS среднее AUC лучшей контрольной точки): 2e-5 — {fa(pc['2e-5']['mean'], 4)} (шаг {pc['2e-5']['ckpt']}), 1e-4 — {fa(pc['1e-4']['mean'], 4)} (шаг {pc['1e-4']['ckpt']}); выбрано {M['pilot']['lr_vis']} по правилу (1e-4 только при выигрыше не менее 0,005).
- **Хэши:** `sha256sum -c prereg/s8_hashes.txt`: {hm['ok']} OK, расхождений {len(hm['failed'])} ({', '.join(bad) if bad else '—'}); `prereg/s8_hashes_phase2.txt`: {h2['ok']} OK; `prereg/s8_hashes_dev1.txt`: {h1['ok']} OK; хэш сохранённого замороженного скрипта анализа {'равен' if M['orig_frozen_sha'] == M['recorded_sha'] else 'НЕ РАВЕН'} записи в основном файле ({M['recorded_sha'][:16]}).
- **Отклонение 1** (`prereg/s8_deviations.md`): в скрипте анализа порядок проверок вердиктов «эквивалентно» и «снижает» приведён к порядку из предрегистрации (раздел 5); до правки анализ не запускался и метрики теста не просматривались; меняется только вердикт интервала целиком внутри (−0,03; 0). {DEV1_TXT}
- **Исполнительное замечание:** первая попытка проб VCP для {len(ret)} прогонов завершилась кодом 1 с ошибкой `BadZipFile` (очередь запускает пробу, как только файл признаков появился, а запись архива ещё не закончена) и сразу перезапускалась автоматически после окончания записи; перезапуск прошёл успешно. Записи очереди: {'; '.join(r.strip() for r in ret) if ret else '—'}.
- **Анализ:** один запуск `s8_analysis.py --out runs_s8/analysis --boot 2000`: {M.get('analysis_start') or '—'} — {M.get('analysis_end') or '—'}; найдены рукава: {', '.join(AM['arms_found'])}; сидов: {', '.join(f'{k} {v}' for k, v in AM['seeds_per_arm'].items())}. Бутстреп по пациентам теста, seed 20261007, {AM['n_boot']} повторов (те же выборки пациентов, что в s6lc).
- **Постхок-анализ** (не предрегистрирован, раздел 3.6): `s8_posthoc.py` (sha256 {PH_HASH[0]}…) записан в `prereg/s8_hashes_posthoc.txt` {PH_TIME} (UTC+5) до запуска, после основного анализа; один запуск; те же выборки пациентов и загрузка данных (исходный текст `s8_analysis.py` выполняется до определения контрастов); дублирующие выходы `s8_auc.csv` и `s8_auc_by_run.csv` {'совпадают побайтно' if PH_DET else 'НЕ СОВПАДАЮТ'} с основными (проверка воспроизводимости).
- **Постхок-анализ 2** (не предрегистрирован, раздел 3.7): `s8_posthoc2.py` (sha256 {PH2_HASH}…) записан в `prereg/s8_hashes_posthoc2.txt` {PH2_TIME} (UTC+5) до запуска; один запуск; только разности «рукав − D» для рисунка рукописи; дублирующие `s8_auc.csv` и `s8_auc_by_run.csv` {'совпадают побайтно' if PH2_DET else 'НЕ СОВПАДАЮТ'} с основными; контрасты V − D, VC − D, V2x − D, VP − D, VCP − D совпали с ранее вычисленными (K3 из s6lc, S2, описательный, P1, P2).

## 2. Что сравнивалось

Вопрос: ограничивает ли VLM замороженная зрительная башня (VC) и бюджет обучения (V2x) в рецепте s6lc. **VC** — Qwen3-VL-8B с LoRA r=16, α=32 на линейных слоях языковой части (как V) и зрительной башни (attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2 во всех 27 блоках, мерджеры), 896 px, шаблонный отчёт, 4 591 шаг, финальный адаптер; **VCP** — линейная проба на скрытых состояниях VC; **V2x** — рецепт V (LoRA только на языковой части) при 9 182 шагах. V, VP, D, A_probe, A_tf — готовые результаты s6lc на полном объёме. Контрасты: S1 = VC − V; S2 = VC − D; S3 = V2x − V; S4 = VCP − VP (границы эквивалентности ±0,03; 0,05 — порог «практически значимо» по точечной оценке; кальцинаты и асимметрия подтверждающие, масса описательно). Тест: весь EMBED test, 13 178 желёз, 2 743 пациента.

## 3. Результаты

### 3.1 Полный объём обучения (f1), AUC на тесте EMBED, среднее по трём сидам [95 % интервал по пациентам теста]

""" + comma("\n".join(top)) + f"""

A_probe детерминирована (один прогон на трёх сидах).

![Рисунок 1]({Mfig(a.fig1)})

**Рисунок 1.** AUC на тесте EMBED на полном объёме: точка — среднее по трём сидам, линия — 95 % интервал по пациентам теста (без разброса между сидами), кружки — отдельные сиды. Разделители: считывания с замороженных признаков (три верхние строки), пробы на состояниях VLM (две следующие), собственные оценки VLM (три нижние).

### 3.2 Контрасты

""" + comma("\n".join(conf)) + f"""

![Рисунок 2]({Mfig(a.fig2)})

**Рисунок 2.** Контрасты S1–S4 (разность AUC, 95 % интервал по пациентам теста). Серая полоса — ±0,03 (граница эквивалентности предрегистрации), пунктир — 0,05 (порог «практически значимо» по точечной оценке, только для групп S1, S3, S4). Заливка: подтверждающие находки; светлые точки: масса (контроль, описательно).

### 3.3 Чтение исходов по таблице предрегистрации (раздел 6), механически

""" + comma(re.sub(r"(?<![\w\d])-(?=\d)", "\u2212", READ.replace("## ", "#### "))) + f"""

### 3.4 Описательные контрасты

""" + comma("\n".join(desc)) + f"""

### 3.5 Разброс между сидами (одиночные AUC)

""" + comma("\n".join(seeds)) + f"""

### 3.6 Постхок-контрасты (не предрегистрированы; написаны и запущены после основного анализа)

Назначение: где остаётся разрыв с головой D (представление или считывание) и как меняется отдача от шагов. Интервалы те же (бутстреп по пациентам теста, 2 000 повторов, seed 20261007); столбец «интервал» показывает только, включает ли он 0; поправок на множественные сравнения нет. V_ctl (s6lc, 2 295 шагов, три сида) в этом разделе используется только как нижняя точка шкалы шагов: AUC {vc_row}. Группы P3 и P4 получены при разных расписаниях обучения.

""" + comma("\n".join(ph_rows)) + f"""

### 3.7 Расстояние до головы стадии 1 (постхок 2, не предрегистрировано)

Разности «рукав − D» по всем рукавам одним запуском (те же выборки пациентов); служат для сводного рисунка рукописи. Интервалы без поправок.

""" + comma("\n".join(g_rows)) + f"""

## 4. Интерпретация и ограничения (автор анализа; не предрегистрировано)

""" + INTERP + f"""

## Приложение. Файлы

`s8_auc.csv`, `s8_auc_by_run.csv`, `s8_contrasts.csv`, `s8_meta.json`, `run_meta.json` — выходы `s8_analysis.py` и `s8_collect_meta.py`; `s8_reading.md` — механическое чтение; код: `s8_train.py`, `s8_analysis.py` (исправленный) и `s8_analysis.py.orig_frozen` (замороженный), `s8_pick_lrvis.py`, `s8_make_tasks.py`, `s8_collect_meta.py`, `s8_posthoc.py`, `s8_posthoc2.py`, `s8_reading.py`, `s8_figure.py`, `build_s8_report.py`. Данные уровня строки (идентификаторы EMBED) остаются на сервере.
""")
open(a.out, "w", encoding="utf-8").write("\n".join(T)); print("wrote", a.out, sum(len(t) for t in T), "chars")
