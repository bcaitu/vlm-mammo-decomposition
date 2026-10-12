#!/bin/bash
# s8 phase 2: record the chosen lr_vis and the pilot files BEFORE the main VC runs start (opens gate 2).
cd ~/cbmammo
python3 - <<'PY'
import json
c = json.load(open("runs_s8/pilot_choice.json")); p = c["pilot"]
lines = ["# s8: приложение фазы 2 (выбор скорости обучения LoRA башни по пилоту)", "",
 "lr_vis выбрана: %s. VS среднее AUC лучшей контрольной точки: " % c["lr_vis"] + "; ".join("%s: %.4f (шаг %s)" % (k, x["mean"], x["ckpt"]) for k, x in p.items()) + ".",
 "Правило: " + c["rule"] + ".", "Пилотные прогоны (f16, сид 0) остаются описательной точкой VC f16; в подтверждающие контрасты не входят.", ""]
open("prereg/s8_phase2_ru.md", "w", encoding="utf-8").write("\n".join(lines))
PY
sha256sum prereg/s8_phase2_ru.md runs_s8/lr_vis.txt runs_s8/pilot_choice.json runs_s8/pilot/lrvis2e-5/selected.json runs_s8/pilot/lrvis1e-4/selected.json > prereg/s8_hashes_phase2.txt && date '+%F %T %z' > runs_s8/GATE2 && cat prereg/s8_phase2_ru.md
