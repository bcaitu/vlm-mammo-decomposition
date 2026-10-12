#!/bin/bash
# s6lc phase 2: after the learning-rate pilot (lc_pick_lr.py has written runs_lc/lr_V.txt, lr_D.txt, pilot_choice.json) record the choice and open gate2.
set -e
cd ~/cbmammo; export PYTHONPATH=$PWD:$PWD/analysis_embed
test -f runs_lc/lr_V.txt; test -f runs_lc/lr_D.txt; test -f runs_lc/pilot_choice.json; test -f runs_lc/PREREG_FROZEN; test ! -f runs_lc/LR_FROZEN
~/envs/cbmammo/bin/python - <<'EOF'
import json
c = json.load(open("runs_lc/pilot_choice.json")); lv, ld = open("runs_lc/lr_V.txt").read().strip(), open("runs_lc/lr_D.txt").read().strip()
t = ["# s6lc: приложение фазы 2 к предрегистрации (выбор скорости обучения по пилоту)", "", f"V: выбрана скорость обучения {lv}; VS среднее AUC лучшей контрольной точки по вариантам: " + ", ".join(f"{k}: {v['mean']:.4f} (шаг {v['ckpt']})" for k, v in c["V"].items()) + ".",
     f"D (на кэше): выбрана скорость обучения {ld}; VS среднее AUC по вариантам: " + ", ".join(f"{k}: {v['mean']:.4f}" for k, v in c["D"].items()) + ".", "",
     "Пилотные прогоны размера f16 сида 0 с выбранными значениями используются как основные прогоны этого размера и сида (символические ссылки `runs_lc/V/f16_s0`, `runs_lc/D/f16_s0`)."]
open("prereg/s6lc_phase2_ru.md", "w").write("\n".join(t) + "\n")
EOF
{ echo "# s6lc phase 2 $(date -u +%FT%TZ)"; sha256sum runs_lc/lr_V.txt runs_lc/lr_D.txt runs_lc/pilot_choice.json prereg/s6lc_phase2_ru.md analysis_embed/lc_pick_lr.py runs_lc/tasks_main.json; } > prereg/s6lc_hashes_phase2.txt
touch runs_lc/LR_FROZEN; echo "phase 2 recorded"; cat prereg/s6lc_phase2_ru.md
