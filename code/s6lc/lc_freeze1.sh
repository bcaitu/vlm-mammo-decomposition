#!/bin/bash
# s6lc phase 1: hash the pre-registration, code, settings and row lists; then open gate1 (runs that do not depend on the learning rates).
set -e
cd ~/cbmammo; export PYTHONPATH=$PWD:$PWD/analysis_embed
test -f prereg/s6lc_prereg_ru.md; test ! -f runs_lc/PREREG_FROZEN
~/envs/cbmammo/bin/python analysis_embed/lc_make_tasks.py --stage main --out runs_lc/tasks_main.json
{ ~/envs/cbmammo/bin/python - <<'EOF'
import torch, transformers, peft, sklearn, numpy, scipy, sys
print("python", sys.version.split()[0], "torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__, "sklearn", sklearn.__version__, "numpy", numpy.__version__, "scipy", scipy.__version__)
EOF
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader | sed -n '3,7p'; echo "cgroup cpu.max (account quota): $(cat /sys/fs/cgroup/user.slice/user-1001.slice/cpu.max)"; } > prereg/s6lc_env.txt 2>&1
{ echo "# s6lc phase 1 $(date -u +%FT%TZ)"; sha256sum analysis_embed/lc_*.py analysis_embed/vlm_common.py analysis_embed/vlm_text.py cbmammo/train_stage1.py cbmammo/model.py cbmammo/data.py cbmammo/concepts.py configs/cb_embed_dinov2.yaml runs_lc/sets/*.json runs_lc/sets/sets_summary.csv runs_lc/tasks_main.json prereg/s6lc_prereg_ru.md prereg/s6lc_env.txt; } > prereg/s6lc_hashes.txt
echo "phase 1 hashes recorded: $(grep -vc '^#' prereg/s6lc_hashes.txt) files"
touch runs_lc/PREREG_FROZEN
