#!/bin/bash
# Pipeline of the VLM study V1-V2 (paper 4): waits for the image cache, records hashes, then trains three seeds (train -> generate orig, swap, blank) and runs the zero-shot condition V0.
# usage: tmux new-session -d -s vlmpipe "bash scripts/vlm_pipeline.sh"
cd ~/cbmammo; source ~/envs/cbmammo/bin/activate 2>/dev/null
export PYTHONPATH=$PWD TOKENIZERS_PARALLELISM=false
mkdir -p logs/vlm runs_vlm prereg
while tmux has-session -t vlmcache 2>/dev/null; do sleep 60; done
python3 - <<'EOF'
import json, sys
m = json.load(open('/raid/home/user0/cbmammo_cache/vlm896/cache_meta.json'))
print(m)
if m['errors'] > 0 or m['rows'] < 80000: sys.exit(1)
EOF
if [ $? -ne 0 ]; then echo "cache incomplete or with errors; stopping" > logs/vlm/pipeline_abort.txt; exit 1; fi
{ sha256sum prereg/s4v12_prereg_ru.md analysis_embed/vlm_text.py analysis_embed/vlm_common.py analysis_embed/vlm_train.py analysis_embed/vlm_generate.py analysis_embed/vlm_make_cache.py scripts/vlm_pipeline.sh; cat /raid/home/user0/cbmammo_cache/vlm896/cache_meta.json; echo; echo "recorded $(date -u +%FT%TZ) before launch of training"; } > prereg/s4v12_hashes.txt
# three GPUs with the least used memory among 2-6
GPUS=($(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1>=2 && $1<=6 {print $2" "$1}' | sort -n | head -4 | awk '{print $2}'))
echo "gpus ${GPUS[@]}" >> prereg/s4v12_hashes.txt
for s in 0 1 2; do
  g=${GPUS[$s]}
  tmux new-session -d -s vlm_s$s "cd ~/cbmammo; source ~/envs/cbmammo/bin/activate; export PYTHONPATH=\$PWD TOKENIZERS_PARALLELISM=false; export CUDA_VISIBLE_DEVICES=$g; python3 -u analysis_embed/vlm_train.py --out runs_vlm/v1_s$s --seed $s --workers 8 > logs/vlm/train_s$s.log 2>&1 && python3 -u analysis_embed/vlm_generate.py --adapter runs_vlm/v1_s$s --out runs_vlm/v1_s$s/gen --variants orig,swap,blank > logs/vlm/gen_s$s.log 2>&1; echo exit \$? >> logs/vlm/gen_s$s.log"
  sleep 20
done
g=${GPUS[3]}
tmux new-session -d -s vlm_v0 "cd ~/cbmammo; source ~/envs/cbmammo/bin/activate; export PYTHONPATH=\$PWD TOKENIZERS_PARALLELISM=false; export CUDA_VISIBLE_DEVICES=$g; python3 -u analysis_embed/vlm_generate.py --task zeroshot --out runs_vlm/v0/gen --variants orig > logs/vlm/gen_v0.log 2>&1; echo exit \$? >> logs/vlm/gen_v0.log"
echo "pipeline launched $(date -u +%FT%TZ)" >> logs/vlm/pipeline.log
