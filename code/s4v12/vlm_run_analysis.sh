#!/bin/bash
cd ~/cbmammo; source ~/envs/cbmammo/bin/activate 2>/dev/null; export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=''
while [ $(grep -l '^exit' logs/vlm/gen_s0.log logs/vlm/gen_s1.log logs/vlm/gen_s2.log 2>/dev/null | wc -l) -lt 3 ]; do sleep 60; done
python3 -u analysis_embed/vlm_analysis.py --out runs_vlm/_v12_tables > logs/vlm/analysis.log 2>&1; echo "exit $?" >> logs/vlm/analysis.log
