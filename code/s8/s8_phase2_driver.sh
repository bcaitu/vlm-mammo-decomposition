#!/bin/bash
# s8: waits for the two lr_vis pilots (selected.json), applies the pre-specified rule and records phase 2 (opens gate 2 = runs_s8/GATE2).
cd ~/cbmammo; export PYTHONPATH=$PWD:$PWD/analysis_embed
echo "driver start $(date)"
while true; do ok=1; for lr in 2e-5 1e-4; do test -f runs_s8/pilot/lrvis$lr/selected.json || ok=0; done; [ $ok = 1 ] && break; sleep 60; done
echo "pilots complete $(date)"
~/envs/cbmammo/bin/python analysis_embed/s8_pick_lrvis.py && bash scripts/s8_freeze2.sh
echo "driver done rc=$? $(date)"
