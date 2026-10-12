#!/bin/bash
# s6lc: waits for the learning-rate pilots (V: vsc_pilot selected.json; D: run.json), then applies the pre-specified selection rule and records phase 2 (opens gate 2).
cd ~/cbmammo; export PYTHONPATH=$PWD:$PWD/analysis_embed
echo "driver start $(date)"
while true; do
  ok=1
  for lr in 5e-5 1e-4 2e-4; do test -f runs_lc/V/f16_s0_lr$lr/selected.json || ok=0; done
  for dlr in 1e-4 3e-4 1e-3; do test -f runs_lc/D/f16_s0_lr$dlr/run.json || ok=0; done
  [ $ok = 1 ] && break
  sleep 60
done
echo "pilots complete $(date)"
~/envs/cbmammo/bin/python analysis_embed/lc_pick_lr.py && bash scripts/lc_freeze2.sh
echo "driver done rc=$? $(date)"
