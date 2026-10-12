#!/bin/bash
cd ~/cbmammo
export PYTHONPATH=$PWD:$PWD/analysis_embed PYTHONUNBUFFERED=1
exec ~/envs/cbmammo/bin/python analysis_embed/lc_queue.py --tasks runs_lc/tasks_main.json --gpus 2,3,4,5,6 --cpu_slots 4 --state runs_lc/queue2_state.json --stop runs_lc/queue2.stop >> runs_lc/queue2.log 2>&1
