#!/bin/bash
cd ~/cbmammo
export PYTHONPATH=$PWD:$PWD/analysis_embed PYTHONUNBUFFERED=1
exec ~/envs/cbmammo/bin/python analysis_embed/lc_queue.py --tasks runs_s8/tasks.json --gpus 2,3,4,5,6 --cpu_slots 4 --state runs_s8/queue_state.json --stop runs_s8/queue.stop >> runs_s8/queue.log 2>&1
