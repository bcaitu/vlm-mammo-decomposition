#!/bin/bash
# s10db: runs runs_s10db/tasks.json on the given GPUs (one worker per GPU, tasks dealt out round-robin, longest first). usage: bash scripts/s10db_run.sh "1 2 3"
cd ~/cbmammo
GPUS=($1); NW=${#GPUS[@]}
~/envs/cbmammo/bin/python - <<EOF > runs_s10db/tasks_lines.txt
import json
for i, t in enumerate(json.load(open("runs_s10db/tasks.json"))): print(i, t["name"], t["cmd"], sep="\t")
EOF
for ((w=0; w<NW; w++)); do
  ( while IFS=$'\t' read -r i name cmd; do
      if [ $((i % NW)) -eq $w ]; then
        echo "[$(date '+%F %T')] worker $w gpu ${GPUS[$w]} start $name" >> runs_s10db/driver.log
        CUDA_VISIBLE_DEVICES=${GPUS[$w]} bash -c "$cmd" > runs_s10db/logs/$name.log 2>&1 < /dev/null
        rc=$?
        echo "[$(date '+%F %T')] worker $w done $name exit $rc" >> runs_s10db/driver.log
      fi
    done < runs_s10db/tasks_lines.txt ) &
done
wait
echo "[$(date '+%F %T')] all workers finished" >> runs_s10db/driver.log
