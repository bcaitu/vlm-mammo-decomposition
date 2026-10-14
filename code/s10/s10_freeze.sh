#!/bin/bash
# s10 freeze: SHA-256 of the protocols, scripts, row lists and task list (run once, after smoke tests, before the main runs)
cd ~/cbmammo || exit 1
OUT=prereg/s10_hashes.txt
[ -e $OUT ] && { echo "already frozen: $OUT exists"; exit 1; }
{ ls prereg/s10an_prereg_ru.md prereg/s10db_prereg_ru.md scripts/s10_freeze.sh
  ls analysis_embed/s10_common.py analysis_embed/s10an_analysis.py analysis_embed/s10db_analysis.py analysis_embed/lc_dtrain_budget.py analysis_embed/s10db_make_tasks.py analysis_embed/lc_dtrain.py analysis_embed/lc_analysis.py scripts/s10db_run.sh
  ls runs_s10db/vbudget.csv runs_s10db/tasks.json runs_lc/lr_D.txt prereg/s6lc_env.txt
  ls runs_lc/sets/subset_s*_f*.json runs_lc/sets/val_sample.json runs_lc/sets/test_embed.json runs_lc/sets/test_sub3000.json runs_lc/sets/meta.json
} | sort -u | xargs sha256sum > $OUT
date '+%F %T %z' > prereg/PREREG_FROZEN_s10
wc -l $OUT; cat prereg/PREREG_FROZEN_s10
