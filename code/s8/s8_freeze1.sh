#!/bin/bash
# s8 phase 1: record sha256 of the pre-registration, scripts, row lists and task file BEFORE the main runs (pilot excluded by design).
cd ~/cbmammo
F="prereg/s8_prereg_ru.md prereg/s8_smoke.txt analysis_embed/s8_train.py analysis_embed/s8_analysis.py analysis_embed/s8_pick_lrvis.py analysis_embed/lc_score.py analysis_embed/lc_probe.py analysis_embed/lc_pick.py analysis_embed/lc_queue.py analysis_embed/vlm_common.py analysis_embed/vlm_text.py scripts/s8_freeze1.sh scripts/s8_freeze2.sh scripts/s8_phase2_driver.sh runs_s8/tasks.json runs_lc/sets/meta.json runs_lc/sets/val_sample.json runs_lc/sets/test_embed.json runs_lc/sets/subset_s0_f16.json"
for s in 0 1 2; do F="$F runs_lc/sets/subset_s${s}_f1.json runs_lc/sets/ext_s${s}_f1.json"; done
sha256sum $F > prereg/s8_hashes.txt && date '+%F %T %z' > runs_s8/PREREG_FROZEN && wc -l prereg/s8_hashes.txt
