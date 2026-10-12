#!/bin/bash
# Diagnostic (paper 4, s4diag): cache, greedy generation, feature extraction, analysis. Refuses to start without the hash record.
cd ~/cbmammo || exit 1
source ~/envs/cbmammo/bin/activate; export PYTHONPATH=$PWD:$PWD/analysis_embed
test -f prereg/s4diag_hashes.txt || { echo "no hash record, stop"; exit 1; }
ROWS=runs_vlm/_diag/dev_rows.json
echo "[$(date -u +%FT%TZ)] cache"; python3 analysis_embed/vlm_make_cache_rows.py --rows $ROWS --out /raid/home/user0/cbmammo_cache/vlm896 --workers 32 > logs/vlm/diag_cache.log 2>&1 || exit 2
echo "[$(date -u +%FT%TZ)] generation and base features"
for s in 0 1 2; do
  ( CUDA_VISIBLE_DEVICES=$((s+2)) python3 analysis_embed/vlm_generate_fresh.py --adapter runs_vlm/v1_s$s --rows $ROWS --out runs_vlm/v1_s$s/gen_diag --variants orig > logs/vlm/diag_gen_s$s.log 2>&1 && \
    CUDA_VISIBLE_DEVICES=$((s+2)) python3 analysis_embed/vlm_diag_extract.py --adapter runs_vlm/v1_s$s --rows $ROWS --gen runs_vlm/v1_s$s/gen_diag/gen_dev_embed_val_orig.jsonl --out runs_vlm/_diag/feat_v1_s$s.npz > logs/vlm/diag_ext_s$s.log 2>&1 ) &
done
( CUDA_VISIBLE_DEVICES=5 python3 analysis_embed/vlm_diag_extract.py --rows $ROWS --out runs_vlm/_diag/feat_base.npz > logs/vlm/diag_ext_base.log 2>&1 ) &
wait
for t in v1_s0 v1_s1 v1_s2 base; do test -f runs_vlm/_diag/feat_$t.npz || { echo "missing features $t"; exit 3; }; done
echo "[$(date -u +%FT%TZ)] extraction finished"
CUDA_VISIBLE_DEVICES='' python3 analysis_embed/vlm_diag_analysis.py --out runs_vlm/_diag/tables > logs/vlm/diag_an.log 2>&1
echo "[$(date -u +%FT%TZ)] analysis finished"
