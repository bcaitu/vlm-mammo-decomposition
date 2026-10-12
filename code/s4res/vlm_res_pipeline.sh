#!/bin/bash
# Resolution test (paper 4, s4res): caches at 1344 and 1792 px for the development sample, feature extraction with the base model, analysis. Refuses to start without the hash record.
cd ~/cbmammo || exit 1
source ~/envs/cbmammo/bin/activate; export PYTHONPATH=$PWD:$PWD/analysis_embed
test -f prereg/s4res_hashes.txt || { echo "no hash record, stop"; exit 1; }
ROWS=runs_vlm/_diag/dev_rows.json
echo "[$(date -u +%FT%TZ)] caches"
mkdir -p /raid/home/user0/cbmammo_cache/vlm1344 /raid/home/user0/cbmammo_cache/vlm1792
( python3 analysis_embed/vlm_make_cache_rows.py --rows $ROWS --out /raid/home/user0/cbmammo_cache/vlm1344 --side 1344 --workers 24 > logs/vlm/res_cache_1344.log 2>&1 ) &
( python3 analysis_embed/vlm_make_cache_rows.py --rows $ROWS --out /raid/home/user0/cbmammo_cache/vlm1792 --side 1792 --workers 24 > logs/vlm/res_cache_1792.log 2>&1 ) &
wait
for r in 1344 1792; do grep -q "errors 0" logs/vlm/res_cache_$r.log || { echo "cache $r failed"; exit 2; }; done
echo "[$(date -u +%FT%TZ)] extraction"
( CUDA_VISIBLE_DEVICES=2 python3 analysis_embed/vlm_diag_extract_res.py --rows $ROWS --cache /raid/home/user0/cbmammo_cache/vlm1344 --out runs_vlm/_diag/feat_base_1344.npz > logs/vlm/res_ext_1344.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=3 python3 analysis_embed/vlm_diag_extract_res.py --rows $ROWS --cache /raid/home/user0/cbmammo_cache/vlm1792 --out runs_vlm/_diag/feat_base_1792.npz > logs/vlm/res_ext_1792.log 2>&1 ) &
wait
for r in 1344 1792; do test -f runs_vlm/_diag/feat_base_$r.npz || { echo "missing features $r"; exit 3; }; done
echo "[$(date -u +%FT%TZ)] extraction finished"
CUDA_VISIBLE_DEVICES='' python3 analysis_embed/vlm_res_analysis.py --out runs_vlm/_diag/tables_res > logs/vlm/res_an.log 2>&1
echo "[$(date -u +%FT%TZ)] analysis finished"
