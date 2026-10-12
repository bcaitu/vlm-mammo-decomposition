#!/bin/bash
# Second VLM study (paper 4, s4v3eval): generation on the fresh rows and analysis. Refuses to start without the hash record.
cd ~/cbmammo || exit 1
source ~/envs/cbmammo/bin/activate; export PYTHONPATH=$PWD:$PWD/analysis_embed
test -f prereg/s4v3eval_hashes.txt || { echo "no hash record, stop"; exit 1; }
mkdir -p logs/vlm runs_vlm/_fresh/dec
ROWS=runs_vlm/_fresh/fresh_rows.json
echo "[$(date -u +%FT%TZ)] cache check"; python3 analysis_embed/vlm_make_cache_rows.py --rows $ROWS --out /raid/home/user0/cbmammo_cache/vlm896 --workers 32 > logs/vlm/fresh_cache2.log 2>&1 || exit 2
echo "[$(date -u +%FT%TZ)] vlm generation start"
for s in 0 1 2; do
  gpu=$((s+2))
  CUDA_VISIBLE_DEVICES=$gpu python3 analysis_embed/vlm_generate_fresh.py --adapter runs_vlm/v1_s$s --rows $ROWS --out runs_vlm/v1_s$s/gen_fresh --variants orig,swap,blank > logs/vlm/fresh_gen_s$s.log 2>&1 &
done
(for r in cb_embed_dinov2_s0 cb_embed_dinov2_s1 cb_embed_dinov2_s2 cb_embed_ft2_s0 cb_embed_ft2_s1 cb_embed_ft2_s2; do
  CUDA_VISIBLE_DEVICES=5 python3 analysis_embed/cb_generate_rows.py --stage1 runs/$r --s2 runs_stage2/grid2_$r --rows $ROWS --out runs_vlm/_fresh/dec/$r >> logs/vlm/fresh_dec_cb.log 2>&1; done) &
(for r in opq_embed_dinov2_s0 opq_embed_dinov2_s1 opq_embed_dinov2_s2 opq_embed_ft2_s0 opq_embed_ft2_s1 opq_embed_ft2_s2; do
  CUDA_VISIBLE_DEVICES=6 python3 analysis_embed/cb_generate_rows.py --stage1 runs/$r --s2 runs_stage2/grid2_$r --rows $ROWS --out runs_vlm/_fresh/dec/$r >> logs/vlm/fresh_dec_opq.log 2>&1; done) &
wait
echo "[$(date -u +%FT%TZ)] generation finished"
for s in 0 1 2; do grep -q finished logs/vlm/fresh_gen_s$s.log || { echo "vlm seed $s did not finish"; exit 3; }; done
CUDA_VISIBLE_DEVICES='' python3 analysis_embed/vlm_fresh_analysis.py --mode fresh --out runs_vlm/_fresh/tables --boot 2000 > logs/vlm/fresh_an.log 2>&1
CUDA_VISIBLE_DEVICES='' python3 analysis_embed/vlm_fresh_analysis.py --mode seen --out runs_vlm/_fresh/tables_seen --boot 2000 > logs/vlm/fresh_an_seen.log 2>&1
echo "[$(date -u +%FT%TZ)] analysis finished"
