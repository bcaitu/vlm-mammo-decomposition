#!/bin/bash
# s8: waits for the queue to finish without failures, then (1) integrity pre-check without metrics, (2) hash checks, (3) ONE analysis run, (4) run metadata. Stops without analysis on any violation.
cd ~/cbmammo; source ~/envs/cbmammo/bin/activate; export PYTHONPATH=$PWD:$PWD/analysis_embed
echo "final pipeline start $(date '+%F %T')"
until grep -q 'queue finished; failed: \[\]' runs_s8/queue.log; do
  if grep -q 'queue finished; failed: \[.\+\]' runs_s8/queue.log; then echo "queue finished with failures: stop, no analysis"; exit 2; fi
  sleep 60
done
echo "queue finished $(date '+%F %T')"
python analysis_embed/s8_precheck.py > runs_s8/precheck.log 2>&1 || { echo "precheck FAILED (runs_s8/precheck.log): stop, no analysis"; exit 3; }
echo "precheck pass $(date '+%F %T')"
sha256sum -c prereg/s8_hashes.txt > runs_s8/hashcheck_main.log 2>&1
sha256sum -c prereg/s8_hashes_phase2.txt > runs_s8/hashcheck_phase2.log 2>&1
sha256sum -c prereg/s8_hashes_dev1.txt > runs_s8/hashcheck_dev1.log 2>&1
nm=$(grep -c ': FAILED$' runs_s8/hashcheck_main.log); n2=$(grep -c ': FAILED$' runs_s8/hashcheck_phase2.log); n1=$(grep -c ': FAILED$' runs_s8/hashcheck_dev1.log)
only=$(grep ': FAILED$' runs_s8/hashcheck_main.log | grep -c 'analysis_embed/s8_analysis.py: FAILED')
orig=$(sha256sum analysis_embed/s8_analysis.py.orig_frozen | cut -c1-64); rec=$(grep 'analysis_embed/s8_analysis.py$' prereg/s8_hashes.txt | cut -c1-64)
echo "hash check: main failed=$nm (only s8_analysis.py: $only), phase2 failed=$n2, dev1 failed=$n1, orig==recorded: $([ "$orig" = "$rec" ] && echo yes || echo NO)"
if [ "$nm" != "1" ] || [ "$only" != "1" ] || [ "$n2" != "0" ] || [ "$n1" != "0" ] || [ "$orig" != "$rec" ]; then echo "hash check violated: stop, no analysis"; exit 4; fi
date '+%F %T' > runs_s8/analysis_start.txt
python analysis_embed/s8_analysis.py --out runs_s8/analysis --boot 2000 > runs_s8/analysis.log 2>&1; rc=$?
date '+%F %T' > runs_s8/analysis_end.txt
echo "analysis rc=$rc $(date '+%F %T')"
[ $rc = 0 ] && python analysis_embed/s8_collect_meta.py
echo "final pipeline done $(date '+%F %T')"
