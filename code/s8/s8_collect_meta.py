#!/usr/bin/env python3
"""s8: collects run metadata for the report (queue state and retries, training logs, pilot choice, hash checks) into runs_s8/analysis/run_meta.json. No test metrics are read.
usage (cwd ~/cbmammo): python3 analysis_embed/s8_collect_meta.py"""
import glob, json, os, subprocess
out = {}
st = json.load(open("runs_s8/queue_state.json")); out["queue"] = dict(n_tasks=st["n_tasks"], n_done=st["n_done"], failed=st["failed"], pending=len(st["pending"]), time=st["time"])
log = open("runs_s8/queue.log").read().splitlines(); out["queue_first_start"] = next((l for l in log if " START " in l), None); out["queue_last"] = log[-1] if log else None
out["retries"] = [l for l in log if "FAILED" in l and "attempt" in l]
rows = []
for p in sorted(glob.glob("runs_s8/*/f1_s*/train_log.json") + glob.glob("runs_s8/pilot/*/train_log.json")):
    d = json.load(open(p)); lg = d["log"]
    rows.append(dict(run=os.path.dirname(p).replace("runs_s8/", ""), steps=d["steps"], epochs=d["epochs"], seconds=d["seconds"], lr=d["lr"], lr_vis=d.get("lr_vis"), n_trainable=d.get("n_trainable"), n_vision=d.get("n_vision"), loss_first=lg[0]["loss"], loss_last=lg[-1]["loss"]))
out["train"] = rows; out["pilot"] = json.load(open("runs_s8/pilot_choice.json"))
def hc(f):
    r = subprocess.run(["sha256sum", "-c", f], capture_output=True, text=True)
    return dict(ok=len([l for l in r.stdout.splitlines() if l.endswith(": OK")]), failed=[l for l in r.stdout.splitlines() if "FAILED" in l])
out["hash_main"] = hc("prereg/s8_hashes.txt"); out["hash_phase2"] = hc("prereg/s8_hashes_phase2.txt"); out["hash_dev1"] = hc("prereg/s8_hashes_dev1.txt")
sh = lambda cmd: subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
out["orig_frozen_sha"] = sh("sha256sum analysis_embed/s8_analysis.py.orig_frozen | cut -c1-64"); out["recorded_sha"] = sh("grep 'analysis_embed/s8_analysis.py$' prereg/s8_hashes.txt | cut -c1-64")
out["analysis_start"] = sh("cat runs_s8/analysis_start.txt 2>/dev/null"); out["analysis_end"] = sh("cat runs_s8/analysis_end.txt 2>/dev/null")
os.makedirs("runs_s8/analysis", exist_ok=True); json.dump(out, open("runs_s8/analysis/run_meta.json", "w"), indent=1, ensure_ascii=False); print("wrote runs_s8/analysis/run_meta.json", len(rows), "train logs")
