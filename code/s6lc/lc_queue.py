#!/usr/bin/env python3
"""s6lc job queue: runs shell tasks, one per free GPU, respecting dependencies; idempotent (a task whose 'done' files exist is skipped); one retry on failure.
tasks.json = list of {"name", "cmd", "needs": [names], "done": [files], "gpu": true/false, "prio": number (higher first)}.
usage: lc_queue.py --tasks runs_lc/tasks_main.json --gpus 2,3,4,5,6 [--cpu_slots 6]; create runs_lc/queue.stop to stop launching new tasks."""
import argparse, json, os, subprocess, time
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--gpus", default="2,3,4,5,6"); ap.add_argument("--cpu_slots", type=int, default=6); ap.add_argument("--state", default="runs_lc/queue_state.json"); ap.add_argument("--stop", default="runs_lc/queue.stop")
a = ap.parse_args(); gpus = [int(x) for x in a.gpus.split(",")]
tasks = {t["name"]: t for t in json.load(open(a.tasks))}; os.makedirs("runs_lc/logs", exist_ok=True)
def finished(t): return all(os.path.exists(f) for f in t["done"])
def waited(t): return all(os.path.exists(f) for f in t.get("wait", []))
def gpu_free(g):
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(g), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip()
        return int(out) < 3000
    except Exception: return False
running, failed, attempts, t_start = {}, set(), {}, {}
env0 = dict(os.environ, PYTHONPATH=f"{os.getcwd()}:{os.getcwd()}/analysis_embed", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
def write_state():
    done = [n for n, t in tasks.items() if finished(t)]
    json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), n_tasks=len(tasks), n_done=len(done), running={n: dict(gpu=g, min=round((time.time() - t_start[n]) / 60, 1)) for n, (p, g) in running.items()},
                   failed=sorted(failed), pending=[n for n in tasks if n not in done and n not in running and n not in failed]), open(a.state, "w"), indent=1)
while True:
    for n, (p, g) in list(running.items()):
        rc = p.poll()
        if rc is None: continue
        del running[n]
        if rc == 0 and finished(tasks[n]): print(time.strftime("%H:%M:%S"), "DONE", n, flush=True)
        else:
            attempts[n] = attempts.get(n, 0) + 1; print(time.strftime("%H:%M:%S"), "FAILED", n, "rc", rc, "attempt", attempts[n], flush=True)
            if attempts[n] >= 2: failed.add(n)
    stop = os.path.exists(a.stop)
    ready = [t for n, t in tasks.items() if n not in running and n not in failed and not finished(t) and not t.get("manual") and waited(t) and all(finished(tasks[d]) for d in t["needs"])]
    ready.sort(key=lambda t: -t.get("prio", 0))
    if not stop:
        busy = {g for _, g in running.values()}
        for t in ready:
            if t["gpu"]:
                free = [g for g in gpus if g not in busy and gpu_free(g)]
                if not free: continue
                g = free[0]; busy.add(g)
            else:
                if sum(1 for n, (p, gg) in running.items() if gg is None) >= a.cpu_slots: continue
                g = None
            env = dict(env0); 
            if g is not None: env["CUDA_VISIBLE_DEVICES"] = str(g)
            lg = open(f"runs_lc/logs/{t['name']}.log", "a"); p = subprocess.Popen(["bash", "-c", t["cmd"]], env=env, stdout=lg, stderr=subprocess.STDOUT)
            running[t["name"]] = (p, g); t_start[t["name"]] = time.time(); print(time.strftime("%H:%M:%S"), "START", t["name"], "gpu", g, flush=True)
    write_state()
    pend = [n for n, t in tasks.items() if not finished(t) and n not in failed]
    waiting_gate = any(tasks[n].get("manual") for n in pend)
    waiting_file = any(not waited(tasks[n]) for n in pend)
    if not running and not waiting_gate and not waiting_file and (stop or not any(all(finished(tasks[d]) for d in tasks[n]["needs"]) for n in pend)): break
    time.sleep(20)
write_state(); print("queue finished; failed:", sorted(failed), flush=True)
