#!/usr/bin/env python3
"""Benchmark harness for Assignment 4 (Spark standalone cluster in Docker containers on one machine).

For every cluster layout 'ExC' (E worker containers with C cores each; the executors are the JVMs inside the
workers) the harness
  1. starts the cluster (docker compose up --scale worker=E, WORKER_CORES=C) and waits for E executors,
  2. runs every experiment that uses this layout as ONE spark-submit application (bd.matmul4.Bench):
     inputs are read outside the timer, 2 warm-up runs and 5-7 measured runs per size, raw samples kept,
  3. stops the cluster.
The local baselines (sequential and multi-threaded Java in the driver container, no Spark) run with the
cluster stopped. A short anchor run (same sequential kernel, n = 1024) is repeated between the groups so that
drift of the machine (frequency, temperature, background load) over the campaign can be seen.

Budgets (config/benchmark_config.json): a run longer than 120 s ends the series (TIMEOUT) and larger sizes
of that series are NOT_ATTEMPTED; no time is invented for them.

Outputs (results/raw/, appended; delete the folder for a clean full run):
  samples_<exp>_<layout>.csv   every warm-up and measured sample with the Spark task metrics
  runs_<exp>_<layout>.csv      one row per (method, n): status, repetitions, load time, reference time
  info_<exp>_<layout>.csv      Spark / JVM / cluster description printed by the application
  logs/*.log                   stdout + stderr of every spark-submit / java process

Usage: python scripts/run_benchmarks.py [--only D1_strong_rowblock,L1_local] [--results DIR] [--data-dir DIR]
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCKER = os.path.join(ROOT, "docker")
CFG = json.load(open(os.path.join(ROOT, "config", "benchmark_config.json")))
SPARK = CFG["spark"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sh(cmd, env=None, log_path=None, timeout=None, check=True):
    e = dict(os.environ)
    e.update(env or {})
    e["MSYS_NO_PATHCONV"] = "1"
    out = open(log_path, "a", encoding="utf8") if log_path else subprocess.DEVNULL
    if log_path:
        out.write("$ " + " ".join(cmd) + "\n")
        out.flush()
    r = subprocess.run(cmd, cwd=DOCKER, env=e, stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
    if log_path:
        out.close()
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")
    return r.returncode


class Cluster:
    def __init__(self, results, data):
        self.env = {"A4_RESULTS": results.replace("\\", "/"), "A4_DATA": data.replace("\\", "/")}
        self.results = results

    def down(self):
        sh(["docker", "compose", "down", "-t", "3"], self.env, check=False)

    def up(self, workers, cores):
        self.down()
        env = dict(self.env, WORKER_CORES=str(cores), WORKER_MEM="640m")
        sh(["docker", "compose", "up", "-d", "--scale", f"worker={workers}"], env)
        time.sleep(12)          # master and workers register; the application waits for the executors itself

    def up_driver_only(self):
        self.down()
        sh(["docker", "compose", "up", "-d", "--no-deps", "driver"], self.env)
        time.sleep(3)

    def exec_driver(self, cmd, log_path, timeout):
        return sh(["docker", "compose", "exec", "-T", "driver"] + cmd, self.env, log_path, timeout, check=False)


def spark_submit(cl, workers, cores, args, log_path):
    total = workers * cores
    conf = ["--conf", f"spark.executor.cores={cores}", "--conf", f"spark.cores.max={total}",
            "--conf", "spark.driver.host=driver", "--conf", "spark.driver.bindAddress=0.0.0.0"]
    for k, v in SPARK["extra_conf"].items():
        conf += ["--conf", f"{k}={v}"]
    cmd = ["/opt/spark/bin/spark-submit", "--master", "spark://master:7077", "--driver-memory", SPARK["driver_memory"],
           "--executor-memory", SPARK["executor_memory"]] + conf + ["--class", "bd.matmul4.Bench", "/app/matmul4.jar"] + args
    return cl.exec_driver(cmd, log_path, timeout=3 * 3600)


def local_java(cl, args, log_path, heap="1g"):
    cmd = ["java", f"-Xmx{heap}", "-cp", "/app/matmul4.jar:/opt/spark/jars/*", "bd.matmul4.Bench"] + args
    return cl.exec_driver(cmd, log_path, timeout=1800)


def reps_args(common=()):
    return [f"warmup={CFG['warmup_runs']}", f"reps={CFG['repetitions']['n<=1024']}",
            f"repsLarge={CFG['repetitions']['n>1024']}", f"budgetS={CFG['budget']['max_seconds_per_sample']}",
            "data=/data", "out=/results/raw"] + list(common)


def anchor(cl, tag, logs):
    a = CFG["anchor"]
    local_java(cl, ["mode=local", f"exp=ANCHOR_{tag}", f"method={a['method']}", f"kernel={a['kernel']}", "threads=1",
                    f"n={a['n']}", "warmup=2", f"reps={a['reps']}", "data=/data", "out=/results/raw"],
               os.path.join(logs, "anchor.log"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated experiment ids (default: all)")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--data-dir", default=r"C:\bench_a4\data")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    results = os.path.abspath(args.results)
    logs = os.path.join(results, "raw", "logs")
    os.makedirs(logs, exist_ok=True)
    cl = Cluster(results, args.data_dir)

    spark_exps = [e for e in CFG["experiments"] if not only or e["id"] in only]
    layouts = []
    for e in spark_exps:
        for l in e["layouts"]:
            if l not in layouts:
                layouts.append(l)
    layouts.sort(key=lambda l: (int(l.split("x")[0]) * int(l.split("x")[1]), int(l.split("x")[0])))

    run_local = not only or "L1_local" in only
    if run_local:
        cl.up_driver_only()
        anchor(cl, "start", logs)

    for layout in layouts:
        w, c = (int(x) for x in layout.split("x"))
        log(f"=== cluster {layout}: {w} worker containers x {c} cores")
        cl.up(w, c)
        for e in spark_exps:
            if layout not in e["layouts"]:
                continue
            sizes = e["sizes_by_cores"][str(w * c)] if "sizes_by_cores" in e else e["sizes"]
            a = ["mode=spark", f"exp={e['id']}", f"method={e['method']}", f"kernel={e['kernel']}", f"workers={w}", f"cores={c}",
                 f"sizes={','.join(map(str, sizes))}", f"partsPerCore={CFG['partitions_per_core']}"]
            if "grid" in e:
                a.append(f"grid={e['grid']}")
            if "partsList" in e:
                a.append("partsList=" + ",".join(map(str, e["partsList"])))
            log(f"  {e['id']} sizes={sizes}")
            rc = spark_submit(cl, w, c, a + reps_args(), os.path.join(logs, f"{e['id']}_L{w}x{c}.log"))
            if rc != 0:
                log(f"  !! {e['id']} on {layout} exited with code {rc} (see log)")
        cl.down()

    if run_local:
        cl.up_driver_only()
        anchor(cl, "mid", logs)
        for grp in CFG["local"]:
            for r in grp["runs"]:
                for n in r["sizes"]:
                    for t in r["threads"]:
                        log(f"  {grp['id']} {r['method']}/{r['kernel']} p={t} n={n}")
                        local_java(cl, ["mode=local", f"exp={grp['id']}", f"method={r['method']}", f"kernel={r['kernel']}",
                                        f"threads={t}", f"n={n}"] + reps_args(), os.path.join(logs, f"{grp['id']}.log"), grp["driver_heap"])
        anchor(cl, "end", logs)
        cl.down()
    log("done")


if __name__ == "__main__":
    sys.exit(main())
