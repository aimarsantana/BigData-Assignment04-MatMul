#!/usr/bin/env python3
"""Re-run the C / OpenMP program of Assignment 3 on THIS machine, with the same input files, as the native
shared-memory reference of Assignment 4.

Assignment 3 was measured on a different computer (i7-13700K). To compare the distributed runs with "the parallel
version" without mixing machines, the unchanged Assignment 3 executable (scalar build, kernel unroll4, static
schedule, no pinning because this CPU is not hybrid) is run here for seq and for par with p threads.

Needs the Assignment 3 build:  <assignment-3 materials>\\build.ps1   (gcc with OpenMP, Windows).
Writes results/reference_a3_raw.csv (every sample) and results/reference_a3_native.csv (median / quartiles).
Usage: python scripts/run_reference_a3.py [--a3 DIR] [--data-dir DIR]
"""
import argparse
import csv
import os
import statistics
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIZES = (1024, 2048, 3072)
THREADS = (1, 2, 4, 8, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a3", default=os.path.join(ROOT, "..", "..", "assigment-3", "Assignment03_AlejandroSantana_Materials"))
    ap.add_argument("--data-dir", default=r"C:\bench_a4\data")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    exe = os.path.join(args.a3, "build", "matmul3_scalar.exe")
    if not os.path.exists(exe):
        sys.exit(f"{exe} not found: build Assignment 3 first (build.ps1)")
    commit = subprocess.run(["git", "-C", args.a3, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMP_")}
    raw, summ = [], []
    jobs = [("seq", 1)] + [("par", p) for p in THREADS]
    for n in SIZES:
        a, b = (os.path.join(args.data_dir, f"{x}_n{n}.bin") for x in "AB")
        for method, p in jobs:
            warm, reps = (1, 5)
            cmd = [exe, "dense", method, a, b, str(n), str(warm), str(reps), "0", str(p), "static", "none"]
            out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1200).stdout
            vals = []
            for line in out.splitlines():
                f = line.split(",")
                if f[0] == "SAMPLE":
                    raw.append({"method": f[1], "threads": p, "n": n, "phase": f[3], "index": f[4], "batch": f[5],
                                "seconds": float(f[6]) / 1e9, "checksum": f[7]})
                    if f[3] == "measured":
                        vals.append(float(f[6]) / 1e9)
            if vals:
                q = statistics.quantiles(vals, n=4, method="inclusive") if len(vals) > 1 else [vals[0]] * 3
                summ.append({"method": method, "threads": p, "n": n, "reps": len(vals), "median_s": statistics.median(vals),
                             "q1_s": q[0], "q3_s": q[2], "min_s": min(vals), "max_s": max(vals), "a3_commit": commit})
                print(method, p, n, f"{statistics.median(vals):.4f} s", flush=True)
    for name, rows in (("reference_a3_raw.csv", raw), ("reference_a3_native.csv", summ)):
        with open(os.path.join(args.results, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
