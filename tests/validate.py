#!/usr/bin/env python3
"""Correctness tests of Assignment 4 (run before the benchmarks).

1. Edge cases (bd.matmul4.Validate mode=edge, Spark local[3]): hand-computed 3x3 product, identity (left/right),
   zero, random matrices of size 1, 2, 3, 5, 7, 33, 100, 127, 130 against the naive triple loop, with 1, 3, 8 and
   40 partitions (more partitions than rows) and 1-5 blocks per side (more blocks than rows), for both
   distributed algorithms and both kernels.
2. The shared benchmark inputs (n = 512 and 1024) are multiplied on a real 2-container Spark cluster by both
   distributed algorithms and by the Java sequential kernels; the results are compared with NumPy (OpenBLAS, used
   only as the reference) with |dC| <= 1e-9 + 1e-9 |C|, and bit by bit with the sequential Java kernel.

Writes results/validation.csv (one row per check) and exits with status 1 if a check fails.
Usage: python tests/validate.py [--data-dir DIR] [--results DIR]
"""
import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_benchmarks as rb  # noqa: E402

ATOL = RTOL = 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=r"C:\bench_a4\data")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    results = os.path.abspath(args.results)
    out_dir = os.path.join(results, "validation_out")
    os.makedirs(out_dir, exist_ok=True)
    logs = os.path.join(results, "raw", "logs")
    os.makedirs(logs, exist_ok=True)
    cl = rb.Cluster(results, args.data_dir)
    rows = []
    edge_csv = os.path.join(results, "validation_edge.csv")
    if os.path.exists(edge_csv):
        os.remove(edge_csv)

    cl.up(2, 1)
    cl.exec_driver(["/opt/spark/bin/spark-submit", "--master", "local[3]", "--driver-memory", "1g",
                         "--conf", "spark.ui.enabled=false", "--class", "bd.matmul4.Validate", "/app/matmul4.jar",
                         "mode=edge", "out=/results"], os.path.join(logs, "validate_edge.log"), 3600)
    for r in csv.DictReader(open(edge_csv)):
        rows.append({"test": "edge:" + r["test"], "method": r["method"], "n": r["n"], "partitions": r["partitions"],
                     "grid": r["grid"], "max_abs_diff": r["max_abs_diff"],
                     "bitwise_equal_to_reference": r["bit_equal_to_reference"], "result": r["result"]})
    for n in (512, 1024):
        args_dump = ["mode=dump", f"n={n}", "out=/results/validation_out", "data=/data", "parts=8"]
        cl.exec_driver(["/opt/spark/bin/spark-submit", "--master", "spark://master:7077", "--driver-memory", "1g",
                        "--executor-memory", "640m", "--conf", "spark.executor.cores=1", "--conf", "spark.cores.max=2",
                        "--conf", "spark.ui.enabled=false", "--conf", "spark.driver.host=driver",
                        "--conf", "spark.driver.bindAddress=0.0.0.0", "--conf", "spark.rpc.message.maxSize=512",
                        "--class", "bd.matmul4.Validate", "/app/matmul4.jar"] + args_dump,
                       os.path.join(logs, f"validate_dump_{n}.log"), 3600)
        a = np.fromfile(os.path.join(args.data_dir, f"A_n{n}.bin"), dtype="<f8").reshape(n, n)
        b = np.fromfile(os.path.join(args.data_dir, f"B_n{n}.bin"), dtype="<f8").reshape(n, n)
        ref = a @ b
        seq = np.fromfile(os.path.join(out_dir, f"C_seq_n{n}.bin"), dtype="<f8").reshape(n, n)
        for m in ("seq", "ijk", "rowblock", "blockshuffle"):
            c = np.fromfile(os.path.join(out_dir, f"C_{m}_n{n}.bin"), dtype="<f8").reshape(n, n)
            ok = bool(np.all(np.abs(c - ref) <= ATOL + RTOL * np.abs(ref)))
            rows.append({"test": "benchmark_inputs_vs_numpy", "method": m, "n": n, "partitions": 8 if "block" in m else "",
                         "grid": 4 if m == "blockshuffle" else "", "max_abs_diff": float(np.max(np.abs(c - ref))),
                         "bitwise_equal_to_reference": bool(np.array_equal(c.view("<u8"), seq.view("<u8"))),
                         "result": "PASS" if ok else "FAIL"})
    cl.down()
    with open(os.path.join(results, "validation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["result"] != "PASS"]
    print(f"{len(rows) - len(bad)} of {len(rows)} checks passed")
    for r in bad:
        print("FAIL", r)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
