#!/usr/bin/env python3
"""Analysis of the Assignment 4 measurements: summary statistics, speed-up, efficiency, overhead breakdown, figures
and the tables of the report.

Input : results/raw/{samples,runs,info}_*.csv (written by run_benchmarks.py), results/validation.csv,
        results/reference_a3_native.csv (optional, scripts/run_reference_a3.py)
Output: results/raw_samples.csv, runs.csv, info.csv   the per-process files concatenated
        results/summary.csv          median / quartiles / min / max / std of every configuration and phase medians
        results/scaling.csv          speed-up and efficiency of every distributed configuration
        results/key_numbers.json     numbers quoted in the report
        figures/*.png, *.pdf         all figures
        report/generated/*.html      tables included in the report

Statistics: central value = median of the measured samples (warm-up excluded); variability = interquartile range (IQR,
linear quartiles). Times are end-to-end seconds of ONE multiplication (see Bench.java for the timer boundary).
Speed-up definitions (C = total cores = containers x cores per container):
  S_seq(C) = T_seq / T_dist(C)    T_seq = sequential Java run (same kernel, same container image) of the same n
  S_thr(C) = T_thr(C) / T_dist(C) T_thr = multi-threaded Java run with p = C threads in ONE JVM (same kernel)
  S_1(C)   = T_dist(1 core) / T_dist(C)  scaling of the framework itself;  E = S / C
All from medians. Weak scaling: E_w(C) = T(1 core, n_1) / T(C, n_C) with n_C^3 = C * n_1^3.
"""
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
GEN = os.path.join(ROOT, "report", "generated")
KEY = ["experiment", "method", "kernel", "workers", "cores_per_worker", "threads", "partitions", "grid", "n"]
NCOL = {512: "#009E73", 1024: "#0072B2", 1290: "#56B4E9", 1626: "#E69F00", 2048: "#D55E00", 3072: "#CC79A7", 1536: "#F0E442"}
LCOL = {"1x1": "#0072B2", "2x1": "#009E73", "4x1": "#E69F00", "4x2": "#D55E00", "1x2": "#56B4E9", "1x4": "#999999",
        "2x2": "#CC79A7", "1x8": "#000000", "2x4": "#F0E442"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 130, "legend.fontsize": 8})


# ----------------------------------------------------------------------------- loading
def cat(prefix):
    files = sorted(glob.glob(os.path.join(RES, "raw", f"{prefix}_*.csv")))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load():
    s, r, i = cat("samples"), cat("runs"), cat("info")
    for df in (s, r):
        df["cores_per_worker"] = np.where(df["workers"] == 0, 0, df["cores_per_worker"])
    s.to_csv(os.path.join(RES, "raw_samples.csv"), index=False)
    r.to_csv(os.path.join(RES, "runs.csv"), index=False)
    i.to_csv(os.path.join(RES, "info.csv"), index=False)
    s["t"] = s["t_total_ns"] / 1e9
    return s, r, i


def summarise(s):
    m = s[s["phase"] == "measured"].copy()
    g = m.groupby(KEY)
    out = g["t"].agg(reps="count", median="median", min="min", max="max", mean="mean", std="std")
    out["q1"] = g["t"].quantile(0.25)
    out["q3"] = g["t"].quantile(0.75)
    out["iqr"] = out["q3"] - out["q1"]
    out["rel_iqr"] = out["iqr"] / out["median"]
    med_cols = ["t_partition_ns", "t_broadcast_ns", "t_job_ns", "t_assemble_ns", "sum_kernel_ns", "sum_fetch_ns", "tasks",
                "executor_run_ms", "executor_cpu_ns", "deserialize_ms", "result_ser_ms", "result_size_bytes", "gc_ms",
                "shuffle_remote_bytes", "shuffle_local_bytes", "shuffle_write_bytes", "shuffle_fetch_wait_ms",
                "spill_mem_bytes", "spill_disk_bytes", "driver_rx_bytes", "driver_tx_bytes"]
    for c in med_cols:
        out["med_" + c] = g[c].median()
    for c in ("peak_exec_vmhwm_kib", "driver_vmhwm_kib"):
        out["max_" + c] = g[c].max()
    chk = m.assign(ok=m["check_ok"].astype(str).str.lower() == "true", bit=m["bit_equal"].astype(str).str.lower() == "true")
    gg = chk.groupby(KEY)
    out["check_ok_all"] = gg["ok"].all()
    out["bit_equal_all"] = gg["bit"].all()
    out["max_abs_diff"] = g["max_abs_diff"].max()
    out = out.reset_index()
    out["cores"] = np.where(out["workers"] == 0, out["threads"], out["workers"] * out["cores_per_worker"])
    out["layout"] = np.where(out["workers"] == 0, "local", out["workers"].astype(str) + "x" + out["cores_per_worker"].astype(str))
    return out


# ----------------------------------------------------------------------------- helpers
def fmt_t(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "&ndash;"
    if x >= 100:
        return f"{x:.0f}&nbsp;s"
    if x >= 1:
        return f"{x:.2f}&nbsp;s"
    return f"{x * 1e3:.0f}&nbsp;ms"


def fmt_mib(x):
    return "&ndash;" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0f}"


def table(header, rows, caption, cls="small"):
    h = "".join(f"<th>{c}</th>" for c in header)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="{cls}"><caption>{caption}</caption><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def write(name, html):
    os.makedirs(GEN, exist_ok=True)
    open(os.path.join(GEN, name), "w", encoding="utf8").write(html)


class Lookup:
    def __init__(self, summ):
        self.s = summ
        self.idx = {}
        for r in summ.itertuples(index=False):
            self.idx[(r.experiment, r.method, r.kernel, r.workers, r.cores_per_worker, r.threads, r.partitions, r.grid, r.n)] = r

    def spark(self, exp, layout, n, parts=None):
        w, c = (int(x) for x in layout.split("x"))
        d = self.s[(self.s.experiment == exp) & (self.s.workers == w) & (self.s.cores_per_worker == c) & (self.s.n == n)]
        if parts is not None:
            d = d[d.partitions == parts]
        return d.iloc[0] if len(d) else None

    def local(self, method, kernel, threads, n, exp="L1_local"):
        d = self.s[(self.s.experiment == exp) & (self.s.method == method) & (self.s.kernel == kernel)
                   & (self.s.threads == threads) & (self.s.n == n)]
        return d.iloc[0] if len(d) else None


def tl(lk, method, kernel, threads, n):
    r = lk.local(method, kernel, threads, n)
    return float(r["median"]) if r is not None else float("nan")


# ----------------------------------------------------------------------------- scaling table
def scaling(summ, lk):
    rows = []
    sp = summ[summ.workers > 0]
    for r in sp.itertuples(index=False):
        if r.experiment in ("D3_layout", "D5_partitions"):
            pass
        C = r.workers * r.cores_per_worker
        base = lk.spark(r.experiment, "1x1", r.n)
        kern = r.kernel
        seqk = "unroll4" if kern == "unroll4" else "blocked"
        tseq = tl(lk, "seq", seqk, 1, r.n)
        tthr = tl(lk, "threads", seqk, C, r.n)
        rows.append({"experiment": r.experiment, "method": r.method, "kernel": kern, "layout": r.layout, "cores": C, "n": r.n,
                     "partitions": r.partitions, "median_s": r.median, "iqr_s": r.iqr, "reps": r.reps,
                     "T_seq_java_s": tseq, "S_seq": tseq / r.median, "E_seq": tseq / r.median / C,
                     "T_threads_java_s": tthr, "S_threads": tthr / r.median,
                     "T_1core_dist_s": float(base["median"]) if base is not None else float("nan"),
                     "S_1": (float(base["median"]) / r.median) if base is not None else float("nan"),
                     "E_1": (float(base["median"]) / r.median / C) if base is not None else float("nan")})
    sc = pd.DataFrame(rows)
    sc.to_csv(os.path.join(RES, "scaling.csv"), index=False)
    return sc


# ----------------------------------------------------------------------------- figures
def fig_strong(sc, lk, exp, name, title, sizes):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    d = sc[sc.experiment == exp]
    Cs = sorted(d.cores.unique())
    for n in sizes:
        dd = d[d.n == n].sort_values("cores")
        if dd.empty:
            continue
        ax[0].plot(dd.cores, dd.median_s, "o-", color=NCOL[n], label=f"n = {n}")
        ax[1].plot(dd.cores, dd.S_seq, "o-", color=NCOL[n], label=f"n = {n}")
        thr = [(c, tl(lk, "threads", "blocked", c, n)) for c in Cs]
        thr = [(c, t) for c, t in thr if not math.isnan(t)]
        if exp != "D2_strong_blockshuffle" and thr:
            ax[0].plot([c for c, _ in thr], [t for _, t in thr], "--", color=NCOL[n], alpha=0.7)
            ts = tl(lk, "seq", "blocked", 1, n)
            ax[1].plot([c for c, _ in thr], [ts / t for _, t in thr], "--", color=NCOL[n], alpha=0.7)
        ax[2].plot(dd.cores, dd.E_1, "o-", color=NCOL[n], label=f"n = {n}")
    ax[0].set(xscale="log", yscale="log", xlabel="total cores C", ylabel="end-to-end time (s)", title="Time (median; IQR below marker size)")
    ax[1].plot(Cs, Cs, ":", color="grey", label="ideal S = C")
    ax[1].set(xscale="log", yscale="log", xlabel="total cores C", ylabel="S_seq = T_seq(Java) / T(C)", title="Speed-up over sequential Java")
    ax[2].axhline(1, ls=":", color="grey")
    ax[2].set(xscale="log", xlabel="total cores C", ylabel="E_1 = T(1 core) / (C T(C))", title="Efficiency of the distributed run vs itself", ylim=(0, 1.15))
    for a in ax:
        a.set_xticks(Cs)
        a.set_xticklabels([str(c) for c in Cs])
        a.minorticks_off()
    ax[1].legend(loc="upper left")
    if exp != "D2_strong_blockshuffle":
        ax[0].plot([], [], "k--", label="Java threads, 1 JVM")
        ax[0].legend(loc="lower left")
    fig.suptitle(title, y=1.02)
    save(fig, name)


def fig_breakdown(summ, exp, name, sizes, layouts):
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.4 * len(sizes), 3.8), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, sizes):
        parts = {k: [] for k in ("driver: build RDD + A bands", "driver: broadcast", "compute (Σ kernel / C)",
                                 "job rest: scheduling, B fetch, result transfer", "driver: assemble C")}
        labels = []
        for l in layouts:
            w, c = (int(x) for x in l.split("x"))
            d = summ[(summ.experiment == exp) & (summ.workers == w) & (summ.cores_per_worker == c) & (summ.n == n)]
            if d.empty:
                continue
            r = d.iloc[0]
            C = w * c
            job = r.med_t_job_ns / 1e9
            comp = min(r.med_sum_kernel_ns / 1e9 / C, job)
            for k, v in zip(parts, (r.med_t_partition_ns / 1e9, r.med_t_broadcast_ns / 1e9, comp, job - comp, r.med_t_assemble_ns / 1e9)):
                parts[k].append(v)
            labels.append(f"{l}\n(C={C})")
        bottom = np.zeros(len(labels))
        for (k, v), col in zip(parts.items(), ["#999999", "#E69F00", "#0072B2", "#D55E00", "#009E73"]):
            ax.bar(range(len(labels)), v, bottom=bottom, label=k, color=col)
            bottom += np.array(v)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set(title=f"n = {n}", ylabel="time (s)")
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Where the time goes (median per phase; the job phase overlaps compute and transfer on the executors)", y=1.02)
    save(fig, name)


def fig_layout(summ, name):
    d = summ[summ.experiment == "D3_layout"]
    sizes = sorted(d.n.unique())
    fig, ax = plt.subplots(1, len(sizes), figsize=(4.3 * len(sizes), 3.6))
    ax = np.atleast_1d(ax)
    groups = [("C = 4", ["1x4", "2x2", "4x1"]), ("C = 8", ["1x8", "2x4", "4x2"])]
    for a, n in zip(ax, sizes):
        x = 0
        ticks, labs = [], []
        for gname, ls in groups:
            for l in ls:
                w, c = (int(v) for v in l.split("x"))
                r = d[(d.workers == w) & (d.cores_per_worker == c) & (d.n == n)]
                if r.empty:
                    x += 1
                    continue
                r = r.iloc[0]
                a.bar(x, r["median"], color=LCOL[l], yerr=[[r["median"] - r.q1], [r.q3 - r["median"]]], capsize=3)
                ticks.append(x)
                labs.append(f"{l}\n{w} exec")
                x += 1
            x += 0.6
        a.set_xticks(ticks)
        a.set_xticklabels(labs, fontsize=7)
        a.set(title=f"n = {n}", ylabel="end-to-end time (s)")
    fig.suptitle("Same total cores, different split: ExC = E executor containers x C cores (bars = median, whiskers = quartiles)", y=1.02)
    save(fig, name)


def fig_weak(summ, name):
    d = summ[(summ.experiment == "D6_weak")].copy()
    d["C"] = d.workers * d.cores_per_worker
    d = d.sort_values("C")
    if d.empty:
        return None
    t1 = float(d.iloc[0]["median"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].errorbar(d.C, d["median"], yerr=[d["median"] - d.q1, d.q3 - d["median"]], fmt="o-", color="#0072B2", capsize=3)
    for c, n, t in zip(d.C, d.n, d["median"]):
        ax[0].annotate(f"n={n}", (c, t), textcoords="offset points", xytext=(4, 6), fontsize=7)
    ax[0].axhline(t1, ls=":", color="grey")
    ax[0].set(xscale="log", xlabel="total cores C", ylabel="end-to-end time (s)", title="Weak scaling: n^3 = C * 1024^3 (ideal: flat)", ylim=(0, None))
    ax[1].plot(d.C, t1 / d["median"], "o-", color="#D55E00")
    ax[1].axhline(1, ls=":", color="grey")
    ax[1].set(xscale="log", xlabel="total cores C", ylabel="E_w = T(1) / T(C)", title="Weak-scaling efficiency", ylim=(0, 1.15))
    for a in ax:
        a.set_xticks(list(d.C))
        a.set_xticklabels([str(int(c)) for c in d.C])
        a.minorticks_off()
    save(fig, name)
    return {int(c): float(t1 / t) for c, t in zip(d.C, d["median"])}


def fig_partitions(summ, name):
    d = summ[summ.experiment == "D5_partitions"].sort_values("partitions")
    if d.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].errorbar(d.partitions, d["median"], yerr=[d["median"] - d.q1, d.q3 - d["median"]], fmt="o-", color="#0072B2", capsize=3, label="end-to-end")
    ax[0].plot(d.partitions, d.med_t_job_ns / 1e9, "s--", color="#D55E00", label="job (driver clock)")
    ax[0].set(xscale="log", xlabel="number of tasks (row bands)", ylabel="time (s)", title="4 containers x 2 cores, n = 2048")
    ax[0].legend()
    ax[1].plot(d.partitions, d.med_result_size_bytes / 1e6, "o-", color="#009E73", label="result bytes (Σ tasks)")
    ax2 = ax[1].twinx()
    ax2.plot(d.partitions, d.med_deserialize_ms / 1e3, "s--", color="#CC79A7")
    ax2.set_ylabel("Σ task deserialisation (s)", color="#CC79A7")
    ax2.grid(False)
    ax[1].set(xscale="log", xlabel="number of tasks (row bands)", ylabel="MB returned to the driver", title="Per-task costs")
    save(fig, name)


def fig_result(sc, summ, name):
    a = summ[summ.experiment == "D1_strong_rowblock"]
    b = summ[summ.experiment == "D4_result_stays"]
    if b.empty:
        return
    sizes = [1024, 2048, 3072]
    fig, ax = plt.subplots(1, len(sizes), figsize=(4.3 * len(sizes), 3.4))
    for axx, n in zip(ax, sizes):
        for k, (lab, df, col) in enumerate([("C returned to driver", a, "#0072B2"), ("only checksums returned", b, "#D55E00")]):
            xs, ys = [], []
            for l in ["1x1", "2x1", "4x1", "4x2"]:
                w, c = (int(v) for v in l.split("x"))
                r = df[(df.workers == w) & (df.cores_per_worker == c) & (df.n == n)]
                if len(r):
                    xs.append(w * c)
                    ys.append(float(r.iloc[0]["median"]))
            axx.plot(xs, ys, "o-", color=col, label=lab)
        axx.set(xscale="log", yscale="log", xlabel="total cores C", ylabel="time (s)", title=f"n = {n}")
        axx.set_xticks([1, 2, 4, 8])
        axx.set_xticklabels(["1", "2", "4", "8"])
        axx.minorticks_off()
    ax[0].legend()
    save(fig, name)


def fig_kernel(sc, lk, name):
    d1 = sc[(sc.experiment == "D1_strong_rowblock") & (sc.n == 2048)].sort_values("cores")
    d7 = sc[(sc.experiment == "D7_unroll4_kernel") & (sc.n == 2048)].sort_values("cores")
    if d7.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    Cs = [1, 2, 4, 8]
    for kern, dd, col in (("blocked", d1, "#0072B2"), ("unroll4", d7, "#D55E00")):
        ts = tl(lk, "seq", kern, 1, 2048)
        ax[0].plot(dd.cores, dd.median_s, "o-", color=col, label=f"Spark, {kern}")
        thr = [(c, tl(lk, "threads", kern, c, 2048)) for c in Cs]
        thr = [(c, t) for c, t in thr if not math.isnan(t)]
        ax[0].plot([c for c, _ in thr], [t for _, t in thr], "s--", color=col, alpha=0.7, label=f"Java threads, {kern}")
        ax[1].plot(dd.cores, ts / dd.median_s, "o-", color=col, label=f"Spark, {kern}")
        ax[1].plot([c for c, _ in thr], [ts / t for _, t in thr], "s--", color=col, alpha=0.7, label=f"Java threads, {kern}")
    ax[1].plot(Cs, Cs, ":", color="grey")
    ax[0].set(xscale="log", yscale="log", xlabel="total cores C", ylabel="time (s)", title="n = 2048: time")
    ax[1].set(xscale="log", yscale="log", xlabel="total cores C", ylabel="speed-up over the same sequential kernel", title="n = 2048: speed-up")
    for a in ax:
        a.set_xticks(Cs)
        a.set_xticklabels([str(c) for c in Cs])
        a.minorticks_off()
    ax[0].legend(fontsize=7)
    save(fig, name)


def fig_baselines(summ, lk, ref, name):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sizes = [512, 1024, 2048, 3072]

    def series(f, **kw):
        xs = [n for n in sizes if not math.isnan(f(n))]
        ax.plot(xs, [f(n) for n in xs], **kw)

    series(lambda n: tl(lk, "ijk", "ijk", 1, n) if n <= 1536 else float("nan"), marker="^", color="#999999", label="Java ijk (basic, A1 algorithm)")
    series(lambda n: tl(lk, "seq", "unroll4", 1, n), marker="o", color="#CC79A7", label="Java seq, unroll4")
    series(lambda n: tl(lk, "seq", "blocked", 1, n), marker="o", color="#0072B2", label="Java seq, blocked")
    series(lambda n: tl(lk, "threads", "blocked", 8, n), marker="s", color="#009E73", label="Java 8 threads, 1 JVM, blocked")
    for l, col in (("1x1", "#E69F00"), ("4x2", "#D55E00")):
        def f(n, l=l):
            r = lk.spark("D1_strong_rowblock", l, n)
            return float(r["median"]) if r is not None else float("nan")
        series(f, marker="D", color=col, label=f"Spark rowblock {l}")
    if ref is not None:
        for m, p, col, ls in (("seq", 1, "#000000", "-."), ("par", 8, "#56B4E9", "-.")):
            d = ref[(ref.method == m) & (ref.threads == p)]
            if len(d):
                ax.plot(d.n, d.median_s, ls, marker="x", color=col, label=f"C (A3 build, this machine) {m}" + (f" p={p}" if m == "par" else ""))
    ax.set(xscale="log", yscale="log", xlabel="matrix size n", ylabel="time (s)", title="Baselines and distributed runs, same machine")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.minorticks_off()
    ax.legend(fontsize=7)
    save(fig, name)


def fig_memory(summ, name):
    d = summ[summ.experiment == "D1_strong_rowblock"]
    loc = summ[(summ.experiment == "L1_local") & (summ.method == "seq") & (summ.kernel == "blocked")]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for l in ["1x1", "2x1", "4x1", "4x2"]:
        w, c = (int(v) for v in l.split("x"))
        dd = d[(d.workers == w) & (d.cores_per_worker == c)].sort_values("n")
        ax[0].plot(dd.n, dd.max_peak_exec_vmhwm_kib / 1024, "o-", color=LCOL[l], label=f"executor JVM, {l}")
    ax[0].plot(loc.n, loc.max_driver_vmhwm_kib / 1024, "k--s", label="Java sequential process (A, B, C, reference)")
    ns = np.array(sorted(d.n.unique()))
    ax[0].plot(ns, 8 * ns.astype(float) ** 2 / 2 ** 20, ":", color="grey", label="one n x n matrix (8 n^2 bytes)")
    ax[0].set(xlabel="n", ylabel="peak resident memory VmHWM (MiB)", title="Memory per process (peak so far)", yscale="log")
    ax[0].legend(fontsize=7)
    for l in ["1x1", "2x1", "4x1", "4x2"]:
        w, c = (int(v) for v in l.split("x"))
        dd = d[(d.workers == w) & (d.cores_per_worker == c)].sort_values("n")
        ax[1].plot(dd.n, (dd.med_driver_tx_bytes) / 2 ** 20, "o-", color=LCOL[l], label=f"{l}: driver sent")
    ax[1].plot(ns, 8 * ns.astype(float) ** 2 / 2 ** 20, ":", color="grey", label="8 n^2 bytes")
    ax[1].set(xlabel="n", ylabel="MiB per multiplication", title="Driver network traffic (eth0 tx, Σ over the run)", yscale="log")
    ax[1].legend(fontsize=7)
    save(fig, name)


def fig_drift(summ, name):
    d = summ[summ.experiment.str.startswith("ANCHOR")].copy()
    if d.empty:
        return None
    order = ["ANCHOR_start", "ANCHOR_mid", "ANCHOR_end"]
    d = d.set_index("experiment").reindex(order).dropna(how="all")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.errorbar(range(len(d)), d["median"], yerr=[d["median"] - d.q1, d.q3 - d["median"]], fmt="o", color="#0072B2", capsize=4)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([e.replace("ANCHOR_", "") for e in d.index])
    ax.set(ylabel="time, sequential blocked n=1024 (s)", title="Machine drift during the campaign", ylim=(0, float(d["median"].max()) * 1.3))
    save(fig, name)
    return {e.replace("ANCHOR_", ""): float(v) for e, v in d["median"].items()}


# ----------------------------------------------------------------------------- tables
def tables(summ, sc, lk):
    sizes = [512, 1024, 2048, 3072]
    lay = ["1x1", "2x1", "4x1", "4x2"]
    rows = []
    for n in sizes:
        row = [n, fmt_t(tl(lk, "seq", "blocked", 1, n)) + "<br><span class='iqr'>" + fmt_t(float(lk.local('seq', 'blocked', 1, n)['iqr']) if lk.local('seq', 'blocked', 1, n) is not None else None) + "</span>"]
        t8 = lk.local("threads", "blocked", 8, n)
        row.append(fmt_t(float(t8["median"])) + "<br><span class='iqr'>" + fmt_t(float(t8["iqr"])) + "</span>" if t8 is not None else "&ndash;")
        for l in lay:
            r = lk.spark("D1_strong_rowblock", l, n)
            row.append(fmt_t(float(r["median"])) + "<br><span class='iqr'>" + fmt_t(float(r["iqr"])) + "</span>" if r is not None else "&ndash;")
        rows.append(row)
    write("table_time.html", table(["n", "Java seq", "Java 8 threads", "Spark 1x1 (C=1)", "Spark 2x1 (C=2)", "Spark 4x1 (C=4)", "Spark 4x2 (C=8)"], rows,
                                  "Table 3. D1: end-to-end time of the row-band/broadcast algorithm (median; IQR in grey; 7 repetitions for n&nbsp;&le;&nbsp;1024, 5 above). "
                                  "Java seq and Java 8 threads: local baselines with the same blocked kernel in one JVM. ExC = E worker containers with C cores each."))

    def sp_table(exp, cap, name, sizes_):
        rows = []
        for n in sizes_:
            for l in lay:
                d = sc[(sc.experiment == exp) & (sc.layout == l) & (sc.n == n)]
                if d.empty:
                    continue
                r = d.iloc[0]
                rows.append([n, l, int(r.cores), fmt_t(r.median_s), f"{r.S_seq:.2f}", f"{r.E_seq:.2f}",
                             "&ndash;" if math.isnan(r.S_threads) else f"{r.S_threads:.2f}", f"{r.S_1:.2f}", f"{r.E_1:.2f}"])
        write(name, table(["n", "layout", "C", "time", "S_seq", "E_seq", "S_thr", "S_1", "E_1"], rows, cap))

    sp_table("D1_strong_rowblock", "Table 4. D1: speed-up S_seq = T_seq(Java)/T, efficiency E_seq = S_seq/C, S_thr = T(Java threads, p = C)/T (&lt;&nbsp;1: the in-JVM threads are faster), S_1 = T(1x1)/T and E_1 = S_1/C.",
             "table_speedup.html", sizes)
    sp_table("D2_strong_blockshuffle", "Table 6. D2: block algorithm with two shuffles (4&times;4 blocks, 64 block products); same columns as Table 4.",
             "table_blockshuffle.html", [512, 1024, 2048])
    sp_table("D7_unroll4_kernel", "Table 8. D7: the unroll4 kernel of Assignments 2 and 3 inside the same row-band algorithm (T_seq and T_thr are the unroll4 Java baselines).",
             "table_unroll4.html", [1024, 2048])
    # layouts
    rows = []
    for n in (1024, 2048, 3072):
        for C, ls in ((4, ["1x4", "2x2", "4x1"]), (8, ["1x8", "2x4", "4x2"])):
            row = [n, C]
            for l in ls:
                r = lk.spark("D3_layout", l, n)
                row.append(fmt_t(float(r["median"])) + "<br><span class='iqr'>" + fmt_t(float(r["iqr"])) + "</span>" if r is not None else "&ndash;")
            rows.append(row)
    write("table_layout.html", table(["n", "C", "1 container / E=2 / E=4 ...", "", ""], rows, ""))
    # breakdown of D1 at 2048 and 3072
    rows = []
    for n in (2048, 3072):
        for l in lay:
            r = lk.spark("D1_strong_rowblock", l, n)
            if r is None:
                continue
            C = r.workers * r.cores_per_worker
            job = r.med_t_job_ns / 1e9
            rows.append([n, l, f"{r.med_t_partition_ns / 1e9:.3f}", f"{r.med_t_broadcast_ns / 1e9:.3f}", f"{job:.3f}",
                         f"{r.med_sum_kernel_ns / 1e9:.2f}", f"{r.med_sum_kernel_ns / 1e9 / C:.3f}", f"{r.med_sum_fetch_ns / 1e9:.2f}",
                         f"{r.med_t_assemble_ns / 1e9:.3f}", f"{r.med_gc_ms / 1e3:.2f}", f"{r.med_t_job_ns / 1e9 - min(r.med_sum_kernel_ns / 1e9 / C, job):.3f}"])
    write("table_breakdown.html", table(["n", "layout", "build RDD (s)", "broadcast (s)", "job (s)", "Σ kernel (task-s)", "Σ kernel / C (s)", "Σ B fetch (task-s)",
                                         "assemble (s)", "Σ GC (s)", "job &minus; Σ kernel / C (s)"], rows,
                                        "Table 5. D1 phase medians. The timed phases (build RDD, broadcast, job, assemble) add up to the end-to-end time; kernel, fetch and GC are sums over tasks and overlap in time."))
    # memory + network
    rows = []
    for n in sizes:
        for l in lay:
            r = lk.spark("D1_strong_rowblock", l, n)
            if r is None:
                continue
            w = r.workers
            rows.append([n, l, f"{8 * n * n / 2 ** 20:.0f}", fmt_mib(r.max_peak_exec_vmhwm_kib / 1024), fmt_mib(r.max_driver_vmhwm_kib / 1024),
                         f"{r.med_driver_tx_bytes / 2 ** 20:.0f}", f"{r.med_driver_rx_bytes / 2 ** 20:.0f}"])
    write("table_memory.html", table(["n", "layout", "8n<sup>2</sup> (MiB)", "executor VmHWM (MiB)", "driver VmHWM (MiB)", "driver tx (MiB)", "driver rx (MiB)"], rows,
                                     "Table 9. D1: peak resident memory (VmHWM, cumulative maximum over the application: sizes run in increasing order) and driver network traffic (median per multiplication)."))


# ----------------------------------------------------------------------------- reference (A3 native C on this machine)
def load_ref():
    p = os.path.join(RES, "reference_a3_native.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


# ----------------------------------------------------------------------------- main
def main():
    s, r, info = load()
    summ = summarise(s)
    summ.to_csv(os.path.join(RES, "summary.csv"), index=False)
    lk = Lookup(summ)
    sc = scaling(summ, lk)
    ref = load_ref()
    k = {}
    k["n_samples_measured"] = int((s.phase == "measured").sum())
    k["n_samples_warmup"] = int((s.phase == "warmup").sum())
    k["n_configurations"] = int(len(summ))
    k["median_rel_iqr"] = float(summ["rel_iqr"].median())
    k["p95_rel_iqr"] = float(summ["rel_iqr"].quantile(0.95))
    k["max_rel_iqr"] = float(summ["rel_iqr"].max())
    k["all_checks_ok"] = bool(summ["check_ok_all"].all())
    k["bit_equal_rowblock_all"] = bool(summ[summ.method.str.startswith("rowblock") & (summ.method != "rowblock-sum")]["bit_equal_all"].all())
    bs = summ[summ.method == "blockshuffle"]
    k["max_abs_diff_blockshuffle"] = float(bs["max_abs_diff"].max()) if len(bs) else None
    k["runs_status"] = r["status"].value_counts().to_dict()
    ok = r[r.status == "OK"]
    k["largest_ok_n"] = int(ok["n"].max())
    for kind, df in (("D1", sc[sc.experiment == "D1_strong_rowblock"]), ("D2", sc[sc.experiment == "D2_strong_blockshuffle"]),
                     ("D4", sc[sc.experiment == "D4_result_stays"]), ("D6", sc[sc.experiment == "D6_weak"]), ("D7", sc[sc.experiment == "D7_unroll4_kernel"])):
        for x in df.itertuples(index=False):
            key = f"{kind}_{x.layout}_n{x.n}"
            k[f"T_{key}"] = x.median_s
            k[f"Sseq_{key}"] = x.S_seq
            k[f"Sthr_{key}"] = x.S_threads
            k[f"S1_{key}"] = x.S_1
            k[f"E1_{key}"] = x.E_1
            k[f"Eseq_{key}"] = x.E_seq
    for x in summ[summ.experiment == "D3_layout"].itertuples(index=False):
        k[f"T_D3_{x.layout}_n{x.n}"] = x.median
    for x in summ[summ.experiment == "D5_partitions"].itertuples(index=False):
        k[f"T_D5_P{x.partitions}"] = x.median
        k[f"D5_deser_ms_P{x.partitions}"] = x.med_deserialize_ms
    for x in summ[(summ.experiment == "L1_local")].itertuples(index=False):
        k[f"T_local_{x.method}_{x.kernel}_p{x.threads}_n{x.n}"] = x.median
        k[f"mem_local_{x.method}_{x.kernel}_p{x.threads}_n{x.n}_kib"] = x.max_driver_vmhwm_kib
    for x in summ[summ.experiment == "D1_strong_rowblock"].itertuples(index=False):
        k[f"mem_exec_D1_{x.layout}_n{x.n}_kib"] = x.max_peak_exec_vmhwm_kib
        k[f"mem_driver_D1_{x.layout}_n{x.n}_kib"] = x.max_driver_vmhwm_kib
        k[f"tx_D1_{x.layout}_n{x.n}"] = x.med_driver_tx_bytes
        k[f"rx_D1_{x.layout}_n{x.n}"] = x.med_driver_rx_bytes
        k[f"part_D1_{x.layout}_n{x.n}"] = x.med_t_partition_ns / 1e9
        k[f"bcast_D1_{x.layout}_n{x.n}"] = x.med_t_broadcast_ns / 1e9
        k[f"job_D1_{x.layout}_n{x.n}"] = x.med_t_job_ns / 1e9
        k[f"asm_D1_{x.layout}_n{x.n}"] = x.med_t_assemble_ns / 1e9
        k[f"kern_D1_{x.layout}_n{x.n}"] = x.med_sum_kernel_ns / 1e9
        k[f"fetch_D1_{x.layout}_n{x.n}"] = x.med_sum_fetch_ns / 1e9
        k[f"gc_D1_{x.layout}_n{x.n}"] = x.med_gc_ms / 1e3
    for x in summ[summ.experiment == "D2_strong_blockshuffle"].itertuples(index=False):
        k[f"shuf_remote_D2_{x.layout}_n{x.n}"] = x.med_shuffle_remote_bytes
        k[f"shuf_local_D2_{x.layout}_n{x.n}"] = x.med_shuffle_local_bytes
        k[f"shuf_write_D2_{x.layout}_n{x.n}"] = x.med_shuffle_write_bytes
        k[f"spill_D2_{x.layout}_n{x.n}"] = x.med_spill_disk_bytes
        k[f"part_D2_{x.layout}_n{x.n}"] = x.med_t_partition_ns / 1e9
        k[f"job_D2_{x.layout}_n{x.n}"] = x.med_t_job_ns / 1e9
        k[f"kern_D2_{x.layout}_n{x.n}"] = x.med_sum_kernel_ns / 1e9
        k[f"mem_exec_D2_{x.layout}_n{x.n}_kib"] = x.max_peak_exec_vmhwm_kib
        k[f"gc_D2_{x.layout}_n{x.n}"] = x.med_gc_ms / 1e3
    if ref is not None:
        for x in ref.itertuples(index=False):
            k[f"T_a3native_{x.method}_p{x.threads}_n{x.n}"] = x.median_s

    # figures and tables
    fig_strong(sc, lk, "D1_strong_rowblock", "fig_strong_rowblock", "D1: strong scaling, row bands + broadcast of B (blocked kernel)", [512, 1024, 2048, 3072])
    fig_strong(sc, lk, "D2_strong_blockshuffle", "fig_strong_blockshuffle", "D2: strong scaling, 4x4 blocks with two shuffles", [512, 1024, 2048])
    fig_breakdown(summ, "D1_strong_rowblock", "fig_breakdown", [1024, 2048, 3072], ["1x1", "2x1", "4x1", "4x2"])
    fig_layout(summ, "fig_layout")
    k["weak_efficiency"] = fig_weak(summ, "fig_weak")
    fig_partitions(summ, "fig_partitions")
    fig_result(sc, summ, "fig_result")
    fig_kernel(sc, lk, "fig_kernel")
    fig_baselines(summ, lk, ref, "fig_baselines")
    fig_memory(summ, "fig_memory")
    k["anchor"] = fig_drift(summ, "fig_drift")
    tables(summ, sc, lk)
    json.dump(k, open(os.path.join(RES, "key_numbers.json"), "w"), indent=1, default=float)
    print("analysis done:", len(summ), "configurations,", k["n_samples_measured"], "measured samples")


if __name__ == "__main__":
    main()
