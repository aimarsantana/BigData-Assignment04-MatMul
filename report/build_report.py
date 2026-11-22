#!/usr/bin/env python3
"""Assemble the Assignment 4 report (HTML, figures embedded) and print it to PDF with Chrome/Edge.

Tables come from report/generated/*.html and figures from figures/*.png (both written by scripts/analyze.py). The narrative
quotes numbers from results/key_numbers.json, results/validation.csv and results/environment.txt; numbers written literally
in the text were read from those files and from the generated tables.

Usage: python report/build_report.py [--no-pdf]
"""
import argparse
import base64
import csv
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GEN = os.path.join(ROOT, "report", "generated")
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
OUT_HTML = os.path.join(ROOT, "report", "Assignment04_AlejandroSantana_Report.html")
OUT_PDF = os.path.join(ROOT, "report", "Assignment04_AlejandroSantana_Report.pdf")
REPO_URL = "https://github.com/aimarsantana/BigData-Assignment04-MatMul"

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm;
        @bottom-center { content: counter(page) " / " counter(pages); font: 8pt Georgia, serif; color: #555; } }
body { font: 10pt/1.38 Georgia, 'Times New Roman', serif; color: #111; background: #fff; max-width: 100%; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 4pt; line-height: 1.2; }
.sub { text-align: center; font-size: 11pt; margin: 2pt 0; }
.auth { text-align: center; font-size: 9.5pt; margin: 8pt 0 12pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 4pt; border-bottom: 0.6pt solid #888; padding-bottom: 1pt; break-after: avoid; }
h3 { font-size: 10.5pt; margin: 10pt 0 3pt; break-after: avoid; }
p { margin: 4pt 0; text-align: justify; hyphens: auto; }
.abstract { margin: 8pt 22pt; font-size: 9.5pt; }
table { border-collapse: collapse; margin: 8pt auto; font: 8.5pt/1.25 'Segoe UI', Arial, sans-serif; break-inside: avoid; }
table.small { font-size: 7.6pt; }
caption { caption-side: top; font: 8.5pt/1.3 Georgia, serif; text-align: left; margin-bottom: 3pt; font-weight: normal; }
th, td { border-top: 0.5pt solid #888; border-bottom: 0.5pt solid #888; padding: 2pt 5pt; text-align: right; }
th { background: #f0f0f0; text-align: center; }
td:first-child, th:first-child { text-align: center; }
.iqr { color: #777; font-size: 0.88em; }
figure { margin: 8pt 0; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 8.5pt; text-align: left; margin-top: 2pt; }
code { font: 8.8pt Consolas, 'Courier New', monospace; }
ul, ol { margin: 3pt 0 3pt 16pt; padding: 0; } li { margin: 1.5pt 0; text-align: justify; }
.env td { text-align: left; } .env td:first-child { font-weight: bold; }
.ref li { text-align: left; font-size: 8.8pt; }
"""


def fig(name, caption, width="100%"):
    data = base64.b64encode(open(os.path.join(FIG, name + ".png"), "rb").read()).decode()
    return f'<figure><img style="width:{width}" src="data:image/png;base64,{data}"><figcaption>{caption}</figcaption></figure>'


def table(name):
    return open(os.path.join(GEN, name), encoding="utf8").read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    k = json.load(open(os.path.join(RES, "key_numbers.json")))
    val = list(csv.DictReader(open(os.path.join(RES, "validation.csv"), encoding="utf-8-sig")))
    n_val = len(val)
    n_pass = sum(1 for r in val if r["result"] == "PASS")

    T = lambda e, l, n: k[f"T_{e}_{l}_n{n}"]            # noqa: E731  end-to-end median (s)
    S = lambda e, l, n: k[f"Sseq_{e}_{l}_n{n}"]          # noqa: E731  speed-up over sequential Java
    St = lambda e, l, n: k[f"Sthr_{e}_{l}_n{n}"]         # noqa: E731
    S1 = lambda e, l, n: k[f"S1_{e}_{l}_n{n}"]           # noqa: E731
    E1 = lambda e, l, n: k[f"E1_{e}_{l}_n{n}"]           # noqa: E731
    L = lambda m, kn, p, n: k[f"T_local_{m}_{kn}_p{p}_n{n}"]   # noqa: E731
    A3 = lambda m, p, n: k[f"T_a3native_{m}_p{p}_n{n}"]        # noqa: E731
    we = k["weak_efficiency"]
    anc = k["anchor"]
    mib = lambda x: x / 1024                                   # noqa: E731

    # sequential speed-ups of in-JVM threads (blocked), for the comparison with Spark
    thr = {(p, n): L("seq", "blocked", 1, n) / L("threads", "blocked", p, n) for p in (1, 2, 4, 8, 16) for n in (1024, 2048, 3072)}
    comp_fraction_4x2_2048 = k["kern_D1_4x2_n2048"] / 8 / k["job_D1_4x2_n2048"]
    infl = k["kern_D1_4x2_n2048"] / k["kern_D1_1x1_n2048"]

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Assignment 4 - Distributed Matrix Multiplication</title><style>{CSS}</style></head><body>

<h1>How Far Does Apache Spark Scale a Dense Matrix Product? Computation, Communication and Framework Overhead on a Local Container Cluster</h1>
<div class="sub">Big Data &ndash; Individual Assignment 4: Distributed Matrix Multiplication</div>
<div class="auth"><b>Aimar Daniel Alejandro Santana</b><br>Grado en Ciencia e Ingenier&iacute;a de Datos, Universidad de Las Palmas de Gran Canaria<br>
Repository: {REPO_URL}<br>November 2026</div>

<div class="abstract"><b>Abstract.</b> We study how a dense binary64 matrix multiplication implemented on Apache Spark 3.5.6 (Java, RDD API) scales with the number of executors
and cores, and how much of the end-to-end time is computation and how much is framework overhead. Two algorithms are compared: row bands of A with a broadcast of B
(<i>rowblock</i>) and 4&times;4 blocks combined with two shuffles (<i>blockshuffle</i>). The cluster is <b>local</b>: a master, up to four worker containers and a driver container
run on one laptop (AMD Ryzen 7 7730U, 8 cores / 16 threads; Docker Desktop with 6.7&nbsp;GiB), so the study concerns framework behaviour and process overhead, not
network performance between machines. All {k['n_configurations']} configurations (n&nbsp;=&nbsp;512&ndash;3072, 1&ndash;8 cores, 1&ndash;4 executors) reproduce the reference result
(rowblock bit for bit, blockshuffle within {k['max_abs_diff_blockshuffle']:.1e}). Against a sequential Java kernel, the best distributed run at n&nbsp;=&nbsp;3072 is
{L('seq', 'blocked', 1, 3072) / k['T_D3_2x4_n3072']:.2f}&times; faster with 8 cores (2 executors &times; 4 cores) and {S('D1', '4x2', 3072):.2f}&times; with 4 executors &times; 2 cores, whereas 8 threads in one JVM reach {thr[(8, 3072)]:.2f}&times; and the C/OpenMP
program of Assignment&nbsp;3, on this machine, {A3('seq', 1, 3072) / A3('par', 8, 3072):.2f}&times;. For n&nbsp;&le;&nbsp;1024 the distributed run is not faster than the sequential one (S&nbsp;=&nbsp;{S('D1', '4x2', 512):.2f} at n&nbsp;=&nbsp;512).
The overhead is the broadcast of B to every executor (task-time spent fetching B grows from {k['fetch_D1_1x1_n2048']:.2f} to {k['fetch_D1_4x2_n2048']:.1f}&nbsp;s at n&nbsp;=&nbsp;2048), the kernel running {infl:.1f}&times; slower per core when 8 cores are busy
(shared cache and memory bandwidth), and task scheduling; the weak-scaling efficiency falls to {we['8']:.2f} at 8 cores. Block-shuffle is slower than broadcast at every size.
Conclusions are limited to this machine, this Spark configuration and these sizes.</div>

<h2>1. Introduction and objective</h2>
<p>Assignments 1&ndash;3 made one dense product faster on one machine: a triple loop in three languages, cache blocking and unrolling, and shared-memory threads. A distributed framework
offers a different promise: add workers and the same job finishes sooner, on data that no longer has to fit on one machine. The price is a runtime that must partition the matrices, ship them,
schedule tasks, serialise data and collect results. This study measures that trade-off for the same arithmetic. Matrices are n&times;n; n is the side length; C is the total number of cores of the cluster.</p>
<p><b>Research question.</b> <i>How does a dense binary64 matrix multiplication on a Spark standalone cluster scale with the number of executors and cores (strong and weak scaling), how much of the end-to-end time is computation and how much is framework overhead
(partitioning, broadcast, scheduling, serialisation, shuffle, result collection), and how does it compare with a sequential and with a shared-memory parallel implementation on the same machine?</i></p>
<p><b>Predictions</b> (written down after a short pilot at n&nbsp;=&nbsp;2048 with the kernel of Assignment&nbsp;3, which showed almost no speed-up on 8 cores and motivated the blocked kernel and experiment D7):</p>
<ol>
<li><b>H1.</b> For small matrices (n&nbsp;&le;&nbsp;1024) framework overhead (tens to hundreds of milliseconds of scheduling, serialisation and broadcast) dominates and the distributed run is not faster than a sequential run; for n&nbsp;&ge;&nbsp;2048 computation dominates and the speed-up grows with n.</li>
<li><b>H2.</b> Strong scaling is sub-linear: the efficiency falls with the number of cores because B is copied to every executor, tasks compete for the shared last-level cache and DRAM of the single machine, and the driver is a serial hub.</li>
<li><b>H3.</b> For the same number of cores, fewer and larger executors are faster than many small ones, because B is broadcast once per executor, not once per core.</li>
<li><b>H4.</b> The 2-D block algorithm, which needs two shuffles, is slower than the broadcast algorithm for these sizes, which fit in one machine&rsquo;s memory.</li>
<li><b>H5.</b> A kernel that is limited by memory bandwidth (the unroll4 kernel of Assignments 2 and 3) scales worse than a cache-blocked one, because the containers share one memory system.</li>
<li><b>H6.</b> With the work per core held constant, the time grows with C (weak-scaling efficiency below 1) because the broadcast, the A bands and the result pass through the driver.</li></ol>

<h2>2. Background</h2>
<p><b>Spark.</b> Apache Spark [1, 2, 3] runs a <i>driver</i> program that builds a graph of operations on <i>RDDs</i> (resilient distributed datasets partitioned across the cluster). An action (<code>collect</code>) submits a <i>job</i>, split into <i>stages</i> of <i>tasks</i>
(one per partition), which a <i>scheduler</i> sends to <i>executors</i> (JVM processes, each with a number of cores = task slots). In <i>standalone mode</i> a master tracks <i>workers</i> that launch the executors [2]. Data are serialised (Java serialisation by default) when they travel
from the driver to the tasks, between executors and back. A <i>broadcast variable</i> ships a read-only value to each executor once (torrent-like, in 4&nbsp;MB blocks fetched from the driver or from other executors) instead of with every task [2]. A <i>shuffle</i> repartitions data
by key: map tasks write blocks to local disk, reduce tasks fetch them, locally or over the network [2].</p>
<p><b>Distributed matrix multiplication.</b> In C&nbsp;=&nbsp;AB every row of C depends on one row of A and all of B, so partitioning A into row bands and replicating B (here by broadcast) needs no communication between tasks; its memory cost is one copy of B per executor.
The alternative partitions both matrices into g&times;g blocks and computes C<sub>ij</sub>&nbsp;=&nbsp;&Sigma;<sub>k</sub>&nbsp;A<sub>ik</sub>B<sub>kj</sub>; each block is needed by g products, which requires moving blocks to the products (a join, i.e. a shuffle) and summing the partial products (a second shuffle), as in SUMMA [6]. It scales to matrices that do not fit on one node, at the price of more communication.</p>
<p><b>Scaling and limits.</b> <i>Strong scaling</i> keeps the problem fixed while resources grow; speed-up is S(C)&nbsp;=&nbsp;T<sub>ref</sub>/T(C) and efficiency E&nbsp;=&nbsp;S/C. Amdahl&rsquo;s law [4] bounds S by the serial fraction (here: building and sending data, the driver, scheduling).
<i>Weak scaling</i> grows the problem with the resources [5]; for the cubic work of dense multiplication, work per core is constant when n<sup>3</sup>&nbsp;&prop;&nbsp;C, i.e. n&nbsp;&prop;&nbsp;C<sup>1/3</sup>, not when n per worker is fixed. The roofline model [7] bounds the throughput of a kernel by its compute peak or by bandwidth&nbsp;&times;&nbsp;arithmetic intensity;
unblocked kernels have a low intensity and become memory-bound for large n [8], and all containers of a local cluster share one memory system.</p>

<h2>3. Methodology</h2>
<h3>3.1 Implementations</h3>
<table class="env"><caption>Table 1. Methods (Java 17 release, <code>src/main/java/bd/matmul4</code>; own code; the only library is Spark for the distribution; no numerical library). All use flat row-major <code>double[]</code> arrays.</caption>
<thead><tr><th>Method</th><th>Description</th></tr></thead><tbody>
<tr><td>ijk</td><td>Basic triple loop of Assignment 1 (local, context only, n&nbsp;&le;&nbsp;1536).</td></tr>
<tr><td>seq</td><td><b>Sequential baseline</b>: one thread, one JVM, no Spark. Kernel <i>blocked</i> (main) or <i>unroll4</i> (kernel of Assignments 2 and 3: ikj order, k unrolled by 4).</td></tr>
<tr><td>threads</td><td><b>Shared-memory parallel baseline</b>: p threads of one JVM (persistent pool), p static row bands of C, same kernel.</td></tr>
<tr><td>rowblock</td><td><b>Distributed, broadcast</b>: A is cut into 4C row bands (the partitions of an RDD created on the driver); B is a broadcast variable; each task multiplies its band by B and returns its band of C; the driver assembles C. Variant <i>rowblock-sum</i>: tasks return only the sum of their band (the result stays on the executors).</td></tr>
<tr><td>blockshuffle</td><td><b>Distributed, 2-D blocks</b>: A and B cut into 4&times;4 blocks; blocks A<sub>ik</sub> and B<sub>kj</sub> are joined by k (shuffle 1), multiplied (64 block products) and the partial blocks of C<sub>ij</sub> are summed with <code>reduceByKey</code> (shuffle 2), then collected on the driver.</td></tr></tbody></table>
<p><b>Kernels.</b> <i>unroll4</i> is the kernel of Assignments 2 and 3 translated to Java. <i>blocked</i> adds the cache-blocking idea of Assignment 2: tiles of 64&times;256 of B (128&nbsp;KiB) are reused by all rows of a band before the next tile is loaded. In every kernel, each C[i][j] is the sum of the products added one after the other in increasing k starting from 0, and Java never fuses multiply-add,
so <b>all kernels, all partitionings and the basic ijk loop return bit-identical results</b> (the block algorithm sums partial blocks, which changes the order, and is checked with a tolerance). The blocked kernel was chosen for the main study because with unroll4 a pilot showed that the memory system, not the framework, set the limit (experiment D7).</p>
<p><b>Partitioning, placement and result collection.</b> The driver reads A and B (outside the timer), cuts A into bands and creates the RDD (<code>parallelize</code>, bands travel inside the tasks), broadcasts B, and runs one job whose tasks are placed by the standalone scheduler on any free core (no data locality exists for driver-created data). Results return to the driver, which assembles C in memory (variant rowblock-sum: only checksums). Nothing is written to disk by the application (shuffle files excepted).</p>

<h3>3.2 Inputs</h3>
<p>The matrices of Assignments 1&ndash;3: NumPy PCG64 (<code>default_rng</code>), seeds 20261012 (A) and 20261013 (B), U[&minus;1,1), binary64, little-endian row-major files, n&nbsp;&isin;&nbsp;{{512, 1024, 1536, 2048, 3072}} plus the two weak-scaling sizes 1290 and 1626 generated with the same recipe. The SHA-256 of every regenerated file equals that of the manifests of Assignments 1 and 3 where a size exists in them (<code>data/inputs/manifest.json</code>).</p>

<h3>3.3 Environment (local, single host)</h3>
<table class="env"><caption>Table 2. Measurement environment (<code>results/environment.txt</code>). <b>This is not the machine of Assignments 1&ndash;3</b> (a desktop i7-13700K): every comparison with those studies is made with programs re-run here.</caption><tbody>
<tr><td>Machine</td><td>Lenovo 82YY laptop (plugged in), AMD Ryzen 7 7730U (8 cores, 16 threads, 2.0&nbsp;GHz base, L2 4&nbsp;MB, L3 16&nbsp;MB), 13.8&nbsp;GB RAM (2&times;8&nbsp;GB DDR4-3200), Windows 11 Home build 26200, &ldquo;Balanced&rdquo; power plan. Turbo and temperature not controlled.</td></tr>
<tr><td>Container platform</td><td>Docker Desktop 29.8.0 (WSL2 virtual machine: 16 vCPUs, 6.7&nbsp;GiB), Compose 5.5.1, one bridge network. <b>All containers share the same CPU, memory and operating-system kernel: workers &ne; machines.</b></td></tr>
<tr><td>Spark</td><td>Image <code>apache/spark:3.5.6-java17</code> (Spark 3.5.6, Scala 2.12, Temurin JDK 17.0.15), standalone mode: master (128&nbsp;MB daemon), N worker containers with one executor each (heap 640&nbsp;MB, 1 or 2 cores), driver container (heap 1&nbsp;GB, client mode). Defaults: Java serialiser, LZ4 compression of broadcast and shuffle data, 4 tasks per core, UI disabled, <code>spark.rpc.message.maxSize</code>&nbsp;=&nbsp;512&nbsp;MB.</td></tr>
<tr><td>Build</td><td>JDK 21.0.12 compiling with <code>--release 17</code>, Maven 3.9.16; Python 3.14, NumPy 2.5.5, pandas 3.1.1, matplotlib 3.11.2 (inputs, validation, analysis only).</td></tr>
<tr><td>Why only 4 containers</td><td>The Docker VM has 6.7&nbsp;GiB. A first test with 8 single-core worker containers (executor heap 512&nbsp;MB) used 5.7&nbsp;GiB, and runs at n&nbsp;=&nbsp;2048 took 6&ndash;14&nbsp;s instead of about 1&nbsp;s (memory pressure); this is reported as a <b>memory limit of the environment</b>, not as a result. The layouts therefore use up to 4 containers, and 8 cores are reached with 2 cores per container.</td></tr>
</tbody></table>
<p><b>Workers, processes and machines.</b> A <i>layout</i> E&times;c is E worker containers (each a worker process plus one executor JVM) with c cores each; C&nbsp;=&nbsp;E&middot;c. There is one physical machine. The WSL2 virtual machine exposes all 16 hardware threads, so with C&nbsp;&le;&nbsp;8 task slots the executors can run on 8 distinct physical cores, but the hypervisor and the Windows scheduler decide where; nothing is pinned.
<b>Because the cluster is local, the results characterise framework behaviour and process overhead; they say nothing about inter-machine network latency or bandwidth (the bridge network is in-memory) or about memory capacity beyond one machine.</b></p>
<p><b>Experiments</b> (<code>config/benchmark_config.json</code>): <b>D1</b> strong scaling of rowblock (layouts 1&times;1, 2&times;1, 4&times;1, 4&times;2; n&nbsp;=&nbsp;512&ndash;3072); <b>D2</b> the same for blockshuffle (n&nbsp;&le;&nbsp;2048); <b>D3</b> six layouts with 4 or 8 cores; <b>D4</b> D1 with the result left on the executors; <b>D5</b> 8&ndash;256 tasks at 4&times;2, n&nbsp;=&nbsp;2048; <b>D6</b> weak scaling n&nbsp;=&nbsp;1024, 1290, 1626, 2048 (n<sup>3</sup> doubles with C); <b>D7</b> unroll4 kernel; <b>L1</b> local baselines (ijk, seq, threads&nbsp;1&ndash;16) run in the driver container with the cluster stopped; <b>anchor</b> runs of the sequential kernel at the start, middle and end.
Each cluster layout is started once and runs its experiments as separate applications.</p>

<h3>3.4 Correctness</h3>
<p><code>tests/validate.py</code> ran {n_val} checks before benchmarking (<b>{n_pass} pass, {n_val - n_pass} fail</b>): a hand-computed 3&times;3 product, identity (left and right) and zero matrices, random matrices of size 1, 2, 3, 5, 7, 33, 100, 127 and 130 against the naive triple loop, with 1, 3, 8 and 40 partitions (more partitions than rows) and 1&ndash;5 blocks per side (more blocks than rows), for both algorithms and both kernels;
and the benchmark inputs at n&nbsp;=&nbsp;512 and 1024 multiplied on a real 2-container cluster and compared with NumPy (OpenBLAS, used only as a reference) with |&Delta;C|&nbsp;&le;&nbsp;10<sup>&minus;9</sup>&nbsp;+&nbsp;10<sup>&minus;9</sup>|C|, above the rounding bound n&sup2;u&nbsp;&asymp;&nbsp;1.2&times;10<sup>&minus;10</sup> at n&nbsp;=&nbsp;1024 [9]. In addition, <b>every measured run</b> is compared, outside the timer, with the reference product of the sequential kernel: all {k['n_samples_measured']} measured samples agree
(rowblock bit for bit: {k['bit_equal_rowblock_all']}; blockshuffle: largest difference {k['max_abs_diff_blockshuffle']:.1e}; the checksum-only variant: relative tolerance 10<sup>&minus;9</sup>). No data race is possible in the kernels: each task writes only its own output.</p>

<h3>3.5 Measurement protocol</h3>
<ul>
<li><b>Timer boundary (end-to-end).</b> The inputs are already in the driver&rsquo;s memory. The timer (<code>System.nanoTime</code>, monotonic) covers: building the distributed data (A bands or blocks) and the RDD, creating the broadcast, the Spark job(s), and assembling C on the driver. <b>Outside the timer:</b> reading the input files, Spark application start-up and executor registration, the correctness check, destroying the broadcast. Local baselines (seq, threads) time only the kernel call (including zeroing C and, for threads, submitting and awaiting the band tasks). The distributed and the local times therefore differ by the data movement and framework work that the distributed version must do, which is precisely what is studied; the two are not an isolated kernel comparison.</li>
<li><b>Phases.</b> The driver reports the times of building the RDD, the broadcast, the job and the assembly (they add up to the end-to-end time). Inside the tasks the kernel time and the time to obtain B (first <code>value()</code> on the executor: fetch + deserialisation) are measured and summed over tasks; Spark&rsquo;s task metrics give executor run time, deserialisation, GC, result size and shuffle bytes. <b>These sums are task-seconds and overlap in time: they are never added to the end-to-end time.</b></li>
<li><b>Warm-up and repetitions.</b> Inside one application each (configuration, n) runs 2 warm-up runs (JIT, executor start; kept in the raw data, excluded from statistics) and then 7 measured repetitions for n&nbsp;&le;&nbsp;1024 or 5 above. Total: {k['n_samples_measured']} measured and {k['n_samples_warmup']} warm-up samples. A garbage collection is requested on the driver between runs.</li>
<li><b>Statistics.</b> Median and IQR; speed-ups and efficiencies are computed from medians: S<sub>seq</sub>&nbsp;=&nbsp;T<sub>seq</sub>/T(C) with T<sub>seq</sub> the sequential Java run of the same kernel and n; S<sub>thr</sub> with the p&nbsp;=&nbsp;C-thread JVM; S<sub>1</sub>&nbsp;=&nbsp;T(1&times;1)/T(C), E&nbsp;=&nbsp;S/C; weak-scaling efficiency E<sub>w</sub>&nbsp;=&nbsp;T(C&nbsp;=&nbsp;1,&nbsp;n&nbsp;=&nbsp;1024)/T(C,&nbsp;n<sub>C</sub>). Median relative IQR over all configurations: {100 * k['median_rel_iqr']:.1f}&nbsp;% (95th percentile {100 * k['p95_rel_iqr']:.0f}&nbsp;%, maximum {100 * k['max_rel_iqr']:.0f}&nbsp;%, at n&nbsp;=&nbsp;512 where runs last tenths of a second).</li>
<li><b>Budgets.</b> 120&nbsp;s per run and the memory of the Docker VM; the largest size is n&nbsp;=&nbsp;{k['largest_ok_n']}, chosen so that the whole campaign takes about 1.5 hours; all {k['runs_status']['OK']} configurations completed within the budget, and none was removed. The largest <i>tested</i> size is also the largest <i>attempted</i> in the final campaign; the 8-container attempt that failed for memory is described in Table 2.</li>
<li><b>Memory and network metrics.</b> Executor and driver: peak resident set size of the JVM process (<code>VmHWM</code> in <code>/proc/self/status</code> [11]), a cumulative maximum over the application (sizes run in increasing order), including the JVM, the broadcast copies and serialisation buffers. Network: bytes sent and received by the driver container on <code>eth0</code> (<code>/proc/net/dev</code>) per multiplication; executor-to-executor traffic only through Spark&rsquo;s shuffle metrics (remote/local bytes read). Per-link network time cannot be isolated here (the bridge is in memory).</li>
<li><b>Drift.</b> The sequential kernel at n&nbsp;=&nbsp;1024 was run at the start, middle and end of the campaign: {anc['start']:.3f}, {anc['mid']:.3f} and {anc['end']:.3f}&nbsp;s (end {100 * (1 - anc['end'] / anc['start']):.0f}&nbsp;% faster than start), so slow drift of the machine (frequency, temperature, background load) of the order of 15&nbsp;% exists and is a bound on how much small differences can be trusted.</li></ul>

<h2>4. Results</h2>
<h3>4.1 Baselines and strong scaling of the broadcast algorithm (D1)</h3>
<p><b>Baselines on this machine</b> (medians). Basic Java ijk loop: {L('ijk', 'ijk', 1, 1024):.2f}&nbsp;s at n&nbsp;=&nbsp;1024 and {L('ijk', 'ijk', 1, 1536):.1f}&nbsp;s at 1536; sequential Java unroll4: {L('seq', 'unroll4', 1, 1024):.2f}, {L('seq', 'unroll4', 1, 2048):.2f} and {L('seq', 'unroll4', 1, 3072):.1f}&nbsp;s at n&nbsp;=&nbsp;1024, 2048, 3072; sequential Java blocked: {L('seq', 'blocked', 1, 1024):.2f}, {L('seq', 'blocked', 1, 2048):.2f}, {L('seq', 'blocked', 1, 3072):.1f}&nbsp;s.
In one JVM, 8 threads take {L('threads', 'blocked', 8, 3072):.2f}&nbsp;s at n&nbsp;=&nbsp;3072 (S&nbsp;=&nbsp;{thr[(8, 3072)]:.2f}) and 16 threads {L('threads', 'blocked', 16, 3072):.2f}&nbsp;s (S&nbsp;=&nbsp;{thr[(16, 3072)]:.2f}, SMT); at n&nbsp;=&nbsp;1024, 8 threads give S&nbsp;=&nbsp;{thr[(8, 1024)]:.2f}. The unchanged C/OpenMP program of Assignment&nbsp;3 (scalar build, unroll4, re-run here) takes {A3('seq', 1, 2048):.2f}&nbsp;s sequentially and {A3('par', 8, 2048):.2f}&nbsp;s with 8 threads at n&nbsp;=&nbsp;2048 (S&nbsp;=&nbsp;{A3('seq', 1, 2048) / A3('par', 8, 2048):.2f}),
and {A3('seq', 1, 3072):.1f} and {A3('par', 8, 3072):.2f}&nbsp;s at 3072 (S&nbsp;=&nbsp;{A3('seq', 1, 3072) / A3('par', 8, 3072):.2f}); on the desktop of Assignment&nbsp;3 the same program reached 4.2&ndash;4.8 with 8 P-cores at these sizes, a different machine. Java and C cannot be compared as languages here (different kernels in memory layout and JIT, Windows versus a Linux VM).</p>
{fig('fig_baselines', 'Figure 1. Time versus n on this machine: local Java baselines (ijk, sequential, 8 threads), the C/OpenMP program of Assignment 3 re-run here (dash-dot), and Spark rowblock with 1 and with 8 cores (4 containers &times; 2 cores).', '62%')}
{table('table_time.html')}
{fig('fig_strong_rowblock', 'Figure 2. D1, strong scaling of rowblock (blocked kernel). Left: time (solid: Spark; dashed: Java threads in one JVM with p&nbsp;=&nbsp;C). Centre: speed-up over the sequential Java run of the same n. Right: efficiency T(1 core)/(C&middot;T(C)) of the distributed run relative to itself.')}
{table('table_speedup.html')}
<p><b>Small matrices.</b> At n&nbsp;=&nbsp;512 the distributed run is slower than sequential Java at every layout: {T('D1', '1x1', 512) * 1e3:.0f}&nbsp;ms with one core against {L('seq', 'blocked', 1, 512) * 1e3:.0f}&nbsp;ms sequentially, and it gets <em>slower</em> with more cores ({T('D1', '4x2', 512) * 1e3:.0f}&nbsp;ms with 8; S&nbsp;=&nbsp;{S('D1', '4x2', 512):.2f}). At n&nbsp;=&nbsp;1024 it is on par with the sequential run for C&nbsp;&ge;&nbsp;2
(S&nbsp;=&nbsp;{S('D1', '2x1', 1024):.2f}&ndash;{S('D1', '4x2', 1024):.2f}) and 8 threads of one JVM are {T('D1', '4x2', 1024) / L('threads', 'blocked', 8, 1024):.1f}&times; faster than Spark with 8 cores. <b>With one core Spark costs {100 * (T('D1', '1x1', 1024) / L('seq', 'blocked', 1, 1024) - 1):.0f}&nbsp;% more than sequential at n&nbsp;=&nbsp;1024 but only {100 * (T('D1', '1x1', 2048) / L('seq', 'blocked', 1, 2048) - 1):+.0f}&nbsp;% at 2048 and {100 * (T('D1', '1x1', 3072) / L('seq', 'blocked', 1, 3072) - 1):+.0f}&nbsp;% at 3072</b>: the fixed cost of the framework is about 0.2&nbsp;s.
<b>Large matrices.</b> For n&nbsp;=&nbsp;2048 and 3072 the distributed run gets faster with the cores: n&nbsp;=&nbsp;3072 takes {T('D1', '1x1', 3072):.1f}, {T('D1', '2x1', 3072):.1f}, {T('D1', '4x1', 3072):.1f} and {T('D1', '4x2', 3072):.1f}&nbsp;s with 1, 2, 4 and 8 cores (S<sub>seq</sub>&nbsp;=&nbsp;{S('D1', '2x1', 3072):.2f}, {S('D1', '4x1', 3072):.2f}, {S('D1', '4x2', 3072):.2f}); n&nbsp;=&nbsp;2048: {T('D1', '1x1', 2048):.2f}, {T('D1', '2x1', 2048):.2f}, {T('D1', '4x1', 2048):.2f}, {T('D1', '4x2', 2048):.2f}&nbsp;s.
The efficiency decreases with C at every size (E<sub>1</sub>&nbsp;=&nbsp;{E1('D1', '2x1', 3072):.2f}, {E1('D1', '4x1', 3072):.2f}, {E1('D1', '4x2', 3072):.2f} at n&nbsp;=&nbsp;3072), and from 4 to 8 cores the gain is almost nil at n&nbsp;=&nbsp;3072 ({T('D1', '4x1', 3072):.2f}&nbsp;s versus {T('D1', '4x2', 3072):.2f}&nbsp;s, within the IQR). The shared-memory threads of one JVM do better (S<sub>thr</sub>&nbsp;&lt;&nbsp;1 in Table 4: Spark is {1 / St('D1', '4x2', 3072):.1f}&times; slower than 8 threads at n&nbsp;=&nbsp;3072).</p>

<h3>4.2 Where the time goes</h3>
{fig('fig_breakdown', 'Figure 3. D1: phases of the end-to-end time (medians). &ldquo;compute&rdquo; is the sum of the kernel times of all tasks divided by C, i.e. the time a perfectly packed job would need; the rest of the job phase is scheduling, fetching and deserialising B, serialising and transferring results, and idle time at the end of the stage.')}
{table('table_breakdown.html')}
<p>Building the RDD, broadcasting and assembling take {k['part_D1_4x2_n3072'] + k['bcast_D1_4x2_n3072'] + k['asm_D1_4x2_n3072']:.2f}&nbsp;s at n&nbsp;=&nbsp;3072 with 8 cores (about {100 * (k['part_D1_4x2_n3072'] + k['bcast_D1_4x2_n3072'] + k['asm_D1_4x2_n3072']) / T('D1', '4x2', 3072):.0f}&nbsp;% of the total, serial on the driver), the creation of the broadcast is cheap ({k['bcast_D1_4x2_n2048'] * 1e3:.0f}&nbsp;ms for 32&nbsp;MiB at n&nbsp;=&nbsp;2048) because executors fetch the data later, inside the tasks.
The job phase dominates. Three effects explain why it does not shrink as 1/C:
(i) <b>the kernel gets slower per core</b>: the sum of the task kernel times at n&nbsp;=&nbsp;2048 is {k['kern_D1_1x1_n2048']:.1f}&nbsp;task-s with 1 core and {k['kern_D1_4x2_n2048']:.1f}&nbsp;with 8, i.e. {infl:.1f}&times; more CPU time for the same arithmetic, consistent with the cores sharing the last-level cache (16&nbsp;MB), DRAM bandwidth and the power budget of a 15&nbsp;W laptop CPU (SMT siblings and clocks were not controlled), and with the in-JVM threads experiment of Section 4.1 (8 threads reach {thr[(8, 2048)]:.1f}, not 8);
(ii) <b>B is fetched and deserialised by every executor</b>: the task-seconds spent obtaining B grow from {k['fetch_D1_1x1_n2048']:.2f} (1 executor) to {k['fetch_D1_2x1_n2048']:.2f}, {k['fetch_D1_4x1_n2048']:.2f} and {k['fetch_D1_4x2_n2048']:.2f} (4 executors, 8 task slots; n&nbsp;=&nbsp;2048) and to {k['fetch_D1_4x2_n3072']:.1f}&nbsp;s at 3072;
(iii) <b>scheduling, result transfer and stage tail</b>: the part of the job not explained by the kernel is {k['job_D1_4x2_n2048'] - k['kern_D1_4x2_n2048'] / 8:.2f}&nbsp;s at n&nbsp;=&nbsp;2048 on 8 cores, {100 * (1 - comp_fraction_4x2_2048):.0f}&nbsp;% of the job. GC is small (at most {k['gc_D1_4x2_n3072']:.1f}&nbsp;task-s at 3072).
At n&nbsp;=&nbsp;1024 this non-compute part ({k['job_D1_4x2_n1024'] - k['kern_D1_4x2_n1024'] / 8:.2f}&nbsp;s with 8 cores) is larger than the compute ({k['kern_D1_4x2_n1024'] / 8:.2f}&nbsp;s), which is why the run does not benefit from more cores. Intervals overlap (a task&rsquo;s fetch precedes its kernel; the driver collects results while other tasks run), so they are not additive.</p>

<h3>4.3 The block algorithm with shuffles (D2)</h3>
{fig('fig_strong_blockshuffle', 'Figure 4. D2, strong scaling of blockshuffle (4&times;4 blocks). Same panels as Figure 2 (no thread baseline).')}
{table('table_blockshuffle.html')}
<p>Blockshuffle is slower than rowblock at every size and layout (n&nbsp;=&nbsp;2048, 8 cores: {T('D2', '4x2', 2048):.2f}&nbsp;s versus {T('D1', '4x2', 2048):.2f}&nbsp;s; with one core {T('D2', '1x1', 2048):.2f} versus {T('D1', '1x1', 2048):.2f}&nbsp;s), scales with a similar efficiency
(S<sub>1</sub>&nbsp;=&nbsp;{S1('D2', '4x2', 2048):.2f} at 8 cores, n&nbsp;=&nbsp;2048, against {S1('D1', '4x2', 2048):.2f}), and from 4 to 8 cores it does not improve ({T('D2', '4x1', 2048):.2f} versus {T('D2', '4x2', 2048):.2f}&nbsp;s). Its extra cost is communication that rowblock does not have: at n&nbsp;=&nbsp;2048 the two shuffles write {k['shuf_write_D2_4x2_n2048'] / 2 ** 20:.0f}&nbsp;MiB
(A and B blocks: 64&nbsp;MiB, then 64 partial products of 2&nbsp;MiB: 128&nbsp;MiB); with 4 executors {k['shuf_remote_D2_4x2_n2048'] / 2 ** 20:.0f}&nbsp;MiB of them are read from another executor ({100 * k['shuf_remote_D2_4x2_n2048'] / (k['shuf_remote_D2_4x2_n2048'] + k['shuf_local_D2_4x2_n2048']):.0f}&nbsp;%, the 3/4 expected for uniformly placed data), and with one executor all {k['shuf_local_D2_1x1_n2048'] / 2 ** 20:.0f}&nbsp;MiB are local; nothing spilled to disk.
Because the bridge network is in memory, this volume is serialisation, copying and file I/O rather than wire time. The advantage of the 2-D algorithm, working with matrices larger than one executor&rsquo;s memory (an extension not tested here), is not exercised at these sizes.</p>

<h3>4.4 Containers versus cores per container (D3)</h3>
{fig('fig_layout', 'Figure 5. D3: end-to-end time for the same number of cores split differently (ExC = E executor containers with C cores each). Bars: median; whiskers: quartiles.')}
<table class="small"><caption>Table 7. D3: median end-to-end time (s) and speed-up over sequential Java for 4 and 8 cores.</caption><thead><tr><th>n</th><th>C&nbsp;=&nbsp;4: 1&times;4</th><th>2&times;2</th><th>4&times;1</th><th>C&nbsp;=&nbsp;8: 1&times;8</th><th>2&times;4</th><th>4&times;2</th></tr></thead><tbody>
""" + "".join(
        f"<tr><td>{n}</td>" + "".join(f"<td>{k[f'T_D3_{l}_n{n}']:.2f} s ({L('seq', 'blocked', 1, n) / k[f'T_D3_{l}_n{n}']:.2f}&times;)</td>" for l in ("1x4", "2x2", "4x1", "1x8", "2x4", "4x2")) + "</tr>"
        for n in (1024, 2048, 3072)) + f"""</tbody></table>
<p>For the same total cores, fewer and larger executors are faster at n&nbsp;&ge;&nbsp;2048: with 4 cores, 1&times;4 takes {k['T_D3_1x4_n2048']:.2f}&nbsp;s and 4&times;1 {k['T_D3_4x1_n2048']:.2f}&nbsp;s (n&nbsp;=&nbsp;2048); with 8 cores, 1&times;8 takes {k['T_D3_1x8_n2048']:.2f}&nbsp;s against {k['T_D3_4x2_n2048']:.2f}&nbsp;s for 4&times;2 ({k['T_D3_4x2_n2048'] / k['T_D3_1x8_n2048']:.2f}&times;). At n&nbsp;=&nbsp;3072 the best layout is 2&times;4 ({k['T_D3_2x4_n3072']:.2f}&nbsp;s, S<sub>seq</sub>&nbsp;=&nbsp;{L('seq', 'blocked', 1, 3072) / k['T_D3_2x4_n3072']:.2f}), and differences between layouts are smaller than between sizes.
The reason is the replication of B and of its deserialisation, once per executor (Section 4.2): one executor with many cores shares one copy. The price of the single large executor is no isolation (one JVM, one failure domain) and a larger heap. A single machine with a single JVM with 8 threads remains the fastest of all ({L('threads', 'blocked', 8, 3072):.2f}&nbsp;s at n&nbsp;=&nbsp;3072), which is the cost of a general framework for a problem that fits in a machine.
Differences below about 15&nbsp;% (the drift of Section 3.5 and the IQRs) should not be over-interpreted; at n&nbsp;=&nbsp;1024 the ranking is noisy.</p>

<h3>4.5 Result collection and number of tasks (D4, D5)</h3>
{fig('fig_result', 'Figure 6. D4 versus D1: rowblock returning C to the driver (blue) or only checksums (orange).', '90%')}
{fig('fig_partitions', 'Figure 7. D5: effect of the number of tasks (row bands) with 4 containers &times; 2 cores, n = 2048. Left: end-to-end and job time. Right: bytes returned to the driver and total task deserialisation time.', '78%')}
<p><b>Result collection.</b> Returning C to the driver costs little here: the 72&nbsp;MiB of C at n&nbsp;=&nbsp;3072 cross the in-memory network in the job phase and the assembly takes {k['asm_D1_4x2_n3072'] * 1e3:.0f}&nbsp;ms; with the result left on the executors the times are {T('D4', '4x2', 3072):.2f}&nbsp;s versus {T('D1', '4x2', 3072):.2f}&nbsp;s (8 cores), {T('D4', '4x1', 3072):.2f} versus {T('D1', '4x1', 3072):.2f}&nbsp;s (4 cores): differences of the order of 5&nbsp;%, within the drift and IQR. In a real network the 8n&sup2; bytes returned
(and the 16n&sup2; sent) would weigh more; this cannot be assessed with a local cluster.
<b>Granularity.</b> With 8 cores the time is flat for 8&ndash;128 tasks ({k['T_D5_P8']:.2f}&ndash;{k['T_D5_P128']:.2f}&nbsp;s, best {k['T_D5_P64']:.2f}&nbsp;s with 64) and worse with 256 tasks ({k['T_D5_P256']:.2f}&nbsp;s), where the per-task cost (task launch, deserialisation of the closure and of the band, {k['D5_deser_ms_P256'] / 1e3:.2f}&nbsp;task-s against {k['D5_deser_ms_P8'] / 1e3:.2f} with 8 tasks) accumulates. The default of 4 tasks per core (32 tasks) is a sensible choice here; more tasks than cores help balance load on uneven cores but each costs a few milliseconds.</p>

<h3>4.6 Weak scaling (D6)</h3>
{fig('fig_weak', 'Figure 8. D6: weak scaling, rowblock; n = 1024, 1290, 1626, 2048 for C = 1, 2, 4, 8 so that the work n^3 is proportional to C. Left: time (ideal: flat). Right: efficiency T(1)/T(C).', '78%')}
<p>With n<sup>3</sup> proportional to C the time grows from {T('D6', '1x1', 1024):.2f}&nbsp;s (C&nbsp;=&nbsp;1) to {T('D6', '2x1', 1290):.2f}, {T('D6', '4x1', 1626):.2f} and {T('D6', '4x2', 2048):.2f}&nbsp;s (C&nbsp;=&nbsp;2, 4, 8): the weak-scaling efficiency is {we['2']:.2f}, {we['4']:.2f} and {we['8']:.2f}. This is much lower than in an ideal system and consistent with Section 4.2: the work per core is constant, but the data that every executor must receive (B, 8n&sup2; bytes) and the driver-side work grow with n and with the number of executors,
and the kernel slows down per core when more cores are busy. The first point (n&nbsp;=&nbsp;1024) is also the one where the fixed overhead is largest relative to the work, which flatters the efficiency of the larger runs slightly.</p>

<h3>4.7 The kernel of Assignments 2 and 3 (D7)</h3>
{fig('fig_kernel', 'Figure 9. n = 2048: time and speed-up of Spark rowblock and of Java threads in one JVM with the blocked kernel (blue) and the unroll4 kernel of Assignments 2 and 3 (orange), each relative to its own sequential kernel.', '88%')}
{table('table_unroll4.html')}
<p>With the unroll4 kernel Spark does not scale: at n&nbsp;=&nbsp;2048 the run takes {T('D7', '1x1', 2048):.2f}, {T('D7', '2x1', 2048):.2f}, {T('D7', '4x1', 2048):.2f} and {T('D7', '4x2', 2048):.2f}&nbsp;s with 1, 2, 4 and 8 cores, i.e. <b>8 cores are slower than 2</b> and no faster than the sequential run (S<sub>seq</sub>&nbsp;=&nbsp;{S('D7', '4x2', 2048):.2f}); at n&nbsp;=&nbsp;1024 it is {T('D7', '4x2', 1024) / T('D7', '1x1', 1024):.1f}&times; <em>slower</em> with 8 cores than with one.
Eight threads of one JVM reach only {L('seq', 'unroll4', 1, 2048) / L('threads', 'unroll4', 8, 2048):.2f}&times; with the same kernel (against {thr[(8, 2048)]:.2f}&times; for the blocked one), and the C program of Assignment&nbsp;3 {A3('seq', 1, 3072) / A3('par', 8, 3072):.2f}&times; at n&nbsp;=&nbsp;3072. The unroll4 kernel reads all of B for every row of C (8n&sup3; bytes for 2n&sup3; flops), so it is bounded by the cache and DRAM bandwidth, which the containers share;
in addition, each executor has its own copy of B, so threads do not help each other through a common L3 copy as in a single JVM. This supports H5 and explains the first pilot. A blocked kernel moves less data and leaves room for scaling, but even then the 8-core run reaches only {S('D1', '4x2', 3072):.1f}&times;.</p>

<h3>4.8 Memory and data transfer</h3>
{table('table_memory.html')}
{fig('fig_memory', 'Figure 10. Left: peak resident memory (VmHWM) of the executor JVMs (D1, 640 MB heap), of the sequential Java process (A, B, C and a reference matrix) and the size of one matrix. Right: bytes sent by the driver per multiplication (eth0).', '90%')}
<p><b>Memory.</b> An executor JVM peaks at {mib(k['mem_exec_D1_1x1_n1024_kib']):.0f}&ndash;{mib(k['mem_exec_D1_4x2_n3072_kib']):.0f}&nbsp;MiB resident ({mib(k['mem_exec_D1_4x1_n3072_kib']):.0f}&nbsp;MiB at n&nbsp;=&nbsp;3072 with 1 core): its 640&nbsp;MB heap plus JVM overhead, for a 72&nbsp;MiB matrix B; broadcast data exist in the executor twice (serialised blocks and the deserialised array). The per-executor memory does not decrease with the number of executors, so <b>the total memory of the cluster grows with E</b> (4 executors at n&nbsp;=&nbsp;3072: about {4 * mib(k['mem_exec_D1_4x1_n3072_kib']) / 1024:.1f}&nbsp;GiB),
while the sequential process needs {mib(k['mem_local_seq_blocked_p1_n3072_kib']):.0f}&nbsp;MiB (A, B, C and the reference). Rowblock therefore does not reduce memory per node: it needs all of B on every executor, and it would fail first on matrices that do not fit in one executor, where blockshuffle (each task sees only blocks) would be needed. The driver peaks at {mib(k['mem_driver_D1_4x2_n3072_kib']) / 1024:.2f}&nbsp;GiB (1&nbsp;GB heap) at n&nbsp;=&nbsp;3072 because it holds A, B, the A bands, the broadcast and C.
<b>Data transfer.</b> The driver sends {k['tx_D1_1x1_n2048'] / 2 ** 20:.0f}&nbsp;MiB at n&nbsp;=&nbsp;2048 with one executor (A: 32&nbsp;MiB, B: 32&nbsp;MiB) and {k['tx_D1_4x2_n2048'] / 2 ** 20:.0f}&nbsp;MiB with four executors: not 32&nbsp;MiB&nbsp;&times;&nbsp;(1&nbsp;+&nbsp;4) because executors that already have blocks of B serve the others (torrent broadcast), and it receives {k['rx_D1_4x2_n2048'] / 2 ** 20:.0f}&nbsp;MiB, the size of C. These volumes are small compared with the compute and cross an in-memory network; their cost appears as serialisation and fetch time inside the job, not as wire time.</p>

<h3>4.9 Unusual results and checks</h3>
<p>(a) <b>Spark with one core at n&nbsp;=&nbsp;2048 and 3072 equals the sequential run</b> (S&nbsp;=&nbsp;{S('D1', '1x1', 2048):.3f} and {S('D1', '1x1', 3072):.3f}): plausible because the fixed overhead (&asymp;&nbsp;0.1&ndash;0.2&nbsp;s) is a small fraction of 3.5&ndash;12&nbsp;s and both run the same kernel; it does not mean that the framework is free. (b) <b>The unroll4 and blocked sequential runs differ by only {100 * (L('seq', 'unroll4', 1, 3072) / L('seq', 'blocked', 1, 3072) - 1):.0f}&nbsp;% at n&nbsp;=&nbsp;3072</b>, so the benefit of blocking appears mainly under parallel load. (c) The sequential run at n&nbsp;=&nbsp;1024 is {L('seq', 'blocked', 1, 1024) / L('seq', 'unroll4', 1, 1024):.2f} of unroll4 (blocked is slightly slower there: tile overhead when B fits in cache).
(d) <b>Drift:</b> the anchor run was {100 * (1 - anc['end'] / anc['start']):.0f}&nbsp;% faster at the end of the campaign than at its start; the order of execution (layouts by increasing cores, local baselines last) means a part of the advantage of the baselines over Spark could be a few percent of drift. (e) Within a layout, the IQR is up to {100 * k['max_rel_iqr']:.0f}&nbsp;% of the median for the shortest runs; the speed-ups for n&nbsp;=&nbsp;512 and part of 1024 are therefore indicative.</p>

<h2>5. Discussion</h2>
<p><b>H1 (small sizes) is supported.</b> Up to n&nbsp;=&nbsp;1024 the end-to-end time of the distributed run does not benefit from the cores and, at n&nbsp;=&nbsp;512, grows with them; the fixed cost of the framework is about 0.2&nbsp;s with one core (Spark 1&times;1 minus sequential Java: 0.21&nbsp;s at n&nbsp;=&nbsp;512, 0.20&nbsp;s at 1024). From n&nbsp;=&nbsp;2048 the compute dominates: {S('D1', '4x2', 2048):.1f}&times; at 2048 and {S('D1', '4x2', 3072):.1f}&times; (best layout {L('seq', 'blocked', 1, 3072) / k['T_D3_2x4_n3072']:.1f}&times;) at 3072. Distributed execution pays only when the computation lasts several seconds.</p>
<p><b>H2 (sub-linear strong scaling) is supported, and the evidence points to the shared hardware and to the replication of B.</b> The efficiency against the sequential run is {S('D1', '4x2', 3072) / 8:.2f} at 8 cores (n&nbsp;=&nbsp;3072). Decomposition (Section 4.2): the kernel is {infl:.1f}&times; slower per core with 8 cores (a property of the machine, also seen in 8 threads of one JVM: {thr[(8, 3072)]:.1f}&times;, and in C: {A3('seq', 1, 3072) / A3('par', 8, 3072):.1f}&times;), the fetch of B costs more with more executors, and the driver work is serial.
We did not measure cache or memory counters, so the attribution of the kernel slowdown to cache, DRAM bandwidth or clocks is an interpretation consistent with D7 and with the baselines, not a direct observation. How much of the loss is specific to Spark can be read from S<sub>thr</sub>: at 8 cores and n&nbsp;=&nbsp;3072 Spark reaches {St('D1', '4x2', 3072):.2f} of what 8 threads of one JVM obtain with the same kernel, a measure of the cost of distribution on one machine.</p>
<p><b>H3 (fewer, larger executors) is supported</b> for n&nbsp;&ge;&nbsp;2048 (up to {k['T_D3_4x2_n2048'] / k['T_D3_1x8_n2048']:.2f}&times; at 8 cores), with the caveat on the size of the effect (Section 4.4). <b>H4 is supported:</b> two shuffles make the block algorithm {T('D2', '4x2', 2048) / T('D1', '4x2', 2048):.1f}&times; slower at 8 cores and n&nbsp;=&nbsp;2048, with {k['shuf_write_D2_4x2_n2048'] / 2 ** 20:.0f}&nbsp;MiB written to shuffle files against 0 for rowblock; its justification would be matrices that exceed one executor&rsquo;s memory.
<b>H5 is supported:</b> with unroll4 more cores do not help beyond two. <b>H6 is supported:</b> E<sub>w</sub>&nbsp;=&nbsp;{we['8']:.2f} at 8 cores.</p>
<p><b>Distributed versus sequential and parallel.</b> The cluster is not the fastest option for any size tested: a multi-threaded JVM (or the C/OpenMP program) on the same machine is {1 / St('D1', '4x2', 3072):.1f}&times; faster than Spark at n&nbsp;=&nbsp;3072. This is expected: Spark pays for generality (serialisation, scheduling, separate processes) and gains fault tolerance and the ability to span machines and memories, none of which is exercised on one laptop. The conclusion applies to a local deployment; with a network, the broadcast and shuffle volumes measured here would cost real transfer time, and per-core compute would not suffer from sharing one memory system, so strong scaling on separate machines could be better or worse.</p>
<p><b>Limitations.</b> (i) One machine, one laptop CPU (15&nbsp;W class) and one Docker VM with 6.7&nbsp;GiB: eight containers did not fit and the largest size is 3072; the local cluster hides network costs and shares all hardware. (ii) Local execution: a local multi-container run is appropriate when no multi-machine infrastructure is available and must be clearly labelled, as done here; it does not demonstrate inter-machine performance. (iii) Turbo, temperature and power plan were not controlled; drift of up to 15&nbsp;% was measured. (iv) The baselines were measured in a different harness than the Spark runs (kernel-only timing versus end-to-end), which is deliberate but means the speed-ups include framework work. (v) Java serialisation and default compression were used; Kryo or no compression would change the broadcast and shuffle costs. (vi) 5&ndash;7 repetitions; no hardware counters. (vii) The sizes are small for a distributed framework; matrices larger than one machine&rsquo;s memory were not tested.</p>

<h2>6. Conclusions</h2>
<ol>
<li>Within the tested scope (a local Spark 3.5.6 standalone cluster of up to 4 containers and 8 cores on one 8-core laptop, Java, binary64, n&nbsp;=&nbsp;512&ndash;3072), both distributed algorithms are correct (rowblock bit-identical to the sequential kernel; blockshuffle within {k['max_abs_diff_blockshuffle']:.0e}).</li>
<li>Distributing pays only for computations of several seconds: for n&nbsp;&le;&nbsp;1024 the distributed run is not faster than sequential Java; at n&nbsp;=&nbsp;3072 it reaches {S('D1', '4x2', 3072):.1f}&times; with 8 cores (best layout {L('seq', 'blocked', 1, 3072) / k['T_D3_2x4_n3072']:.1f}&times;), an efficiency of 0.30&ndash;0.38, and 4 to 8 cores bring almost nothing.</li>
<li>The loss comes from a fixed framework cost of about 0.2&nbsp;s, the replication and deserialisation of B in every executor, serial work on the driver and a kernel that runs {infl:.1f}&times; slower per core when the cores share one memory system; fewer and larger executors reduce it. The weak-scaling efficiency is {we['8']:.2f} at 8 cores.</li>
<li>The 2-D block algorithm with two shuffles is slower than broadcasting B at these sizes; the kernel of Assignments 2 and 3 does not scale on a shared memory system, while a cache-blocked kernel does, moderately. A multi-threaded program on the same machine is faster than the cluster at every size tested.</li></ol>
<p>Unresolved: network cost and scaling on several machines, the exact hardware cause of the per-core slowdown (counters were not available), the behaviour for matrices larger than one executor&rsquo;s memory, and the effect of Kryo serialisation and of other broadcast settings.</p>

<h2>Reproducibility materials</h2>
<p style="text-align:left">Folder <code>Assignment04_AlejandroSantana_Materials</code> (and the repository {REPO_URL}): the Java sources (<code>src/main/java/bd/matmul4</code>) and <code>pom.xml</code>, the Docker Compose cluster (<code>docker/docker-compose.yml</code>), <code>README.md</code> with exact build and run commands and versions, <code>config/benchmark_config.json</code> (sizes, layouts, experiments, repetitions, budgets, Spark settings), the input generator and manifest, the raw samples of every warm-up and measured run with the Spark task metrics (<code>results/raw/</code>, <code>results/raw_samples.csv</code>), the validation output, the environment description, the native C reference, the Spark logs, and the scripts that produce all tables and figures. <code>run_all.ps1</code> rebuilds, validates, measures, analyses and regenerates this report (about 1.5 hours).
Reused: the unroll4 kernel of Assignments 2 and 3, the ijk loop of Assignment 1, the cache-blocking idea of Assignment 2, the input matrices of Assignments 1&ndash;3 (same seeds, verified hashes), and the unchanged Assignment 3 executable (commit {open(os.path.join(RES, 'reference_a3_native.csv')).readlines()[1].strip().split(',')[-1][:7]}) re-run on this machine.</p>

<h2>References</h2>
<ol class="ref">
<li>M. Zaharia <i>et al.</i>, &ldquo;Resilient distributed datasets: A fault-tolerant abstraction for in-memory cluster computing,&rdquo; in <i>Proc. 9th USENIX Symp. Networked Systems Design and Implementation (NSDI)</i>, San Jose, CA, USA, 2012, pp. 15&ndash;28.</li>
<li>The Apache Software Foundation, <i>Apache Spark 3.5.6 Documentation</i>: &ldquo;RDD Programming Guide&rdquo; (shuffle operations, broadcast variables), &ldquo;Cluster Mode Overview&rdquo; and &ldquo;Spark Standalone Mode.&rdquo; https://spark.apache.org/docs/3.5.6/</li>
<li>M. Zaharia <i>et al.</i>, &ldquo;Apache Spark: A unified engine for big data processing,&rdquo; <i>Commun. ACM</i>, vol. 59, no. 11, pp. 56&ndash;65, 2016.</li>
<li>G. M. Amdahl, &ldquo;Validity of the single processor approach to achieving large scale computing capabilities,&rdquo; in <i>Proc. AFIPS Spring Joint Computer Conf.</i>, 1967, pp. 483&ndash;485.</li>
<li>J. L. Gustafson, &ldquo;Reevaluating Amdahl&rsquo;s law,&rdquo; <i>Commun. ACM</i>, vol. 31, no. 5, pp. 532&ndash;533, 1988.</li>
<li>R. A. van de Geijn and J. Watts, &ldquo;SUMMA: Scalable universal matrix multiplication algorithm,&rdquo; <i>Concurrency: Practice and Experience</i>, vol. 9, no. 4, pp. 255&ndash;274, 1997.</li>
<li>S. Williams, A. Waterman, and D. Patterson, &ldquo;Roofline: An insightful visual performance model for multicore architectures,&rdquo; <i>Commun. ACM</i>, vol. 52, no. 4, pp. 65&ndash;76, 2009.</li>
<li>K. Goto and R. A. van de Geijn, &ldquo;Anatomy of high-performance matrix multiplication,&rdquo; <i>ACM Trans. Math. Softw.</i>, vol. 34, no. 3, Art. 12, 2008.</li>
<li>N. J. Higham, <i>Accuracy and Stability of Numerical Algorithms</i>, 2nd ed. Philadelphia, PA, USA: SIAM, 2002.</li>
<li>Docker, Inc., &ldquo;Docker Desktop WSL 2 backend on Windows&rdquo; and &ldquo;Docker Compose overview,&rdquo; Docker Documentation. https://docs.docker.com/</li>
<li>M. Kerrisk, <i>proc(5) &ndash; process information pseudo-filesystem</i>, Linux man-pages (<code>/proc/[pid]/status</code>: VmHWM; <code>/proc/net/dev</code>).</li>
<li>J. L. Hennessy and D. A. Patterson, <i>Computer Architecture: A Quantitative Approach</i>, 6th ed. Cambridge, MA, USA: Morgan Kaufmann, 2017.</li></ol>
</body></html>"""
    open(OUT_HTML, "w", encoding="utf8").write(html)
    print("wrote", OUT_HTML)
    if args.no_pdf:
        return
    browsers = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
    exe = next((b for b in browsers if os.path.exists(b)), shutil.which("chrome") or shutil.which("msedge"))
    if not exe:
        sys.exit("No Chrome/Edge found: open the HTML file and print it to PDF manually.")
    subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT_PDF}",
                    "file:///" + OUT_HTML.replace("\\", "/")], check=True, timeout=180)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
