# Assignment 4 — Distributed Matrix Multiplication (Apache Spark, Java, Docker)

**Big Data · Grado en Ciencia e Ingeniería de Datos · ULPGC · 2026-2027**
Author: Aimar Daniel Alejandro Santana

Research question:

* *How does a dense `float64` matrix multiplication on an Apache Spark standalone cluster scale with the number of
  executors and cores, how much of the end-to-end time is computation and how much is framework overhead (task
  scheduling, broadcast, serialisation, shuffle, result collection), and how does it compare with a sequential and a
  multi-threaded implementation on the same machine?*

**Environment label:** the cluster is *local*: a master, up to four worker containers and a driver container run on
**one** physical machine (Docker Desktop / WSL2 virtual machine, one bridge network). It studies framework behaviour and
process overhead; it does **not** measure inter-machine network performance or multi-machine memory capacity.

The report is `../Assignment04_AlejandroSantana_Report.pdf` (source: `report/build_report.py`).

## What is in this folder

```
src/main/java/bd/matmul4/
  Kernels.java        float64 kernels on flat row-major arrays: ijk (Assignment 1), unroll4 (Assignments 2/3), blocked (tiles 64 x 256)
  SparkMatmul.java    the two distributed algorithms: rowblock (row bands + broadcast of B) and blockshuffle (4 x 4 blocks, join + reduceByKey)
  Bench.java          benchmark driver: spark mode (one experiment on one cluster layout) and local mode (Java baselines, no Spark)
  Validate.java       correctness tests (edge cases; benchmark inputs dumped for the NumPy comparison)
  StatsListener.java  Spark task metrics (run / deserialisation / GC / shuffle bytes ...);  MaxAccumulator.java  peak memory of the executors
  Inputs.java  Csv.java  Sys.java   input files, CSV output, /proc counters (peak RSS, network bytes)
pom.xml                          Maven build (Spark 3.5.6 'provided', Java release 17)
docker/docker-compose.yml        master + N workers (--scale worker=N, WORKER_CORES=C) + driver container
build.ps1                        mvn package + docker pull
scripts/generate_inputs.py       shared inputs (same seeds and SHA-256 as Assignments 1-3, plus the weak-scaling sizes 1290 and 1626)
scripts/run_benchmarks.py        harness: starts each cluster layout, runs each experiment as one spark-submit, budgets, anchors
scripts/run_reference_a3.py      native C/OpenMP program of Assignment 3 re-run on THIS machine (same inputs)
scripts/collect_environment.ps1  hardware / software description -> results/environment.txt
scripts/analyze.py               statistics, speed-up / efficiency, overhead breakdown, figures/*.png|pdf, report/generated/*.html
tests/validate.py                correctness: edge cases + benchmark inputs against NumPy and bit-for-bit against the sequential kernel
report/build_report.py           assembles the English report (HTML) and prints it to PDF with Chrome/Edge
config/benchmark_config.json     sizes, layouts, experiments, repetitions, budgets, Spark settings
results/raw/                     per-run raw data: samples_*.csv (every warm-up and measured sample), runs_*.csv, info_*.csv, logs/
results/                         raw_samples.csv, runs.csv, info.csv, summary.csv, scaling.csv, validation*.csv, environment.txt, key_numbers.json ...
figures/                         all figures (PNG + PDF)
run_all.ps1                      full reproduction (Windows PowerShell)
```

## Reused material (Assignments 1-3)

* **Inputs:** the Assignment 1-3 matrices: NumPy `default_rng` (PCG64), seeds 20261012 (A) and 20261013 (B), uniform in
  [-1, 1). `generate_inputs.py --check <manifest>` compares the SHA-256 of every regenerated file with the manifests of
  Assignments 1 and 3 (all files present there match, key `checked_against` in `data/inputs/manifest.json`).
* **Kernels:** `unroll4` is the ikj kernel with the k loop unrolled by 4 of Assignments 2 and 3 (`row_unroll4` in
  `matmul3.c`), rewritten in Java for rectangular blocks; `ijk` is the basic loop of Assignment 1; `blocked` adds the
  cache-blocking idea of Assignment 2 (tiles of 64 x 256) while keeping, for every C[i][j], the same additions in the same
  order, so that **all kernels return bit-identical results** (checked).
* **Native reference:** `scripts/run_reference_a3.py` runs the unchanged Assignment 3 executable
  (`Assignment03_AlejandroSantana_Materials/build/matmul3_scalar.exe`, repository commit recorded in the output) on this machine.

## Requirements

* Windows 10/11 (the scripts are PowerShell/Python), **Docker Desktop** with Compose (WSL2 back end). The Docker VM needs
  **at least 6 GB of RAM** (tested: 6.7 GiB, 16 vCPUs); the layouts were chosen to fit in it.
* **JDK 17+** and **Maven 3.9** to build the jar (tested: Temurin 21.0.12 with `--release 17`, Maven 3.9.16). The runtime inside the
  containers is the JDK 17 of the image `apache/spark:3.5.6-java17` (Spark 3.5.6, Scala 2.12).
* **Python 3.11+** with `numpy`, `pandas`, `matplotlib`; with [uv](https://docs.astral.sh/uv/) nothing needs to be installed:
  `uv run --with numpy --with pandas --with matplotlib python <script>`.
* Google Chrome or Microsoft Edge (only to print the report to PDF).
* About 0.4 GB of free disk space for the inputs and 5 GB of free RAM outside Docker.

## Reproduce everything

```powershell
.\run_all.ps1      # build -> environment -> inputs -> validation -> benchmarks (~1.5 h) -> reference -> analysis -> report
```

Step by step:

```powershell
.\build.ps1                                                   # target/matmul4.jar + docker pull apache/spark:3.5.6-java17
.\scripts\collect_environment.ps1                             # results\environment.txt
uv run --with numpy python scripts/generate_inputs.py --outdir C:\bench_a4\data `
        --check ..\assigment-3\Assignment03_AlejandroSantana_Materials\data\inputs\manifest.json `
        --check ..\Assignment01_AlejandroSantana_Materials\data\inputs\manifest.json
uv run --with numpy python tests/validate.py --data-dir C:\bench_a4\data       # results\validation.csv
python scripts/run_benchmarks.py --data-dir C:\bench_a4\data   # all experiments; --only D1_strong_rowblock,L1_local selects some
python scripts/run_reference_a3.py --data-dir C:\bench_a4\data # native C reference (stop Docker workloads first)
uv run --with numpy --with pandas --with matplotlib python scripts/analyze.py
uv run --with numpy --with pandas --with matplotlib python report/build_report.py
```

`run_benchmarks.py` only needs the standard library. It starts and stops the cluster itself
(`docker compose up -d --scale worker=E` with `WORKER_CORES=C`), runs one `spark-submit` per experiment and layout, and
removes nothing: delete `results/raw` before a full re-run.

Running one configuration by hand (cluster of 2 workers x 1 core already started with `docker compose up -d --scale worker=2`):

```powershell
cd docker
docker compose exec driver /opt/spark/bin/spark-submit --master spark://master:7077 --driver-memory 1g --executor-memory 640m `
   --conf spark.executor.cores=1 --conf spark.cores.max=2 --conf spark.ui.enabled=false `
   --conf spark.driver.host=driver --conf spark.driver.bindAddress=0.0.0.0 --class bd.matmul4.Bench /app/matmul4.jar `
   mode=spark exp=X method=rowblock kernel=blocked workers=2 cores=1 sizes=1024,2048 warmup=2 reps=7 repsLarge=5 data=/data out=/results/raw
#  method: rowblock | rowblock-sum | blockshuffle     kernel: blocked | unroll4     partsPerCore=4 (tasks per core) or partsList=8,16,...
docker compose exec driver java -Xmx1g -cp "/app/matmul4.jar:/opt/spark/jars/*" bd.matmul4.Bench mode=local exp=X method=threads kernel=blocked threads=8 n=2048 out=/results/raw
#  local methods: ijk | seq | threads (p static row bands, persistent thread pool, one JVM)
```

## The study

| Id | Algorithm / kernel | Layouts (E containers x C cores) | Sizes n | Purpose |
|---|---|---|---|---|
| D1_strong_rowblock | rowblock, blocked | 1x1, 2x1, 4x1, 4x2 | 512, 1024, 2048, 3072 | strong scaling, main study |
| D2_strong_blockshuffle | blockshuffle (4x4 blocks), blocked | same | 512, 1024, 2048 | strong scaling with two shuffles |
| D3_layout | rowblock | 1x4, 2x2, 4x1 (4 cores); 1x8, 2x4, 4x2 (8 cores) | 1024, 2048, 3072 | containers vs cores per container |
| D4_result_stays | rowblock-sum | as D1 | 1024, 2048, 3072 | cost of returning C to the driver |
| D5_partitions | rowblock | 4x2 | 2048 | number of tasks 8 ... 256 |
| D6_weak | rowblock | as D1 | 1024, 1290, 1626, 2048 (n^3 doubles with the cores) | weak scaling |
| D7_unroll4_kernel | rowblock, unroll4 | as D1 | 1024, 2048 | kernel of Assignments 2/3: memory wall |
| L1_local | Java ijk / seq / threads (1-16) in one JVM | driver container | 512 ... 3072 | sequential and shared-memory baselines |
| ANCHOR_* | Java seq, blocked, n = 1024 | driver container | 1024 | drift of the machine during the campaign |

Spark settings (all in `config/benchmark_config.json`, `docker/docker-compose.yml`): one executor per worker container
(`spark.executor.cores` = cores of the container, executor heap 640 MB), driver heap 1 GB, Java serialiser and LZ4
compression (defaults), UI disabled, `spark.rpc.message.maxSize` 512 MB, tasks = 4 per core.

## Experimental protocol (summary — details in the report)

* **Timer boundary (end-to-end):** inputs already in the driver's memory as `double[]`; the timer covers building the
  distributed data (A bands or blocks), the broadcast of B, the Spark job(s) and the assembly of C on the driver (or of the
  checksums when the result stays distributed). Reading the files, Spark start-up, the correctness check and
  the destruction of the broadcast are outside it. Phase times (build RDD / broadcast / job / assemble) are measured on the driver;
  per-task kernel and B-fetch times inside the tasks; deserialisation, GC, shuffle bytes from Spark's task metrics;
  driver network bytes from `/proc/net/dev`; peak resident memory (`VmHWM`) of the executor JVMs. Task-level sums overlap in time and are
  never added to the end-to-end time.
* **Warm-up and repetitions:** 2 warm-up runs per (configuration, n) inside the same application (kept in the raw data, excluded
  from statistics), 7 measured repetitions for n ≤ 1024 and 5 above. **Budget:** 120 s per run.
* **Metrics:** median and IQR; S_seq = T_seq / T, E = S / C, S_1 = T(1 core) / T(C) and weak-scaling efficiency from medians.
* **Correctness:** `tests/validate.py` and a comparison of every measured run with the reference product
  (bit-exact for rowblock, |dC| ≤ 1e-9(1 + |C|) for blockshuffle and for checksums).

## AI-use disclosure

Generative AI (Claude, Anthropic) was used as described in the "AI-use disclosure" section of the report. All
measurements in `results/` were produced by running the code in this folder.

Repository: https://github.com/aimarsantana/BigData-Assignment04-MatMul
