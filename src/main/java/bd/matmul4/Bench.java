package bd.matmul4;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.util.LongAccumulator;

/**
 * Benchmark driver of Assignment 4. One invocation = one experiment on one cluster size (spark mode) or one
 * local baseline configuration (local mode). Arguments are key=value pairs, see README.md.
 *
 * Timed region of every sample ("end-to-end"): the inputs are already in the driver's memory as flat
 * double[] arrays; the timer covers building the distributed data, the broadcast, the Spark job(s) and the
 * assembly of C on the driver (or of the checksum when the result stays distributed). Reading the input
 * files, the correctness check against the reference and the destruction of the broadcast are outside it.
 */
public final class Bench {
    private Bench() { }

    static Map<String, String> opts = new LinkedHashMap<>();

    static String opt(String k, String def) {
        return opts.getOrDefault(k, def);
    }

    static int[] ints(String s) {
        if (s == null || s.isBlank()) {
            return new int[0];
        }
        String[] f = s.split(",");
        int[] r = new int[f.length];
        for (int i = 0; i < f.length; i++) {
            r[i] = Integer.parseInt(f[i].trim());
        }
        return r;
    }

    public static void main(String[] args) throws Exception {
        for (String a : args) {
            int eq = a.indexOf('=');
            opts.put(a.substring(0, eq), a.substring(eq + 1));
        }
        String mode = opt("mode", "spark");
        if (mode.equals("local")) {
            local();
        } else {
            spark();
        }
    }

    // ---------------------------------------------------------------------------------------- common

    static final class Check {
        boolean ok;
        boolean bitEqual;
        double maxAbsDiff;
    }

    /** Compare a result with the reference: tolerance |dC| <= tol * (1 + |ref|), reported with the bit-exactness. */
    static Check compare(double[] c, double[] ref, double tol) {
        Check k = new Check();
        k.bitEqual = true;
        double max = 0;
        boolean ok = true;
        for (int i = 0; i < ref.length; i++) {
            if (Double.doubleToLongBits(c[i]) != Double.doubleToLongBits(ref[i])) {
                k.bitEqual = false;
            }
            double d = Math.abs(c[i] - ref[i]);
            if (d > max) {
                max = d;
            }
            if (!(d <= tol * (1.0 + Math.abs(ref[i])))) {
                ok = false;
            }
        }
        k.ok = ok;
        k.maxAbsDiff = max;
        return k;
    }

    static Map<String, Object> baseRow(String exp, String method, String kernel, int workers, int parts, int grid, int threads, int n,
                                       String phase, int index) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("experiment", exp);
        m.put("method", method);
        m.put("kernel", method.equals("ijk") ? "ijk" : kernel);
        m.put("workers", workers);
        m.put("cores_per_worker", Integer.parseInt(opt("cores", "1")));
        m.put("partitions", parts);
        m.put("grid", grid);
        m.put("threads", threads);
        m.put("n", n);
        m.put("phase", phase);
        m.put("index", index);
        return m;
    }

    static final String[] METRIC_COLS = {"t_total_ns", "t_partition_ns", "t_broadcast_ns", "t_job_ns", "t_assemble_ns",
        "sum_kernel_ns", "sum_fetch_ns", "tasks", "executor_run_ms", "executor_cpu_ns", "deserialize_ms", "result_ser_ms",
        "result_size_bytes", "gc_ms", "shuffle_remote_bytes", "shuffle_local_bytes", "shuffle_write_bytes",
        "shuffle_fetch_wait_ms", "spill_mem_bytes", "spill_disk_bytes", "peak_exec_vmhwm_kib",
        "driver_rx_bytes", "driver_tx_bytes", "driver_vmhwm_kib", "check_ok", "max_abs_diff", "bit_equal",
        "checksum_sum", "checksum_abs"};

    static void fillEmpty(Map<String, Object> row) {
        for (String c : METRIC_COLS) {
            row.putIfAbsent(c, null);
        }
    }

    // ---------------------------------------------------------------------------------------- spark mode

    static void spark() throws Exception {
        final String exp = opt("exp", "X");
        final String method = opt("method", "rowblock");          // rowblock | rowblock-sum | blockshuffle
        final String kernel = opt("kernel", "blocked");           // blocked | unroll4
        final int workers = Integer.parseInt(opt("workers", "1"));
        final int[] sizes = ints(opt("sizes", "512"));
        final int warmup = Integer.parseInt(opt("warmup", "2"));
        final int reps = Integer.parseInt(opt("reps", "7"));
        final int repsLarge = Integer.parseInt(opt("repsLarge", "5"));
        final int largeAbove = Integer.parseInt(opt("largeAbove", "1024"));
        final double budgetS = Double.parseDouble(opt("budgetS", "120"));
        final int grid = Integer.parseInt(opt("grid", "4"));
        final int partsPerWorker = Integer.parseInt(opt("partsPerCore", "4"));
        final int[] partsList = opts.containsKey("partsList") ? ints(opt("partsList", "")) : new int[] {partsPerWorker * workers * Integer.parseInt(opt("cores", "1"))};
        final Path data = Path.of(opt("data", "/data"));
        final Path out = Path.of(opt("out", "/results/raw"));
        final int cores = Integer.parseInt(opt("cores", "1"));
        final String tag = "L" + workers + "x" + cores;

        SparkConf conf = new SparkConf().setAppName("matmul4-" + exp + "-" + tag);
        // master, executor memory/cores and the other settings come from spark-submit (scripts/run_benchmarks.py)
        try (JavaSparkContext sc = new JavaSparkContext(conf)) {
            StatsListener stats = new StatsListener();
            sc.sc().addSparkListener(stats);
            LongAccumulator kernelAcc = sc.sc().longAccumulator("kernel_ns");
            LongAccumulator fetchAcc = sc.sc().longAccumulator("fetch_ns");
            MaxAccumulator peakAcc = new MaxAccumulator();
            sc.sc().register(peakAcc, "peak_vmhwm_kib");
            Csv samples = new Csv(out.resolve("samples_" + exp + "_" + tag + ".csv"));
            Csv runs = new Csv(out.resolve("runs_" + exp + "_" + tag + ".csv"));
            Csv info = new Csv(out.resolve("info_" + exp + "_" + tag + ".csv"));

            // wait for the requested executors
            long deadline = System.currentTimeMillis() + 120_000;
            int execs = 0;
            while (System.currentTimeMillis() < deadline) {
                execs = sc.sc().statusTracker().getExecutorInfos().length - 1;   // minus the driver
                if (execs >= workers) {
                    break;
                }
                Thread.sleep(500);
            }
            Thread.sleep(1500);   // let the executors finish registering
            execs = sc.sc().statusTracker().getExecutorInfos().length - 1;
            for (String[] kv : new String[][] {
                {"spark_version", sc.version()}, {"java_version", System.getProperty("java.version")},
                {"java_vm", System.getProperty("java.vm.name")}, {"os_arch", System.getProperty("os.arch")},
                {"driver_available_processors", String.valueOf(Runtime.getRuntime().availableProcessors())},
                {"driver_max_heap_mib", String.valueOf(Runtime.getRuntime().maxMemory() >> 20)},
                {"master", sc.master()}, {"workers_requested", String.valueOf(workers)}, {"cores_per_worker", String.valueOf(cores)},
                {"executors_registered", String.valueOf(execs)},
                {"executor_memory", sc.getConf().get("spark.executor.memory", "")},
                {"executor_cores", sc.getConf().get("spark.executor.cores", "")},
                {"cores_max", sc.getConf().get("spark.cores.max", "")},
                {"serializer", sc.getConf().get("spark.serializer", "org.apache.spark.serializer.JavaSerializer (default)")},
                {"broadcast_compress", sc.getConf().get("spark.broadcast.compress", "true (default)")},
                {"shuffle_compress", sc.getConf().get("spark.shuffle.compress", "true (default)")}}) {
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("experiment", exp);
                row.put("workers", workers);
                row.put("key", kv[0]);
                row.put("value", kv[1]);
                info.append(row);
            }
            if (execs < workers) {
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("experiment", exp);
                row.put("method", method);
                row.put("kernel", kernel);
                row.put("workers", workers);
                row.put("cores_per_worker", Integer.parseInt(opt("cores", "1")));
                row.put("partitions", "");
                row.put("grid", grid);
                row.put("threads", "");
                row.put("n", "");
                row.put("status", "EXECUTORS_MISSING");
                row.put("warmup", warmup);
                row.put("reps", 0);
                row.put("load_s", "");
                row.put("ref_s", "");
                row.put("note", execs + " of " + workers + " executors registered");
                runs.append(row);
                System.exit(3);
            }

            boolean stop = false;
            for (int parts : partsList) {
                stop = false;
                for (int n : sizes) {
                    if (stop) {
                        Map<String, Object> row = runRow(exp, method, kernel, workers, parts, grid, 0, n, "NOT_ATTEMPTED", warmup, 0, 0, 0,
                                "larger than a size that exceeded the budget");
                        runs.append(row);
                        continue;
                    }
                    long l0 = System.nanoTime();
                    double[] a = Inputs.read(data.resolve("A_n" + n + ".bin"), n);
                    double[] b = Inputs.read(data.resolve("B_n" + n + ".bin"), n);
                    double loadS = (System.nanoTime() - l0) / 1e9;
                    long r0 = System.nanoTime();
                    double[] ref = new double[n * n];
                    Kernels.seq("unroll4", a, b, ref, n);            // untimed reference (sequential unroll4 kernel; all kernels give the same bits)
                    double[] refSum = Kernels.checksum(ref);
                    double refS = (System.nanoTime() - r0) / 1e9;
                    int nReps = n <= largeAbove ? reps : repsLarge;
                    String status = "OK";
                    String note = "";
                    int done = 0;
                    for (int it = 0; it < warmup + nReps; it++) {
                        boolean isWarm = it < warmup;
                        System.gc();
                        stats.reset();
                        long[] net0 = Sys.netBytes();
                        SparkMatmul.Run run;
                        if (method.equals("blockshuffle")) {
                            run = SparkMatmul.blockShuffle(sc, a, b, n, grid, parts, kernel, kernelAcc, peakAcc);
                        } else {
                            run = SparkMatmul.rowBlock(sc, a, b, n, parts, method.equals("rowblock"), kernel, kernelAcc, fetchAcc, peakAcc);
                        }
                        long[] net1 = Sys.netBytes();
                        stats.awaitJobs(1, 10_000);
                        Map<String, Object> row = baseRow(exp, method, kernel, workers, parts, grid, 0, n,
                                isWarm ? "warmup" : "measured", isWarm ? it : it - warmup);
                        row.put("t_total_ns", run.tTotal);
                        row.put("t_partition_ns", run.tPartition);
                        row.put("t_broadcast_ns", run.tBroadcast);
                        row.put("t_job_ns", run.tJob);
                        row.put("t_assemble_ns", run.tAssemble);
                        row.put("sum_kernel_ns", run.sumKernelNs);
                        row.put("sum_fetch_ns", run.sumFetchNs);
                        row.put("tasks", stats.tasks);
                        row.put("executor_run_ms", stats.executorRunMs);
                        row.put("executor_cpu_ns", stats.executorCpuNs);
                        row.put("deserialize_ms", stats.deserializeMs);
                        row.put("result_ser_ms", stats.resultSerializationMs);
                        row.put("result_size_bytes", stats.resultSizeBytes);
                        row.put("gc_ms", stats.gcMs);
                        row.put("shuffle_remote_bytes", stats.shuffleRemoteBytes);
                        row.put("shuffle_local_bytes", stats.shuffleLocalBytes);
                        row.put("shuffle_write_bytes", stats.shuffleWriteBytes);
                        row.put("shuffle_fetch_wait_ms", stats.shuffleFetchWaitMs);
                        row.put("spill_mem_bytes", stats.spillMemBytes);
                        row.put("spill_disk_bytes", stats.spillDiskBytes);
                        row.put("peak_exec_vmhwm_kib", run.peakExecVmHwmKib);
                        row.put("driver_rx_bytes", net1[0] - net0[0]);
                        row.put("driver_tx_bytes", net1[1] - net0[1]);
                        row.put("driver_vmhwm_kib", Sys.vmHwmKib());
                        // correctness (outside the timer)
                        if (run.c != null) {
                            Check k = compare(run.c, ref, 1e-9);
                            row.put("check_ok", k.ok);
                            row.put("max_abs_diff", k.maxAbsDiff);
                            row.put("bit_equal", k.bitEqual);
                            double[] cs = Kernels.checksum(run.c);
                            row.put("checksum_sum", cs[0]);
                            row.put("checksum_abs", cs[1]);
                        } else {
                            row.put("check_ok", Math.abs(run.sum - refSum[0]) <= 1e-9 * refSum[1]
                                    && Math.abs(run.sumAbs - refSum[1]) <= 1e-9 * refSum[1]);
                            row.put("max_abs_diff", Math.abs(run.sum - refSum[0]));
                            row.put("bit_equal", null);
                            row.put("checksum_sum", run.sum);
                            row.put("checksum_abs", run.sumAbs);
                        }
                        samples.append(row);
                        if (!isWarm) {
                            done++;
                        }
                        if (run.tTotal / 1e9 > budgetS) {
                            status = "TIMEOUT";
                            note = "a run took " + String.format("%.1f", run.tTotal / 1e9) + " s > " + budgetS + " s budget";
                            stop = true;
                            break;
                        }
                    }
                    runs.append(runRow(exp, method, kernel, workers, parts, grid, 0, n, status, warmup, done, loadS, refS, note));
                }
            }
        }
    }

    static Map<String, Object> runRow(String exp, String method, String kernel, int workers, int parts, int grid, int threads, int n,
                                      String status, int warmup, int reps, double loadS, double refS, String note) {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("experiment", exp);
        row.put("method", method);
        row.put("kernel", method.equals("ijk") ? "ijk" : kernel);
        row.put("workers", workers);
        row.put("cores_per_worker", Integer.parseInt(opt("cores", "1")));
        row.put("partitions", parts);
        row.put("grid", grid);
        row.put("threads", threads);
        row.put("n", n);
        row.put("status", status);
        row.put("warmup", warmup);
        row.put("reps", reps);
        row.put("load_s", loadS);
        row.put("ref_s", refS);
        row.put("note", note);
        return row;
    }

    // ---------------------------------------------------------------------------------------- local mode

    /** Local baselines inside one JVM: method = ijk | seq | threads (p = threads, static row bands, persistent pool). */
    static void local() throws Exception {
        final String exp = opt("exp", "L1");
        final String method = opt("method", "seq");
        final String kernel = opt("kernel", "blocked");
        final int threads = Integer.parseInt(opt("threads", "1"));
        final int n = Integer.parseInt(opt("n", "512"));
        final int warmup = Integer.parseInt(opt("warmup", "2"));
        final int reps = Integer.parseInt(opt("reps", "7"));
        final int repsLarge = Integer.parseInt(opt("repsLarge", "5"));
        final int largeAbove = Integer.parseInt(opt("largeAbove", "1024"));
        final double budgetS = Double.parseDouble(opt("budgetS", "120"));
        final Path data = Path.of(opt("data", "/data"));
        final Path out = Path.of(opt("out", "/results/raw"));
        final String tag = "local-" + method + "-" + kernel + "-p" + threads + "-n" + n;
        Csv samples = new Csv(out.resolve("samples_" + exp + "_" + tag + ".csv"));
        Csv runs = new Csv(out.resolve("runs_" + exp + "_" + tag + ".csv"));

        long l0 = System.nanoTime();
        double[] a = Inputs.read(data.resolve("A_n" + n + ".bin"), n);
        double[] b = Inputs.read(data.resolve("B_n" + n + ".bin"), n);
        double loadS = (System.nanoTime() - l0) / 1e9;
        double[] ref = new double[n * n];
        long r0 = System.nanoTime();
        if (!method.equals("seq")) {
            Kernels.seq("unroll4", a, b, ref, n);
        }
        double refS = (System.nanoTime() - r0) / 1e9;
        double[] c = new double[n * n];
        ExecutorService pool = method.equals("threads") ? Executors.newFixedThreadPool(threads) : null;
        int[] bnd = Kernels.bounds(n, Math.max(threads, 1));
        int nReps = n <= largeAbove ? reps : repsLarge;
        String status = "OK";
        String note = "";
        int done = 0;
        boolean seqRefSet = false;
        try {
            for (int it = 0; it < warmup + nReps; it++) {
                boolean isWarm = it < warmup;
                long t0 = System.nanoTime();
                switch (method) {
                    case "ijk" -> Kernels.ijk(a, b, c, n);
                    case "seq" -> Kernels.seq(kernel, a, b, c, n);
                    case "threads" -> {
                        List<Future<?>> fs = new ArrayList<>();
                        for (int t = 0; t < threads; t++) {
                            final int r1 = bnd[t], r2 = bnd[t + 1];
                            fs.add(pool.submit(() -> Kernels.rows(kernel, a, b, c, n, r1, r2)));
                        }
                        for (Future<?> f : fs) {
                            f.get();
                        }
                    }
                    default -> throw new IllegalArgumentException("method " + method);
                }
                long dt = System.nanoTime() - t0;
                if (method.equals("seq") && !seqRefSet) {
                    System.arraycopy(c, 0, ref, 0, c.length);       // the first sequential result is the reference
                    seqRefSet = true;
                }
                Check k = compare(c, ref, 1e-9);
                Map<String, Object> row = baseRow(exp, method, kernel, 0, 0, 0, threads, n, isWarm ? "warmup" : "measured",
                        isWarm ? it : it - warmup);
                row.put("t_total_ns", dt);
                row.put("driver_vmhwm_kib", Sys.vmHwmKib());
                row.put("check_ok", k.ok);
                row.put("max_abs_diff", k.maxAbsDiff);
                row.put("bit_equal", k.bitEqual);
                double[] cs = Kernels.checksum(c);
                row.put("checksum_sum", cs[0]);
                row.put("checksum_abs", cs[1]);
                fillEmpty(row);
                samples.append(row);
                if (!isWarm) {
                    done++;
                }
                if (dt / 1e9 > budgetS) {
                    status = "TIMEOUT";
                    note = "a run took " + String.format("%.1f", dt / 1e9) + " s > " + budgetS + " s budget";
                    break;
                }
            }
        } finally {
            if (pool != null) {
                pool.shutdownNow();
            }
        }
        runs.append(runRow(exp, method, kernel, 0, 0, 0, threads, n, status, warmup, done, loadS, refS, note));
    }
}
