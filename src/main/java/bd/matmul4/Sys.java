package bd.matmul4;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** Linux process and network counters of the container the code runs in (/proc). */
public final class Sys {
    private Sys() { }

    /** Peak resident set size of this process in KiB (VmHWM), or -1. */
    public static long vmHwmKib() {
        return procStatus("VmHWM:");
    }

    private static long procStatus(String key) {
        try {
            for (String line : Files.readAllLines(Path.of("/proc/self/status"))) {
                if (line.startsWith(key)) {
                    return Long.parseLong(line.substring(key.length()).trim().split("\\s+")[0]);
                }
            }
        } catch (IOException | RuntimeException e) {
            // not on Linux
        }
        return -1;
    }

    /** {rx bytes, tx bytes} of interface eth0 of this container (-1, -1 if unavailable). */
    public static long[] netBytes() {
        try {
            List<String> lines = Files.readAllLines(Path.of("/proc/net/dev"));
            for (String line : lines) {
                String t = line.trim();
                if (t.startsWith("eth0:")) {
                    String[] f = t.substring(5).trim().split("\\s+");
                    return new long[] {Long.parseLong(f[0]), Long.parseLong(f[8])};
                }
            }
        } catch (IOException | RuntimeException e) {
            // not on Linux
        }
        return new long[] {-1, -1};
    }
}
