package bd.matmul4;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.stream.Collectors;

/** Minimal CSV appender: the header is written when the file is created. */
public final class Csv {
    private final Path file;
    private boolean headerDone;

    public Csv(Path file) throws IOException {
        this.file = file;
        Files.createDirectories(file.toAbsolutePath().getParent());
        this.headerDone = Files.exists(file) && Files.size(file) > 0;
    }

    public synchronized void append(Map<String, Object> row) throws IOException {
        StringBuilder sb = new StringBuilder();
        if (!headerDone) {
            sb.append(String.join(",", row.keySet())).append('\n');
            headerDone = true;
        }
        sb.append(row.values().stream().map(Csv::cell).collect(Collectors.joining(","))).append('\n');
        Files.writeString(file, sb.toString(), StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    private static String cell(Object v) {
        if (v == null) {
            return "";
        }
        if (v instanceof Double d) {
            return Double.isNaN(d) ? "" : Double.toString(d);
        }
        String s = v.toString();
        return (s.contains(",") || s.contains("\"") || s.contains("\n")) ? "\"" + s.replace("\"", "\"\"") + "\"" : s;
    }
}
