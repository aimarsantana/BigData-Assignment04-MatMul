package bd.matmul4;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/** Reads and writes the shared input files: raw little-endian float64, row-major (A_n{n}.bin, B_n{n}.bin). */
public final class Inputs {
    private Inputs() { }

    public static double[] read(Path file, int n) throws IOException {
        long expected = 8L * n * n;
        if (Files.size(file) != expected) {
            throw new IOException(file + ": expected " + expected + " bytes, found " + Files.size(file));
        }
        double[] m = new double[n * n];
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            ByteBuffer buf = ByteBuffer.allocateDirect(8 << 19).order(ByteOrder.LITTLE_ENDIAN); // 4 MiB, multiple of 8
            int off = 0;
            while (off < m.length) {
                buf.clear();
                int total = 0;
                while (buf.hasRemaining()) {   // fill the buffer completely, so that it always holds whole doubles
                    int got = ch.read(buf);
                    if (got < 0) {
                        break;
                    }
                    total += got;
                }
                if (total == 0 || total % 8 != 0) {
                    throw new IOException("unexpected end of file " + file);
                }
                buf.flip();
                int cnt = total / 8;
                buf.asDoubleBuffer().get(m, off, cnt);
                off += cnt;
            }
        }
        return m;
    }

    public static void write(Path file, double[] m) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8 * m.length).order(ByteOrder.LITTLE_ENDIAN);
        buf.asDoubleBuffer().put(m);
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            while (buf.hasRemaining()) {
                ch.write(buf);
            }
        }
    }
}
