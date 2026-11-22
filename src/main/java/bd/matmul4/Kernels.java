package bd.matmul4;

import java.util.Arrays;

/**
 * Dense float64 kernels on flat row-major arrays. The multiply-add loop is the 'unroll4' kernel of
 * Assignments 2 and 3 (ikj order, k loop unrolled by 4, the four products added one after the other
 * to C[i][j]), generalised to rectangular blocks so that every method of this study uses the same
 * arithmetic. Java never contracts a*b+c into a fused multiply-add (strict IEEE-754 since JDK 17), so
 * the result of C[i][j] does not depend on the machine.
 */
public final class Kernels {
    private Kernels() { }

    /** Tile sizes of the blocked kernel: a KB x JB tile of B (128 KiB) stays in the L2 cache while all rows of the band use it. */
    public static final int KB = 64, JB = 256;

    /** c (r x cols) += a (r x kk) * b (kk x cols), unroll4 kernel (reads all of B for every row); all row-major. */
    public static void mulAdd(double[] a, int r, int kk, double[] b, int cols, double[] c) {
        for (int i = 0; i < r; i++) {
            rowUnroll4(a, i * kk, b, c, i * cols, kk, cols);
        }
    }

    /**
     * Same product, cache-blocked: tiles of KB x JB of B are reused by every row of A before the next tile is loaded.
     * For every C[i][j] the products are still added in increasing k, one after the other, starting from the existing
     * value of C[i][j]: the result is bit-identical to mulAdd (KB is a multiple of 4, so the unrolled groups are the same).
     */
    public static void mulAddBlocked(double[] a, int r, int kk, double[] b, int cols, double[] c) {
        blockedRows(a, kk, b, c, cols, 0, r, kk);
    }

    private static void blockedRows(double[] a, int lda, double[] b, double[] c, int cols, int r0, int r1, int kk) {
        for (int k0 = 0; k0 < kk; k0 += KB) {
            int k1 = Math.min(kk, k0 + KB);
            for (int j0 = 0; j0 < cols; j0 += JB) {
                int j1 = Math.min(cols, j0 + JB);
                for (int i = r0; i < r1; i++) {
                    tile(a, i * lda, b, c, i * cols, k0, k1, j0, j1, cols);
                }
            }
        }
    }

    private static void tile(double[] a, int aOff, double[] b, double[] c, int cOff, int k0, int k1, int j0, int j1, int cols) {
        int k = k0;
        for (; k + 3 < k1; k += 4) {
            final double a0 = a[aOff + k], a1 = a[aOff + k + 1], a2 = a[aOff + k + 2], a3 = a[aOff + k + 3];
            final int b0 = k * cols, b1 = b0 + cols, b2 = b1 + cols, b3 = b2 + cols;
            for (int j = j0; j < j1; j++) {
                double v = c[cOff + j];
                v += a0 * b[b0 + j];
                v += a1 * b[b1 + j];
                v += a2 * b[b2 + j];
                v += a3 * b[b3 + j];
                c[cOff + j] = v;
            }
        }
        for (; k < k1; k++) {
            final double ak = a[aOff + k];
            final int bk = k * cols;
            for (int j = j0; j < j1; j++) {
                c[cOff + j] += ak * b[bk + j];
            }
        }
    }

    /** Dispatch on the kernel name: "unroll4" (Assignments 2 and 3) or "blocked". */
    public static void mulAdd(String kernel, double[] a, int r, int kk, double[] b, int cols, double[] c) {
        if (kernel.equals("blocked")) {
            mulAddBlocked(a, r, kk, b, cols, c);
        } else {
            mulAdd(a, r, kk, b, cols, c);
        }
    }

    /** One row of C: c[cOff .. cOff+cols) += a[aOff .. aOff+kk) * b (kk x cols). */
    public static void rowUnroll4(double[] a, int aOff, double[] b, double[] c, int cOff, int kk, int cols) {
        int k = 0;
        for (; k + 3 < kk; k += 4) {
            final double a0 = a[aOff + k], a1 = a[aOff + k + 1], a2 = a[aOff + k + 2], a3 = a[aOff + k + 3];
            final int b0 = k * cols, b1 = b0 + cols, b2 = b1 + cols, b3 = b2 + cols;
            for (int j = 0; j < cols; j++) {
                double v = c[cOff + j];
                v += a0 * b[b0 + j];
                v += a1 * b[b1 + j];
                v += a2 * b[b2 + j];
                v += a3 * b[b3 + j];
                c[cOff + j] = v;
            }
        }
        for (; k < kk; k++) {
            final double ak = a[aOff + k];
            final int bk = k * cols;
            for (int j = 0; j < cols; j++) {
                c[cOff + j] += ak * b[bk + j];
            }
        }
    }

    /** Rows [r0, r1) of C = A * B for square n x n matrices; C is zeroed first, then the selected kernel runs on the band. */
    public static void rows(String kernel, double[] a, double[] b, double[] c, int n, int r0, int r1) {
        Arrays.fill(c, r0 * n, r1 * n, 0.0);
        if (kernel.equals("blocked")) {
            blockedRows(a, n, b, c, n, r0, r1, n);
        } else {
            for (int i = r0; i < r1; i++) {
                rowUnroll4(a, i * n, b, c, i * n, n, n);
            }
        }
    }

    /** Sequential baseline: the selected kernel on the whole matrix. */
    public static void seq(String kernel, double[] a, double[] b, double[] c, int n) {
        rows(kernel, a, b, c, n, 0, n);
    }

    /** The basic triple loop (i, j, k) of Assignment 1: C[i][j] = sum_k A[i][k] * B[k][j]. */
    public static void ijk(double[] a, double[] b, double[] c, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = s;
            }
        }
    }

    /** Split [0, n) into `parts` contiguous ranges of nearly equal size; returns parts + 1 boundaries. */
    public static int[] bounds(int n, int parts) {
        int[] b = new int[parts + 1];
        for (int p = 0; p <= parts; p++) {
            b[p] = (int) ((long) n * p / parts);
        }
        return b;
    }

    /** {sum of the entries, sum of the absolute values}, in index order. */
    public static double[] checksum(double[] c) {
        double s = 0.0, sa = 0.0;
        for (double v : c) {
            s += v;
            sa += Math.abs(v);
        }
        return new double[] {s, sa};
    }
}
