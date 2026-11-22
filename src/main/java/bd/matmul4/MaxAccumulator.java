package bd.matmul4;

import org.apache.spark.util.AccumulatorV2;

/** Accumulator that keeps the maximum of the values added (used for the peak resident memory of the executors). */
public final class MaxAccumulator extends AccumulatorV2<Long, Long> {
    private static final long serialVersionUID = 1L;
    private long max;

    @Override
    public boolean isZero() {
        return max == 0L;
    }

    @Override
    public AccumulatorV2<Long, Long> copy() {
        MaxAccumulator m = new MaxAccumulator();
        m.max = max;
        return m;
    }

    @Override
    public void reset() {
        max = 0L;
    }

    @Override
    public void add(Long v) {
        if (v > max) {
            max = v;
        }
    }

    @Override
    public void merge(AccumulatorV2<Long, Long> other) {
        add(other.value());
    }

    @Override
    public Long value() {
        return max;
    }
}
