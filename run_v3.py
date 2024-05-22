import argparse
import warnings
from math import ceil

import numpy as np
import nvtx
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

from kernels import (
    BLOCKS_PER_GRID,
    THREADS_PER_BLOCK,
    divide_by,
    partial_reduce,
    single_thread_sum,
)

# Ignore NumbaPerformanceWarning
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


def run(size: int, nstreams: int):
    with nvtx.annotate("Compilation", color="red"):
        dev_a = cuda.device_array((BLOCKS_PER_GRID,), dtype=np.float32)
        dev_a_reduce = cuda.device_array((BLOCKS_PER_GRID,), dtype=dev_a.dtype)
        dev_a_sum = cuda.device_array((1,), dtype=dev_a.dtype)
        partial_reduce[BLOCKS_PER_GRID, THREADS_PER_BLOCK](dev_a, dev_a_reduce)
        single_thread_sum[1, 1](dev_a_reduce, dev_a_sum)
        divide_by[BLOCKS_PER_GRID, THREADS_PER_BLOCK](dev_a, dev_a_sum)

    # Define host array
    a = cuda.pinned_array(size, dtype=np.float32)
    a[...] = 1.0

    # Define regions for streams
    step = ceil(size / nstreams)
    starts = [i * step for i in range(nstreams)]
    ends = [min(s + step, size) for s in starts]
    print(f"Old sum: {a.sum():.3f}")

    # Create streams
    streams = [cuda.stream() for _ in range(nstreams)]
    # Ensure they are all different
    assert all(s1.handle != s2.handle for s1, s2 in zip(streams[:-1], streams[1:]))

    cpu_sums = [cuda.pinned_array(1, dtype=np.float32) for _ in range(nstreams)]
    devs_a = []
    with cuda.defer_cleanup():
        for i, (stream, start, end) in enumerate(zip(streams, starts, ends)):
            cpu_sums[i][...] = np.nan

            # Array copy to device and array creation on the device.
            with nvtx.annotate(f"H2D Memory Stream {i}", color="yellow"):
                dev_a = cuda.to_device(a[start:end], stream=stream)
                dev_a_reduce = cuda.device_array(
                    (BLOCKS_PER_GRID,), dtype=dev_a.dtype, stream=stream
                )
                dev_a_sum = cuda.device_array((1,), dtype=dev_a.dtype, stream=stream)
            devs_a.append(dev_a)

            # Launching kernels to sum array
            with nvtx.annotate(f"Sum Kernels Stream {i}", color="green"):
                for _ in range(50):  # Make it spend more time in compute
                    partial_reduce[BLOCKS_PER_GRID, THREADS_PER_BLOCK, stream](
                        dev_a, dev_a_reduce
                    )
                    single_thread_sum[1, 1, stream](dev_a_reduce, dev_a_sum)
            with nvtx.annotate(f"D2H Memory Stream {i}", color="orange"):
                dev_a_sum.copy_to_host(cpu_sums[i], stream=stream)

        # Ensure all streams are caught up
        cuda.synchronize()

        # Aggregate all 1D arrays into a single 1D array
        a_sum_all = sum(cpu_sums)

        # Send it to the GPU
        with cuda.pinned(a_sum_all):
            with nvtx.annotate("D2H Memory Default Stream", color="orange"):
                dev_a_sum_all = cuda.to_device(a_sum_all)

        # Normalize via streams
        for i, (stream, start, end, dev_a) in enumerate(
            zip(streams, starts, ends, devs_a)
        ):
            with nvtx.annotate(f"Divide Kernel Stream {i}", color="green"):
                divide_by[BLOCKS_PER_GRID, THREADS_PER_BLOCK, stream](
                    dev_a, dev_a_sum_all
                )

            # Array copy to host
            with nvtx.annotate(f"D2H Memory Stream {i}", color="orange"):
                dev_a.copy_to_host(a[start:end], stream=stream)

        cuda.synchronize()
        print(f"New sum: {a.sum():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Simple Example v3")
    parser.add_argument(
        "-n",
        "--array-size",
        type=int,
        default=100_000_000,
        metavar="N",
        help="Array size",
    )
    parser.add_argument(
        "-s",
        "--streams",
        type=int,
        default=4,
        metavar="N",
        help="Array size",
    )

    args = parser.parse_args()
    run(size=args.array_size, nstreams=args.streams)


if __name__ == "__main__":
    main()
