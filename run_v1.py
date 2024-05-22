import argparse
import warnings

import numpy as np
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


def run(size: int):
    # Define host array
    a = np.ones(size, dtype=np.float32)
    print(f"Old sum: {a.sum():.3f}")

    # Array copy to device and array creation on the device.
    dev_a = cuda.to_device(a)
    dev_a_reduce = cuda.device_array((BLOCKS_PER_GRID,), dtype=dev_a.dtype)
    dev_a_sum = cuda.device_array((1,), dtype=dev_a.dtype)

    # Launching kernels to normalize array
    partial_reduce[BLOCKS_PER_GRID, THREADS_PER_BLOCK](dev_a, dev_a_reduce)
    single_thread_sum[1, 1](dev_a_reduce, dev_a_sum)
    divide_by[BLOCKS_PER_GRID, THREADS_PER_BLOCK](dev_a, dev_a_sum)

    # Array copy to host
    dev_a.copy_to_host(a)
    cuda.synchronize()
    print(f"New sum: {a.sum():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Simple Example v1")
    parser.add_argument(
        "-n",
        "--array-size",
        type=int,
        default=100_000_000,
        metavar="N",
        help="Array size",
    )

    args = parser.parse_args()
    run(size=args.array_size)


if __name__ == "__main__":
    main()
