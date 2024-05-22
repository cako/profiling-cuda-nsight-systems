from time import perf_counter_ns


def humanize(elapsed_ns: float, fmt: str):
    if elapsed_ns < 1e3:
        return fmt % elapsed_ns + " ns"
    elif elapsed_ns < 1e6:
        return fmt % (1e-3 * elapsed_ns) + " µs"
    elif elapsed_ns < 1e9:
        return fmt % (1e-6 * elapsed_ns) + " ms"
    else:
        return fmt % (1e-9 * elapsed_ns) + " s"


class TicToc:
    def __init__(self, fmt: str = "%0.4f"):
        self.fmt = fmt
        self.elapsed_ns = -1

    def __enter__(self):
        self._tic = perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._toc = perf_counter_ns()
        self.elapsed_ns = self._toc - self._tic

    def __str__(self) -> str:
        return humanize(self.elapsed_ns, fmt=self.fmt)
