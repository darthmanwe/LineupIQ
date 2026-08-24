"""Process-level resource limits.

This module exists because a training run took a workstation down. The
conditional logit's objective was allocating thirteen ``(n_shots, n_zones)``
matrices per L-BFGS iteration -- about 620 MB of churn per iteration at three
seasons, several hundred iterations per fit, eighteen fits per pass -- while a
dense feature matrix and its copy were also resident. The fix to the hot path
was the right fix. It is not a *guarantee*, because the next model added to this
repository will not have been through it.

So the guarantee is enforced by the operating system instead. A hard cap on the
process's committed memory turns "the machine froze" into "Python raised
MemoryError", which is a bug report rather than a lost afternoon.

Two mechanisms, one per platform:

- **Windows** -- a Job Object with ``JOB_OBJECT_LIMIT_PROCESS_MEMORY``. Windows 8
  and later allow nested jobs, so this works even when the shell has already
  placed the process in one.
- **POSIX** -- ``RLIMIT_AS``.

If neither can be applied the caller is told, and it is the caller's decision
whether to proceed. Silently continuing unprotected is the one behaviour this
module must not have.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

__all__ = [
    "DEFAULT_MEMORY_CAP_GB",
    "MemoryBudget",
    "MemoryCapResult",
    "cap_process_memory",
    "limit_thread_pools",
    "peak_process_memory_bytes",
]

#: Default ceiling. Comfortably above what a three-season fit needs (measured
#: peak is around 170 MB of Python allocation, and roughly 600 MB resident
#: including polars frames and sklearn's binned copies), and comfortably below
#: anything that would put a desktop into swap.
DEFAULT_MEMORY_CAP_GB = 6.0

#: Thread-pool environment variables, and the reason they are bounded.
#:
#: Each BLAS or OpenMP worker carries its own scratch buffers, so peak memory
#: scales with the pool size, not just with the data. On a many-core desktop the
#: default pool is one thread per core and the footprint is multiplied by that.
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "POLARS_MAX_THREADS",
)


@dataclass(frozen=True)
class MemoryCapResult:
    """Whether a cap was actually applied, and by what."""

    applied: bool
    cap_gb: float
    mechanism: str
    detail: str = ""

    def __str__(self) -> str:
        if self.applied:
            return f"memory capped at {self.cap_gb:.1f} GB via {self.mechanism}"
        return f"NO memory cap applied ({self.mechanism}: {self.detail})"


def limit_thread_pools(threads: int = 4) -> dict[str, str]:
    """Bound every numeric thread pool.

    Must run before numpy, polars or scikit-learn are imported -- each reads its
    pool size once, at import. Returns the variables that were set, so a caller
    can report what happened rather than assume it worked.

    Existing values are respected: an operator who has deliberately set
    ``OMP_NUM_THREADS`` should not have it overwritten by a library.
    """
    applied: dict[str, str] = {}
    for name in _THREAD_VARIABLES:
        if name not in os.environ:
            os.environ[name] = str(threads)
            applied[name] = str(threads)
    return applied


#: ``JobObjectExtendedLimitInformation``.
_JOB_EXTENDED_LIMIT_INFORMATION = 9
_JOB_LIMIT_PROCESS_MEMORY = 0x00000100
#: Without this the job outlives the shell and can capture unrelated processes.
_JOB_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


def _kernel32() -> object:
    """``kernel32`` with every signature declared.

    Declaring them is not optional. ctypes types an undeclared return value as
    a 32-bit int, which truncates a 64-bit ``HANDLE``; every later call then
    fails with ``ERROR_INVALID_HANDLE``, a failure that reads exactly like a
    permissions problem and is not one.
    """
    import ctypes
    from ctypes import wintypes

    lib = ctypes.WinDLL("kernel32", use_last_error=True)
    lib.CreateJobObjectW.restype = wintypes.HANDLE
    lib.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    lib.SetInformationJobObject.restype = wintypes.BOOL
    lib.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    lib.QueryInformationJobObject.restype = wintypes.BOOL
    lib.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPVOID,
    ]
    lib.AssignProcessToJobObject.restype = wintypes.BOOL
    lib.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    lib.GetCurrentProcess.restype = wintypes.HANDLE
    lib.GetCurrentProcess.argtypes = []
    return lib


def _extended_limit_struct() -> object:
    """Build ``JOBOBJECT_EXTENDED_LIMIT_INFORMATION`` lazily.

    Defined at call time rather than at module scope so importing this module on
    a non-Windows platform never touches ``ctypes.wintypes``.
    """
    import ctypes
    from ctypes import wintypes

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    return ExtendedLimitInformation


def _cap_windows(cap_bytes: int) -> MemoryCapResult:
    global _JOB_HANDLE
    import ctypes

    kernel32 = _kernel32()
    handle = kernel32.CreateJobObjectW(None, None)  # type: ignore[attr-defined]
    if not handle:
        return MemoryCapResult(
            False,
            cap_bytes / 1e9,
            "Windows job object",
            f"CreateJobObjectW failed: {ctypes.get_last_error()}",
        )

    info = _extended_limit_struct()()  # type: ignore[operator]
    info.BasicLimitInformation.LimitFlags = _JOB_LIMIT_PROCESS_MEMORY | _JOB_LIMIT_KILL_ON_JOB_CLOSE
    info.ProcessMemoryLimit = cap_bytes

    if not kernel32.SetInformationJobObject(  # type: ignore[attr-defined]
        handle, _JOB_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
    ):
        return MemoryCapResult(
            False,
            cap_bytes / 1e9,
            "Windows job object",
            f"SetInformationJobObject failed: {ctypes.get_last_error()}",
        )

    if not kernel32.AssignProcessToJobObject(  # type: ignore[attr-defined]
        handle,
        kernel32.GetCurrentProcess(),  # type: ignore[attr-defined]
    ):
        return MemoryCapResult(
            False,
            cap_bytes / 1e9,
            "Windows job object",
            f"AssignProcessToJobObject failed: {ctypes.get_last_error()}",
        )

    # The handle is deliberately held for the process lifetime: closing it with
    # KILL_ON_JOB_CLOSE set would terminate this process immediately. It is also
    # what `peak_process_memory_bytes` reads back from.
    _JOB_HANDLE = int(handle)
    return MemoryCapResult(True, cap_bytes / 1e9, "Windows job object")


#: Set by :func:`cap_process_memory` on Windows so peak usage can be queried
#: later. The handle is intentionally held for the process lifetime.
_JOB_HANDLE: int | None = None


def peak_process_memory_bytes() -> int | None:
    """Peak committed memory this process has used, or ``None`` if unknown.

    Read from the same Job Object that enforces the cap, which makes it the
    honest number rather than an estimate: ``tracemalloc`` sees only Python
    allocations and misses everything numpy, polars and scikit-learn commit
    through their own allocators -- which is most of it, and was exactly the gap
    that let a run be predicted at 314 MB and then segfault.
    """
    if _JOB_HANDLE is None or sys.platform != "win32":
        return None

    import ctypes

    kernel32 = _kernel32()
    info = _extended_limit_struct()()  # type: ignore[operator]
    if not kernel32.QueryInformationJobObject(  # type: ignore[attr-defined]
        ctypes.c_void_p(_JOB_HANDLE),
        _JOB_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(info),
        ctypes.sizeof(info),
        None,
    ):
        return None
    return int(info.PeakProcessMemoryUsed)


def _cap_posix(cap_bytes: int) -> MemoryCapResult:
    # `resource` is POSIX-only and its type stubs are absent when mypy runs on
    # Windows, so the constants are reached through getattr. A `type: ignore`
    # would be the obvious alternative and is worse: `warn_unused_ignores` is on,
    # so an ignore needed on Windows fails the Linux leg of the CI matrix.
    import resource

    rlimit_as = getattr(resource, "RLIMIT_AS", None)
    infinity = getattr(resource, "RLIM_INFINITY", -1)
    get_limit = getattr(resource, "getrlimit", None)
    set_limit = getattr(resource, "setrlimit", None)
    if rlimit_as is None or get_limit is None or set_limit is None:
        return MemoryCapResult(
            False, cap_bytes / 1e9, "RLIMIT_AS", "not available on this platform"
        )

    try:
        _, hard = get_limit(rlimit_as)
        ceiling = cap_bytes if hard == infinity else min(cap_bytes, hard)
        set_limit(rlimit_as, (ceiling, hard))
    except (ValueError, OSError) as exc:
        return MemoryCapResult(False, cap_bytes / 1e9, "RLIMIT_AS", str(exc))
    return MemoryCapResult(True, ceiling / 1e9, "RLIMIT_AS")


def cap_process_memory(cap_gb: float = DEFAULT_MEMORY_CAP_GB) -> MemoryCapResult:
    """Impose a hard ceiling on this process's memory.

    Past the cap an allocation fails and Python raises ``MemoryError``. That is
    the entire point: a training run that asks for too much should die and say
    so, not drive the machine into swap and take the desktop with it.
    """
    cap_bytes = int(cap_gb * 1e9)
    if sys.platform == "win32":
        return _cap_windows(cap_bytes)
    return _cap_posix(cap_bytes)


@dataclass(frozen=True)
class MemoryBudget:
    """A pre-flight estimate of what a fit will need.

    Checked *before* allocating, so an over-large job is refused with a number
    attached instead of discovered by the operating system. The estimate does
    not need to be tight -- it needs to catch the order-of-magnitude mistake,
    which is the kind that hurts.
    """

    n_rows: int
    n_zones: int
    n_pair_matrices: int
    n_wide_columns: int
    bytes_per_float: int = 8

    @property
    def design_bytes(self) -> int:
        """The factored design: pair matrices plus the interaction vectors."""
        return self.n_rows * self.n_zones * self.n_pair_matrices * self.bytes_per_float

    @property
    def fit_bytes(self) -> int:
        """Working set of one conditional-logit fit.

        One reused ``(n, n_zones)`` buffer, one scratch matrix of the same
        shape, plus the design itself.
        """
        matrix = self.n_rows * self.n_zones * self.bytes_per_float
        return self.design_bytes + 2 * matrix

    @property
    def gbdt_bytes(self) -> int:
        """Working set of the boosted reference.

        The dense float matrix, plus scikit-learn's uint8 binned copy, plus the
        per-class probability output.
        """
        dense = self.n_rows * self.n_wide_columns * self.bytes_per_float
        binned = self.n_rows * self.n_wide_columns
        probabilities = self.n_rows * self.n_zones * self.bytes_per_float
        return dense + binned + probabilities

    @property
    def peak_bytes(self) -> int:
        """The larger of the two phases; they do not overlap."""
        return max(self.fit_bytes, self.design_bytes + self.gbdt_bytes)

    def describe(self) -> str:
        return (
            f"{self.n_rows:,} rows -- design {self.design_bytes / 1e6:.0f} MB, "
            f"logit fit {self.fit_bytes / 1e6:.0f} MB, "
            f"gbdt {self.gbdt_bytes / 1e6:.0f} MB, "
            f"estimated peak {self.peak_bytes / 1e6:.0f} MB"
        )

    def exceeds(self, cap_gb: float, *, headroom: float = 0.5) -> bool:
        """True when the estimate does not fit under ``cap_gb``.

        ``headroom`` reserves the rest of the cap for everything the estimate
        does not model: the polars frames, the interpreter, sklearn's trees.
        """
        return self.peak_bytes > cap_gb * 1e9 * headroom
