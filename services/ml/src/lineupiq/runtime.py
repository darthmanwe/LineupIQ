"""Process-level resource limits.

This module was written after a training run coincided with a workstation
freezing, and its original docstring said the run caused it. That turned out to
be wrong, and the correction is worth recording because it is the more useful
half of the story: the machine had bugchecked four times a month earlier, the
dump was a kernel-mode access violation (``0x1E``, ``0xC0000005``) that no
user-space process can produce, and a ninety-line script using nothing but
numpy and Python dicts reproduced the same faults with this repository entirely
out of the picture. The fault is hardware. Memory pressure is the load that
exposes it, not the cause.

What survives is narrower and still worth having. The conditional logit's
objective really was allocating thirteen ``(n_shots, n_zones)`` matrices per
L-BFGS iteration -- about 620 MB of churn per iteration at three seasons,
several hundred iterations per fit, eighteen fits per pass -- while a dense
feature matrix and its copy were also resident. Fixing the hot path was right,
but it is not a *guarantee*, because the next model added here will not have
been through it.

So a ceiling is set at the operating-system level, and the honest claim for it
is this: it bounds what this repository can ask of a machine, and it turns
runaway growth into a ``MemoryError`` with a traceback rather than a slow slide
into swap. It cannot make an unreliable machine reliable, and it should not be
described as though it could.

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
from typing import Any

__all__ = [
    "DEFAULT_MEMORY_CAP_GB",
    "DEFAULT_THREADS",
    "MemoryBudget",
    "MemoryCapResult",
    "cap_process_memory",
    "limit_thread_pools",
    "peak_process_memory_bytes",
    "thread_pool_report",
]

#: Share of physical memory the default cap allows.
#:
#: A fixed number is wrong in both directions: 6 GB is generous on a laptop and
#: absurdly tight on a 64 GB workstation, where it blocked a legitimate refit
#: and read as a reproducibility failure. A quarter of RAM leaves the machine
#: usable while still catching the runaway growth this module exists for.
_DEFAULT_CAP_SHARE = 0.25

#: Bounds on the derived default, in GB. The ceiling keeps a large machine from
#: setting a cap so high it stops being a limit.
_MAX_CAP_GB = 24.0

#: Floor on the derived default, and it is **platform-dependent for a real
#: reason**.
#:
#: The two mechanisms do not measure the same thing. A Windows job object's
#: ``ProcessMemoryLimit`` bounds *committed* memory. POSIX ``RLIMIT_AS`` bounds
#: *address space* -- which counts every reservation: each shared library mapped,
#: every arena a numeric allocator reserves up front and may never touch, and the
#: large sparse mappings BLAS makes. A process whose resident set peaks at 2.5 GB
#: routinely maps three or four times that.
#:
#: So the same number means something much stricter on Linux, and a 4 GB
#: ``RLIMIT_AS`` is not a generous guard rail -- it is tight enough that
#: importing scipy fails with ``failed to map segment from shared object``. That
#: is exactly how it surfaced: a CI job died with a ``MemoryError`` from a
#: retrieval evaluation that uses a few hundred megabytes.
#:
#: Treating a limit as portable because the number is the same is the mistake
#: here. The POSIX floor is higher to compensate, and it is still a limit -- it
#: catches runaway growth, which is what it is for.
_MIN_CAP_GB = 4.0 if sys.platform == "win32" else 10.0


def _physical_memory_gb() -> float | None:
    """Total installed RAM, or ``None`` when it cannot be determined."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return float(status.ullTotalPhys) / 1e9

    pages = os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else None
    size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else None
    if not pages or not size:
        return None
    return float(pages * size) / 1e9


def default_memory_cap_gb() -> float:
    """A cap proportional to the machine, clamped to sane bounds.

    Derived rather than fixed because a fixed number is wrong in both
    directions. This is called at import to populate
    :data:`DEFAULT_MEMORY_CAP_GB`, and re-callable for tests.
    """
    total = _physical_memory_gb()
    if total is None:
        return _MIN_CAP_GB * 2
    return max(_MIN_CAP_GB, min(_MAX_CAP_GB, round(total * _DEFAULT_CAP_SHARE, 1)))


DEFAULT_MEMORY_CAP_GB = default_memory_cap_gb()

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
    # Polars reads its own variable, but other Rust extensions read this one.
    "RAYON_NUM_THREADS",
)

#: Threads the numeric pools are allowed. Deliberately small relative to a
#: modern core count.
#:
#: This is a hardware-safety limit, not a performance tuning knob. A sustained
#: all-core AVX load is the worst thing you can ask of a 13th-generation Intel
#: part, and this workstation has a documented history of it: four bugchecks in
#: one day last July, a WHEA corrected machine-check from Processor Core, and a
#: kernel-mode access violation during this project. The operator's mitigation
#: (undervolt plus a BIOS update) held for a year under gaming loads, which are
#: bursty and GPU-bound; a cross-validation refit is neither. Four threads of
#: thirty-two keeps this repository off that failure mode entirely, and the cost
#: is wall-clock time on a job that runs nightly in CI anyway.
DEFAULT_THREADS = 4


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


def limit_thread_pools(threads: int = DEFAULT_THREADS) -> dict[str, str]:
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


def _kernel32() -> Any:
    """``kernel32`` with every signature declared.

    Declaring them is not optional. ctypes types an undeclared return value as
    a 32-bit int, which truncates a 64-bit ``HANDLE``; every later call then
    fails with ``ERROR_INVALID_HANDLE``, a failure that reads exactly like a
    permissions problem and is not one.
    """
    # The guard is for mypy as much as for correctness. `ctypes.WinDLL` does not
    # exist on POSIX, and `warn_unused_ignores` is on -- so a `type: ignore` that
    # silences the Linux leg of the CI matrix fails the Windows leg. Narrowing on
    # `sys.platform` typechecks correctly on both, because mypy treats the other
    # branch as unreachable for the platform it is checking.
    if sys.platform != "win32":  # pragma: no cover - guarded by every caller
        raise RuntimeError("kernel32 is only available on Windows")

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


def _extended_limit_struct() -> Any:
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

    # Same reason as `_kernel32`: `ctypes.get_last_error` is Windows-only, and a
    # platform-specific ignore breaks the other leg of the matrix.
    if sys.platform != "win32":  # pragma: no cover - guarded by `cap_process_memory`
        raise RuntimeError("job objects are only available on Windows")

    import ctypes

    kernel32 = _kernel32()
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        return MemoryCapResult(
            False,
            cap_bytes / 1e9,
            "Windows job object",
            f"CreateJobObjectW failed: {ctypes.get_last_error()}",
        )

    info = _extended_limit_struct()()
    info.BasicLimitInformation.LimitFlags = _JOB_LIMIT_PROCESS_MEMORY | _JOB_LIMIT_KILL_ON_JOB_CLOSE
    info.ProcessMemoryLimit = cap_bytes

    if not kernel32.SetInformationJobObject(
        handle, _JOB_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
    ):
        return MemoryCapResult(
            False,
            cap_bytes / 1e9,
            "Windows job object",
            f"SetInformationJobObject failed: {ctypes.get_last_error()}",
        )

    if not kernel32.AssignProcessToJobObject(
        handle,
        kernel32.GetCurrentProcess(),
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
    info = _extended_limit_struct()()
    if not kernel32.QueryInformationJobObject(
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


def thread_pool_report() -> str:
    """What the numeric pools are *actually* running, not what was requested.

    ``limit_thread_pools`` reports the variables it set, which is not the same
    thing: every one of these libraries reads its pool size once at import, so
    setting the variable after the import silently does nothing. Reading the
    live value back is the only way to know the limit took, and this is a
    hardware-safety limit -- one that silently failed would be worse than none,
    because the operator would believe they were protected.
    """
    parts: list[str] = []
    try:
        import polars as pl

        parts.append(f"polars {pl.thread_pool_size()}")
    except Exception:  # pragma: no cover - polars is a hard dependency
        pass
    omp = os.environ.get("OMP_NUM_THREADS")
    if omp:
        parts.append(f"OpenMP {omp}")
    total = os.cpu_count() or 0
    return f"{', '.join(parts)} of {total} logical CPUs" if parts else "unknown"
