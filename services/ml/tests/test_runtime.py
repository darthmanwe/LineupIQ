"""Tests for the process resource limits.

These exist because a training run drove this machine into swap and took the
desktop down with it. The allocation bug behind that is fixed, but a fix to one
hot path is not a guarantee about the next model somebody adds, so the guarantee
is enforced by the operating system and asserted here.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from lineupiq.runtime import (
    DEFAULT_MEMORY_CAP_GB,
    MemoryBudget,
    MemoryCapResult,
    limit_thread_pools,
)


def test_thread_pools_are_bounded_at_import() -> None:
    """Importing the package must already have bounded the pools.

    They have to be set before numpy, polars or scikit-learn load, because each
    reads its pool size exactly once at import. If this fails, the limits are
    being applied too late to do anything.
    """
    import lineupiq  # noqa: F401 -- imported for its side effect

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
        assert os.environ.get(name), f"{name} was never set"


def test_limit_thread_pools_respects_an_existing_setting() -> None:
    """An operator's own choice must win over the library's default."""
    os.environ["OMP_NUM_THREADS"] = "11"
    try:
        applied = limit_thread_pools(4)
        assert "OMP_NUM_THREADS" not in applied
        assert os.environ["OMP_NUM_THREADS"] == "11"
    finally:
        os.environ["OMP_NUM_THREADS"] = "4"


def test_budget_scales_with_rows_and_zones() -> None:
    small = MemoryBudget(n_rows=1000, n_zones=9, n_pair_matrices=2, n_wide_columns=28)
    large = MemoryBudget(n_rows=100_000, n_zones=9, n_pair_matrices=2, n_wide_columns=28)
    assert large.peak_bytes == 100 * small.peak_bytes
    # One (n, n_zones) float64 matrix per pair term.
    assert small.design_bytes == 1000 * 9 * 2 * 8


def test_the_real_workload_fits_far_under_the_default_cap() -> None:
    """Three seasons must not be anywhere near the ceiling.

    If this ever starts failing, the model grew and the cap needs a deliberate
    decision rather than a quiet bump.
    """
    budget = MemoryBudget(n_rows=671_255, n_zones=9, n_pair_matrices=2, n_wide_columns=28)
    assert not budget.exceeds(DEFAULT_MEMORY_CAP_GB)
    assert budget.peak_bytes < 0.5e9


def test_an_absurd_workload_is_refused() -> None:
    """The guard has to be able to say no, or it is decoration."""
    budget = MemoryBudget(n_rows=200_000_000, n_zones=9, n_pair_matrices=2, n_wide_columns=28)
    assert budget.exceeds(DEFAULT_MEMORY_CAP_GB)


def test_cap_result_reports_failure_loudly() -> None:
    """A cap that could not be applied must not read like success."""
    failed = MemoryCapResult(False, 6.0, "Windows job object", "AssignProcessToJobObject failed: 5")
    assert "NO memory cap" in str(failed)
    assert MemoryCapResult(True, 6.0, "Windows job object").applied


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job object path")
def test_the_cap_actually_stops_an_over_large_allocation() -> None:
    """End-to-end: a capped child process must die instead of the machine.

    Run in a subprocess because the limit is irreversible for the process that
    sets it -- imposing a 256 MB ceiling on the test runner would take the rest
    of the suite with it.
    """
    script = (
        "from lineupiq.runtime import cap_process_memory\n"
        "result = cap_process_memory(0.30)\n"
        "assert result.applied, result\n"
        "buffers = []\n"
        "try:\n"
        "    for _ in range(50):\n"
        "        buffers.append(bytearray(64 * 1024 * 1024))\n"
        "except MemoryError:\n"
        "    print('CAPPED')\n"
        "else:\n"
        "    print('UNCAPPED')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    # Either Python raised MemoryError and said so, or the job object killed the
    # process outright. Both are the guarantee holding; "UNCAPPED" is not.
    assert "UNCAPPED" not in completed.stdout
    assert "CAPPED" in completed.stdout or completed.returncode != 0
