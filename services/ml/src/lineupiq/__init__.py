"""LineupIQ -- NBA lineup and trade forecasting with a published refusal contract."""

from __future__ import annotations

from lineupiq.runtime import limit_thread_pools

__version__ = "0.1.0"

# Bound the numeric thread pools before numpy, polars or scikit-learn are
# imported. Each reads its pool size exactly once, at import, so this has to
# happen here -- the package root is the earliest point that runs for every
# entry point, including `python -c "import lineupiq"` and pytest collection.
#
# Peak memory scales with the pool size, not only with the data: every BLAS and
# OpenMP worker carries its own scratch buffers. On a many-core desktop the
# default is one thread per core, which multiplies the footprint by a number
# nobody chose. An operator's own setting is always respected.
#
# `lineupiq.runtime` deliberately imports nothing but the standard library, so
# this cannot become a circular import.
_THREAD_LIMITS = limit_thread_pools(4)

__all__ = ["__version__"]
