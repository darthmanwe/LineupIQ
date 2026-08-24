"""Comparing a regenerated artefact to a committed one.

``git diff --exit-code`` is the obvious gate for "did this file change", and for
some of the files here it is the right one: gold checksums, MD5 digests, canonical
id strings and tier labels are exact quantities, and a single differing bit is a
bug.

It is the wrong gate for a float. A run log holds correlations, standard errors,
softmax outputs and ridge solutions -- results of logs, exponentials and matrix
products -- and none of those are bit-portable. The same source with the same
library versions differs in the last place between Linux and Windows because the
BLAS and libm underneath do. Requiring byte-identity of such a file gets the gate
exactly backwards: **it fails on a platform change and passes on a rounding
coincidence.**

This module is the comparison in between. Structure is exact -- a missing key, a
changed label, a list that grew, a flipped boolean are differences at any
tolerance -- and floats are allowed to move by a stated amount. Every caller
names its own tolerance, so the choice is visible at the call site rather than
buried here.

Learned the hard way, twice: the selection parity fixture and the RAPM run log
both failed a byte-identity gate on differences of order 1e-15, after the real
reproducibility bugs had already been found and fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Drift", "compare_artefacts"]


@dataclass(frozen=True)
class Drift:
    """One place a regenerated artefact disagrees with the committed one."""

    artefact: str
    path: str
    committed: object
    fresh: object

    def __str__(self) -> str:
        return f"{self.artefact}: {self.path or '<root>'}: {self.committed!r} -> {self.fresh!r}"


def compare_artefacts(
    artefact: str,
    committed: object,
    fresh: object,
    *,
    tolerance: float,
    path: str = "",
    drifts: list[Drift] | None = None,
) -> list[Drift]:
    """Walk two decoded artefacts, allowing floats to differ by ``tolerance``.

    ``tolerance`` of ``0.0`` gives exact comparison, which is what an artefact of
    hashes and integers should get.
    """
    out = [] if drifts is None else drifts

    if isinstance(committed, dict) and isinstance(fresh, dict):
        for key in sorted(set(committed) | set(fresh)):
            if key not in committed or key not in fresh:
                out.append(Drift(artefact, f"{path}.{key}", key in committed, key in fresh))
                continue
            compare_artefacts(
                artefact,
                committed[key],
                fresh[key],
                tolerance=tolerance,
                path=f"{path}.{key}",
                drifts=out,
            )
        return out

    if isinstance(committed, list) and isinstance(fresh, list):
        if len(committed) != len(fresh):
            out.append(Drift(artefact, f"{path}[len]", len(committed), len(fresh)))
            return out
        for i, (a, b) in enumerate(zip(committed, fresh, strict=True)):
            compare_artefacts(artefact, a, b, tolerance=tolerance, path=f"{path}[{i}]", drifts=out)
        return out

    # `bool` subclasses `int`, and a flag flipping is never a rounding
    # difference -- so it is checked for identity before the numeric branch.
    if isinstance(committed, bool) or isinstance(fresh, bool):
        if committed is not fresh:
            out.append(Drift(artefact, path, committed, fresh))
        return out

    if isinstance(committed, (int, float)) and isinstance(fresh, (int, float)):
        if abs(float(committed) - float(fresh)) > tolerance:
            out.append(Drift(artefact, path, committed, fresh))
        return out

    if committed != fresh:
        out.append(Drift(artefact, path, committed, fresh))
    return out
