"""The artefact comparator: what it lets through, and what it must not.

This is the gate three of this repository's checks now run on, so the interesting
tests are the negative ones. A comparator that is too permissive turns every
reproducibility check into a formality, and it would never say so — everything
would simply keep passing.
"""

from __future__ import annotations

from lineupiq.validate.reproduce import compare_artefacts


def _paths(committed: object, fresh: object, *, tolerance: float) -> list[str]:
    return [d.path for d in compare_artefacts("a", committed, fresh, tolerance=tolerance)]


def test_a_float_inside_the_tolerance_is_not_drift() -> None:
    a = {"correlation": 0.9755979585883191}
    b = {"correlation": 0.9755979585883203}
    assert _paths(a, b, tolerance=1e-12) == []


def test_a_float_outside_the_tolerance_is_drift() -> None:
    a = {"correlation": 0.9755}
    b = {"correlation": 0.9760}
    assert _paths(a, b, tolerance=1e-12) == [".correlation"]


def test_zero_tolerance_means_exact() -> None:
    """What the hash-and-tier fixture gets. One bit is a bug there."""
    a = {"lineup_hash": 1.0}
    b = {"lineup_hash": 1.0 + 2**-52}
    assert _paths(a, b, tolerance=0.0) == [".lineup_hash"]


def test_a_flipped_boolean_is_never_a_rounding_difference() -> None:
    """`bool` subclasses `int`, so a numeric comparison would let this through.

    `True` and `False` differ by 1, which is outside any tolerance here — but a
    tolerance of 2 would admit it, and a flag flipping is categorically not a
    rounding difference. It is checked for identity before the numeric branch.
    """
    assert _paths({"counterfactual": True}, {"counterfactual": False}, tolerance=2.0) == [
        ".counterfactual"
    ]


def test_a_changed_label_is_drift_at_any_tolerance() -> None:
    a = {"tier": "reportable"}
    b = {"tier": "directional"}
    assert _paths(a, b, tolerance=1e9) == [".tier"]


def test_a_missing_key_is_reported_by_name() -> None:
    assert _paths({"a": 1, "b": 2}, {"a": 1}, tolerance=1e-9) == [".b"]
    assert _paths({"a": 1}, {"a": 1, "b": 2}, tolerance=1e-9) == [".b"]


def test_a_list_that_changed_length_is_reported_once() -> None:
    """Not once per element. A grown list is one fact, not fifty."""
    assert _paths({"xs": [1, 2, 3]}, {"xs": [1, 2]}, tolerance=1e-9) == [".xs[len]"]


def test_nested_paths_locate_the_disagreement() -> None:
    a = {"co_occurrence": {"non_identified": [{"player_id": 1}, {"player_id": 2}]}}
    b = {"co_occurrence": {"non_identified": [{"player_id": 1}, {"player_id": 9}]}}
    assert _paths(a, b, tolerance=1e-9) == [".co_occurrence.non_identified[1].player_id"]


def test_a_type_change_is_drift() -> None:
    """A number becoming a string is not a value that moved."""
    assert _paths({"n": 5}, {"n": "5"}, tolerance=1e9) == [".n"]


def test_null_appearing_where_a_number_was_is_drift() -> None:
    # This is how a standard error going missing would surface, and it must not
    # be silently tolerated: `None` is not a number near zero.
    assert _paths({"se": 0.0043}, {"se": None}, tolerance=1e9) == [".se"]


def test_identical_artefacts_produce_no_drift() -> None:
    artefact = {
        "n": 3,
        "ok": True,
        "tier": "reportable",
        "values": [1.5, 2.5],
        "nested": {"a": [{"b": 1}]},
    }
    assert _paths(artefact, artefact, tolerance=0.0) == []
