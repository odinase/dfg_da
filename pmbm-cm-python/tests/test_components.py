"""Unit checks for the standalone components of the PMBM port."""

import numpy as np

from pmbm.cardinality import cardinality_mb
from pmbm.gospa import gospa
from pmbm.murty import murty


def test_cardinality_sums_to_one():
    r = np.array([0.2, 0.5, 0.9, 0.1])
    pcard = cardinality_mb(r)
    assert pcard.size == r.size + 1
    assert abs(pcard.sum() - 1.0) < 1e-12


def test_cardinality_known_case():
    # Two independent Bernoullis with r=0.5: P(0)=P(2)=0.25, P(1)=0.5.
    pcard = cardinality_mb([0.5, 0.5])
    assert np.allclose(pcard, [0.25, 0.5, 0.25], atol=1e-12)


def test_murty_best_matches_optimal():
    C = np.array([[7.0, 5.0, 11.0],
                  [5.0, 4.0, 1.0],
                  [9.0, 3.0, 2.0]])
    assigns, costs = murty(C, 6)
    # Costs must be sorted ascending and the first must be the optimal.
    assert np.all(np.diff(costs) >= -1e-9)
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(C)
    assert abs(costs[0] - C[r, c].sum()) < 1e-9


def test_murty_count_and_uniqueness():
    C = np.array([[1.0, 2.0], [2.0, 1.0]])
    assigns, costs = murty(C, 10)
    # A 2x2 problem has exactly 2 assignments.
    assert assigns.shape[0] == 2
    rows = {tuple(a) for a in assigns}
    assert len(rows) == 2


def test_murty_handles_forbidden():
    C = np.array([[1.0, np.inf], [np.inf, 1.0]])
    assigns, costs = murty(C, 5)
    assert assigns.shape[0] == 1  # only one feasible assignment
    assert abs(costs[0] - 2.0) < 1e-9


def test_gospa_perfect_match_zero():
    x = np.array([[1.0, 5.0], [2.0, 6.0]])
    d, _, decomp = gospa(x, x, p=2, c=10.0, alpha=2)
    assert d < 1e-9
    assert decomp.localisation < 1e-9


def test_gospa_cardinality_penalty():
    x = np.array([[0.0], [0.0]])
    y = np.zeros((2, 0))
    d, _, decomp = gospa(x, y, p=2, c=10.0, alpha=2)
    # One missed target: cost = (c^2 / alpha) = 50, d = sqrt(50).
    assert abs(d - np.sqrt(50.0)) < 1e-9
    assert abs(decomp.missed - 50.0) < 1e-9


if __name__ == "__main__":
    import sys
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
