"""Structural tests for the branch-and-bound hypothesis generation (cm/branchbound).

No MATLAB reference exists; these check that the port runs and produces
algorithmically-correct global hypotheses on small hand-checkable inputs:
measurement contention is respected, scores rank descending, and the
enumerated hypotheses match the optimal combinations.
"""

import numpy as np

from cm.branchbound import branch_and_bound_explore as bbe


def _hypos(hl, hcl):
    out, b = [], 0
    for c in hcl:
        out.append(tuple(hl[b:b + c].tolist()))
        b += c
    return out


def test_single_cluster_one_track():
    # 1 real track (id1) gated to 1 measurement (newborn track id2), 1 cluster.
    hl, hcl, pl = bbe(
        np.array([1]), np.array([1]), np.array([1]), np.array([1]),
        np.array([0.0]), 1, np.array([[1.0], [1.0]]),
        np.array([[2.0, 0.0], [0.0, 0.0]]), np.array([2]), 30,
        np.array([[10.0, 11.0]]), 3)
    assert _hypos(hl, hcl) == [(10,), (11, 2)]
    assert np.allclose(pl, [2.0, 0.0])


def test_two_clusters_contend_for_one_measurement():
    # tracks 1 and 2 both gated to the single measurement -> contention forces
    # aggressive switching across the two clusters.
    inf = np.inf
    hl, hcl, pl = bbe(
        np.array([1, 2]), np.array([1, 1]), np.array([1, 2]), np.array([1, 1]),
        np.array([0.0, 0.0]), 1, np.array([[1.0, 1.0], [1.0, 0.0]]),
        np.array([[3.0, 0.0, -inf], [2.0, -inf, 0.0], [0.0, 0.0, 0.0]]),
        np.array([3]), 30, np.array([[10.0, 11.0, np.nan], [20.0, np.nan, 21.0]]), 3)
    hyps = _hypos(hl, hcl)
    # ranked: t1->meas (3), t2->meas (2), both missed + newborn (0)
    assert hyps == [(10, 21), (11, 20), (11, 21, 3)]
    assert np.allclose(pl, [3.0, 2.0, 0.0])
    # scores strictly descending, all finite
    assert np.all(np.diff(pl) < 0)
    assert np.all(np.isfinite(pl))
    # no measurement claimed twice within any hypothesis (last col of lookup is a
    # missed-detection; here just check track ids are unique per hypothesis)
    for h in hyps:
        assert len(set(h)) == len(h)


def test_respects_n_hypo_total_max():
    inf = np.inf
    hl, hcl, pl = bbe(
        np.array([1, 2]), np.array([1, 1]), np.array([1, 2]), np.array([1, 1]),
        np.array([0.0, 0.0]), 1, np.array([[1.0, 1.0], [1.0, 0.0]]),
        np.array([[3.0, 0.0, -inf], [2.0, -inf, 0.0], [0.0, 0.0, 0.0]]),
        np.array([3]), 1, np.array([[10.0, 11.0, np.nan], [20.0, np.nan, 21.0]]), 3)
    assert len(hcl) == 1
    assert _hypos(hl, hcl) == [(10, 21)]


if __name__ == "__main__":
    import sys
    import traceback
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and getattr(v, "__module__", None) == "__main__"]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    sys.exit(1 if failed else 0)
