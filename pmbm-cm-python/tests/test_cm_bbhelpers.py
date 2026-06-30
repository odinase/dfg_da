"""Unit tests for Stage 2b foundational helpers (cm/bbhelpers.py, cm/mathutils m2v)."""

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

from cm.bbhelpers import (assign2d, crouse2d, insert_elements,
                          insert_elements_ind, prob_logs_to_probabilities_for_hypos,
                          reorder_c, sort_hypos_in_cluster,
                          test_clusters_share_measurements,
                          test_clusters_share_tracks, test_repeated_measurements)
from cm.mathutils import m2v, v2m


def _brute_force_best_with_forced(reward, i, j):
    """Max total reward over assignments of every row to a distinct column,
    with row i forced to column j (small matrices only)."""
    n_rows, n_cols = reward.shape
    other_rows = [r for r in range(n_rows) if r != i]
    other_cols = [c for c in range(n_cols) if c != j]
    best = -np.inf
    for perm in itertools.permutations(other_cols, len(other_rows)):
        val = reward[i, j] + sum(reward[r, perm[k]] for k, r in enumerate(other_rows))
        best = max(best, val)
    return best


def test_m2v_v2m_roundtrip():
    siz = (4, 6)
    subs = np.array([[1, 4, 2, 3], [1, 6, 3, 5]])  # 1-based [row; col]
    lin = m2v(subs, siz)
    # sub2ind column-major, 1-based
    expected = (subs[1] - 1) * siz[0] + subs[0]
    assert np.array_equal(lin, expected)
    back = v2m(lin, siz[0])
    assert np.array_equal(back, subs)


def test_assign2d_crouse_example1():
    inf = np.inf
    C = np.array([[inf, 2, inf, inf, 3],
                  [7, inf, 23, inf, inf],
                  [17, 24, inf, inf, inf],
                  [inf, 6, 13, 20, inf]])
    col4row, row4col, gain, u, v = assign2d(C)
    assert abs(gain - 47.0) < 1e-9


def test_assign2d_matches_scipy_square():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(2, 7)
        C = rng.normal(size=(n, n)) * 10
        col4row, _, gain, _, _ = assign2d(C)
        r, c = linear_sum_assignment(C)
        assert abs(gain - C[r, c].sum()) < 1e-7
        # every row assigned, valid permutation
        assert sorted(col4row.tolist()) == list(range(n))


def test_assign2d_rectangular_more_cols():
    rng = np.random.default_rng(1)
    for _ in range(20):
        nr = rng.integers(2, 5)
        nc = nr + rng.integers(1, 4)
        C = rng.normal(size=(nr, nc)) * 10
        col4row, _, gain, _, _ = assign2d(C)
        r, c = linear_sum_assignment(C)  # min, every row matched (nr<=nc)
        assert abs(gain - C[r, c].sum()) < 1e-7
        assert len(set(col4row.tolist())) == nr  # distinct columns


def test_crouse2d_opt_reward_and_bounds():
    rng = np.random.default_rng(2)
    for _ in range(15):
        nr = rng.integers(2, 4)
        nc = nr + rng.integers(0, 3)
        reward = rng.normal(size=(nr, nc)) * 5
        p2i, opt_reward, ub = crouse2d(reward)
        # opt_reward equals true optimum
        r, c = linear_sum_assignment(-reward)
        assert abs(opt_reward - reward[r, c].sum()) < 1e-7
        # person_to_item realises opt_reward
        realised = sum(reward[row, p2i[row] - 1] for row in range(nr))
        assert abs(realised - opt_reward) < 1e-7
        # upper bound is a valid upper bound on forcing each cell
        for i in range(nr):
            for j in range(nc):
                actual = _brute_force_best_with_forced(reward, i, j)
                assert ub[i, j] >= actual - 1e-7
        # at the optimal cells the bound equals the optimum
        for row in range(nr):
            assert abs(ub[row, p2i[row] - 1] - opt_reward) < 1e-6


def test_crouse2d_empty_problem():
    # a branch-and-bound switch to an empty parent hypothesis yields a 0-row
    # reward matrix; crouse2d must return no assignment and zero reward.
    p2i, opt_reward, ub = crouse2d(np.zeros((0, 3)))
    assert p2i.size == 0
    assert opt_reward == 0.0
    assert ub.shape == (0, 3)


def test_reorder_c():
    # two hypotheses: hypo1 has tracks [10,11], hypo2 has [20]
    hypos = np.array([10, 11, 20])
    hypos_card = np.array([2, 1])
    # reverse order
    x, t = reorder_c(hypos, hypos_card, np.array([2, 1]))
    assert np.array_equal(x, [20, 10, 11])
    assert np.array_equal(t, [1, 2])


def test_insert_elements_ind_and_insert():
    # clustersCard = [2,1]; insert into cluster 1
    cWNew = np.array([1, 2, 3])      # cluster1 -> hypos {1,2}, cluster2 -> {3}
    cCardWNew = np.array([2, 1])
    x_new, t_new = insert_elements(1, 99, cWNew, cCardWNew, 1)
    # 99 appended at end of cluster 1
    assert np.array_equal(x_new, [1, 2, 99, 3])
    assert np.array_equal(t_new, [3, 1])
    old, ins, tnew = insert_elements_ind(2, cCardWNew)
    assert np.array_equal(tnew, [2, 2])
    assert ins.tolist() == [4]


def test_sort_hypos_in_cluster():
    # cluster1: hypos {1,2} with logs [-1, -0.5] -> 2 should come first
    hypos = np.array([10, 11, 20, 21])
    hypos_card = np.array([2, 2])      # hypo1 tracks[10,11], hypo2 tracks[20,21]
    clusters = np.array([1, 2])
    clusters_card = np.array([2])      # one cluster with both hypotheses
    prob_logs = np.array([-1.0, -0.5])
    (hn, hcn, cn, ccn, probs, pln) = sort_hypos_in_cluster(
        hypos, hypos_card, clusters, clusters_card, prob_logs)
    # hypo 2 (log -0.5) sorts first
    assert np.array_equal(pln, [-0.5, -1.0])
    assert np.array_equal(hn, [20, 21, 10, 11])
    assert np.array_equal(hcn, [2, 2])
    assert abs(probs.sum() - 1.0) < 1e-12
    assert probs[0] > probs[1]


def test_prob_logs_for_hypos():
    # 3 hypotheses across 2 clusters: c1={1,2}, c2={3}
    clusters = np.array([1, 2, 3])
    clusters_card = np.array([2, 1])
    prob_logs = np.array([np.log(0.6), np.log(0.4), np.log(1.0)])
    probs, a_sel, b, c = prob_logs_to_probabilities_for_hypos(
        np.array([1, 2, 3]), prob_logs, clusters, clusters_card, 3)
    assert np.allclose(probs, [0.6, 0.4, 1.0])


def test_invariants_clean_and_violation():
    # two singleton clusters, distinct tracks/measurements -> clean
    hypos = np.array([1, 2])
    hypos_card = np.array([1, 1])
    clusters = np.array([1, 2])
    clusters_card = np.array([1, 1])
    mea_hist_col = np.array([[5.0, 6.0]])  # last row: track1->mea5, track2->mea6
    st, sc = test_clusters_share_tracks(clusters, clusters_card, hypos, hypos_card)
    assert st.size == 0
    sm, _ = test_clusters_share_measurements(clusters, clusters_card, hypos,
                                             hypos_card, mea_hist_col)
    assert sm.size == 0
    # now make both tracks claim measurement 5 -> share
    mea_hist_col2 = np.array([[5.0, 5.0]])
    sm2, _ = test_clusters_share_measurements(clusters, clusters_card, hypos,
                                              hypos_card, mea_hist_col2)
    assert 5.0 in sm2.tolist()

    # repeated measurement within one hypothesis
    hypos_r = np.array([1, 2])
    hypos_card_r = np.array([2])       # one hypothesis with tracks 1 and 2
    mea_rep = np.array([[7.0, 7.0]])    # both claim mea 7
    bools_h, _ = test_repeated_measurements(hypos_r, hypos_card_r, mea_rep)
    assert bool(bools_h[0]) is True


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
