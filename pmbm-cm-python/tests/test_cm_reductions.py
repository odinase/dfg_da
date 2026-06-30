"""Unit tests for self-contained reduction helpers (cm/reductions.py)."""

import numpy as np

from cm.columns import InCol
from cm.reductions import (check_same_track_in_different_clusters,
                           pruning_adjust_clusters2, repeated_tracks,
                           track_pruning_pmbm)


def test_repeated_tracks():
    # tracks 0 and 2 share a column; track 1 distinct
    mh = np.array([[5.0, 7.0, 5.0],
                   [3.0, 9.0, 3.0]])
    rep = repeated_tracks(mh)
    # master of col0 and col2 is 1 (first occurrence); col1 master is 2
    assert rep[0].tolist() == [1, 2, 1]
    assert rep[1].tolist() == [1, 1, 0]


def test_check_same_track_clean_and_violation():
    # two clusters, distinct tracks -> clean
    hypos = np.array([1, 2])
    hypos_card = np.array([1, 1])
    clusters = np.array([1, 2])
    clusters_card = np.array([1, 1])
    tb, _ = check_same_track_in_different_clusters(
        np.array([1, 2]), hypos, hypos_card, clusters, clusters_card)
    assert not tb.any()
    # track 1 appears in both clusters' hypotheses -> violation
    hypos_v = np.array([1, 1])
    tb2, prob = check_same_track_in_different_clusters(
        np.array([1]), hypos_v, hypos_card, clusters, clusters_card)
    assert bool(tb2[0]) is True


def test_track_pruning_renumbers():
    # 3 tracks in file, only tracks 1 and 3 are claimed by hypos
    incol = InCol()
    nrows = incol.last
    track_file = np.arange(nrows * 3, dtype=float).reshape(nrows, 3)
    shadow = np.zeros((nrows, 3, 2))
    mea = np.array([[1.0, 2.0, 3.0]])
    hypos = np.array([1, 3, 1])  # track 2 unsupported
    hn, tf, sh, mh = track_pruning_pmbm(hypos, track_file, shadow, mea, 5)
    # track 2 dropped; old 1->1, old 3->2
    assert hn.tolist() == [1, 2, 1]
    assert tf.shape[1] == 2
    assert mh[0].tolist() == [1.0, 3.0]


def test_pruning_adjust_clusters_identity():
    # nothing removed -> clusters unchanged
    clusters = np.array([1, 2, 3])
    clusters_card = np.array([2, 1])
    to_be_kept = np.array([1, 2, 3])
    to_be_removed = np.zeros(3, dtype=bool)
    co, cco = pruning_adjust_clusters2(clusters, clusters_card, to_be_kept,
                                       to_be_removed, np.array([10, 11, 20]),
                                       np.array([1, 1, 1]))
    assert co.tolist() == [1, 2, 3]
    assert cco.tolist() == [2, 1]


def test_pruning_adjust_clusters_remove_one():
    # remove hypothesis at A-level position 2 (hypo id 2) from cluster 1
    clusters = np.array([1, 2, 3])
    clusters_card = np.array([2, 1])
    to_be_removed = np.array([False, True, False])
    to_be_kept = np.array([1, 3])
    co, cco = pruning_adjust_clusters2(clusters, clusters_card, to_be_kept,
                                       to_be_removed, np.array([10, 20]),
                                       np.array([1, 1]))
    # hypo 2 removed -> renumber: hypo1->1, hypo3->2; cluster1 now has 1 hypo
    assert cco.tolist() == [1, 1]
    assert co.tolist() == [1, 2]


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
