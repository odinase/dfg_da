"""Tests for the cluster-membership decoder (``cluster_membership.py``).

They encode the structural contract of the PMBM cloud storage:
    - each CSR layer is well-formed (``sum(card) == len(values)``),
    - hypothesis ids are in range,
    - the decoded clusters form a disjoint + covering partition of the tracks,
and cross-check the decoder against the repo's own ``cm.clouds.cluster2tracks``.

Run:  ./.venv/bin/python -m pytest test_cluster_membership.py
"""
import glob
import sys

import numpy as np
import pytest
import scipy.io as sio

import cluster_membership as cmb

REAL_FILES = sorted(glob.glob("data/pmbm_output_files/*.mat"))


# --------------------------------------------------------------------------- #
# 1. Golden: tiny synthetic clouds with a hand-checked answer
# --------------------------------------------------------------------------- #
def test_golden_two_clusters():
    # cluster 0 owns hypotheses {1, 2}; cluster 1 owns hypothesis {3}.
    # hypo1 -> [1, 2], hypo2 -> [1, 3], hypo3 -> [4]
    clusters, clusters_card = np.array([1, 2, 3]), np.array([2, 1])
    hypos, hypos_card = np.array([1, 2, 1, 3, 4]), np.array([2, 2, 1])
    assert cmb.cluster_tracks(clusters, clusters_card, hypos, hypos_card) == [[1, 2, 3], [4]]


def test_track_to_cluster_is_inverse():
    ct = [[1, 2, 3], [4]]
    assert cmb.track_to_cluster(ct) == {1: 0, 2: 0, 3: 0, 4: 1}


def test_from_card_rejects_malformed():
    with pytest.raises(ValueError):
        cmb.Jagged.from_card(np.array([1, 2, 3]), np.array([2, 2]))  # sum(card)=4 != 3


# --------------------------------------------------------------------------- #
# 2. Edge cases
# --------------------------------------------------------------------------- #
def test_empty_scan():
    empty = np.array([], dtype=int)
    assert cmb.cluster_tracks(empty, empty, empty, empty) == []


def test_singleton_cluster_and_hypothesis():
    # one cluster, one hypothesis, one track
    assert cmb.cluster_tracks(np.array([1]), np.array([1]),
                              np.array([7]), np.array([1])) == [[7]]


# --------------------------------------------------------------------------- #
# 3. Structural invariants / partition on real files
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not REAL_FILES, reason="no priorLikelihood*.mat data present")
@pytest.mark.parametrize("path", REAL_FILES[::100])  # a spread-out sample
def test_real_file_structure(path):
    cmb.verify_structure(cmb.load_cloud(path))  # raises AssertionError on violation


# --------------------------------------------------------------------------- #
# 4. Differential: decoder vs the repo's own cm.clouds.cluster2tracks
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not REAL_FILES, reason="no priorLikelihood*.mat data present")
def test_matches_cm_clouds_reference():
    sys.path.insert(0, "pmbm-cm-python")
    cm_clouds = pytest.importorskip("cm.clouds")
    d = sio.loadmat("data/pmbm_output_files/priorLikelihood5.mat", squeeze_me=True)
    a = lambda k: np.atleast_1d(d[k]).astype(int)
    cl, cc, hy, hc = a("clusters"), a("clustersCard"), a("hypos"), a("hyposCard")
    mine = cmb.cluster_tracks(cl, cc, hy, hc)
    for c in range(len(cc)):
        ref = sorted(cm_clouds.cluster2tracks(c + 1, hy, hc, cl, cc).tolist())
        assert mine[c] == ref


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
