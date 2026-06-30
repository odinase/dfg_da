"""Tests for the supercluster size/cardinality balancing loop in
``cm.clustering._clustering_heavy`` (port of clusteringPreprocess.m lines 74-289)."""

import numpy as np

from cm.clustering import (MAX_CLUSTERS_PER_SUPERCLUSTER, clustering_preprocess)


def _chain_inputs(n_clusters):
    """Build a clustering_preprocess input where ``n_clusters`` single-hypothesis,
    single-track clusters are chained by shared measurements into ONE connected
    component (measurement j gates tracks j and j+1)."""
    nT = n_clusters
    m = n_clusters - 1
    hypos = np.arange(1, nT + 1)
    hypos_card = np.ones(nT, dtype=int)
    clusters = np.arange(1, nT + 1)
    clusters_card = np.ones(nT, dtype=int)

    gain = np.full((nT, m + nT), -np.inf)
    for j in range(m):                       # measurement j links track j and j+1
        gain[j, j] = 1.0 + 0.01 * j
        gain[j + 1, j] = 1.0 + 0.01 * j + 0.005
    for t in range(nT):                      # misdetection diagonal
        gain[t, m + t] = 0.0

    mea_hist_col = np.zeros((1, nT))
    return hypos, hypos_card, clusters, clusters_card, gain, mea_hist_col, m


def _group_sizes(assoc_local):
    reps = assoc_local[0].astype(int)
    _, counts = np.unique(reps, return_counts=True)
    return counts


def test_balancing_splits_oversized_supercluster():
    hypos, hypos_card, clusters, clusters_card, gain, mhc, m = _chain_inputs(7)
    # Large pre_cluster_threshold disables the initial betweenness edge-break, so
    # all 7 clusters start in one supercluster and the balancing loop must split it.
    assoc_local, _gain_post, masters = clustering_preprocess(
        hypos, hypos_card, clusters, clusters_card, gain, mhc, m,
        pre_cluster_threshold=1e9)

    sizes = _group_sizes(assoc_local)
    assert sizes.max() <= MAX_CLUSTERS_PER_SUPERCLUSTER  # every supercluster <= 5
    assert masters.size > 1                              # actually got split
    assert int(assoc_local[1].sum()) == masters.size     # one master flag per supercluster
    # assocLocal master flags are consistent with the representatives.
    for rep in np.unique(assoc_local[0].astype(int)):
        assert assoc_local[1, rep - 1] == 1


def test_small_chain_stays_single_supercluster():
    # 4 clusters (<= 5) need no splitting: one supercluster, all members merged.
    hypos, hypos_card, clusters, clusters_card, gain, mhc, m = _chain_inputs(4)
    assoc_local, _gain_post, masters = clustering_preprocess(
        hypos, hypos_card, clusters, clusters_card, gain, mhc, m,
        pre_cluster_threshold=1e9)
    sizes = _group_sizes(assoc_local)
    assert sizes.max() == 4
    assert masters.size == 1
