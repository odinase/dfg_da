"""Partition ("tCloud") and cluster/track index helpers.

Faithful ports of the ABC-hierarchy partition utilities used throughout the CM
filter. A "tCloud" is a vector of partition lengths; the A-level is the flat
concatenation of all partitions, the B-level is the index inside a partition,
and the C-level is the partition index.

Ports of: ``tCloud2BegInd.m``, ``tCloud2EndInd.m`` (at612/jmpdFunctions),
``a2bcFaster.m``, ``a2bc.m``, ``bc2a.m``, ``removeIndA.m``, ``pickIndC.m``,
``probLogs2Probabilities.m``, ``track2Cluster.m``, ``cluster2Tracks.m``,
``union_sorted.m``.

All indices are 1-based to match MATLAB semantics (``hypos``/``clusters`` store
1-based track/hypothesis numbers).
"""

import numpy as np


def tcloud_to_beg(tcloud):
    """1-based beginning index of each partition. Port of ``tCloud2BegInd.m``."""
    tcloud = np.asarray(tcloud, dtype=int).ravel()
    end = np.cumsum(tcloud)
    return end - tcloud + 1


def tcloud_to_end(tcloud):
    """1-based end index of each partition. Port of ``tCloud2EndInd.m``."""
    tcloud = np.asarray(tcloud, dtype=int).ravel()
    return np.cumsum(tcloud)


def a2bc(a, tcloud):
    """A-level -> (b, c). Port of ``a2bcFaster.m`` (1-based in/out).

    ``a`` may contain NaN, which propagates to ``c`` (and ``b``).
    """
    a = np.asarray(a, dtype=float).ravel()
    tcloud = np.asarray(tcloud, dtype=int).ravel()
    tcum = np.cumsum(tcloud)
    c = np.zeros(a.size)
    for ii in range(a.size):
        if np.isnan(a[ii]):
            c[ii] = np.nan
            continue
        c[ii] = np.sum(tcum < a[ii]) + 1
    tcum_ext = np.append(tcum, tcum[-1] if tcum.size else 0)
    tcloud_ext = np.append(tcloud, 0)
    b = np.full(a.size, np.nan)
    valid = ~np.isnan(c)
    cv = c[valid].astype(int)
    subtrahend = tcum_ext[cv - 1] - tcloud_ext[cv - 1]
    b[valid] = a[valid] - subtrahend
    return b, c


def bc2a(b, c, tcloud):
    """(b, c) -> A-level. Port of ``bc2a.m`` (1-based)."""
    b = np.asarray(b, dtype=int).ravel()
    c = np.asarray(c, dtype=int).ravel()
    beg = tcloud_to_beg(tcloud)
    return beg[c - 1] + b - 1


def remove_ind_a(a_remove, tcloud):
    """Port of ``removeIndA.m``. Returns (a_remove_sorted, tcloud_remove,
    a_remain, tcloud_remain), all 1-based."""
    a_remove = np.asarray(a_remove, dtype=int).ravel()
    tcloud = np.asarray(tcloud, dtype=int).ravel()
    a_remove = np.sort(a_remove)
    if a_remove.size == 0:
        a_remain = np.arange(1, int(np.sum(tcloud)) + 1)
        return a_remove, np.zeros(0, dtype=int), a_remain, tcloud.copy()

    b_remove, c_remove = a2bc(a_remove, tcloud)
    c_remove = c_remove.astype(int)
    c_unique = np.unique(c_remove)
    tcloud_remove = np.array([np.sum(c_remove == cu) for cu in c_unique], dtype=int)

    total = int(np.sum(tcloud))
    a_remain = np.arange(1, total + 1)
    a_remain = np.delete(a_remain, a_remove - 1)
    tcloud_remain = tcloud.copy()
    for ii, cu in enumerate(c_unique):
        tcloud_remain[cu - 1] -= tcloud_remove[ii]
    return a_remove, tcloud_remove, a_remain, tcloud_remain


def pick_ind_c(c, tcloud):
    """Port of ``pickIndC.m``: A-level indices of partitions ``c`` (1-based)."""
    c = np.asarray(c, dtype=int).ravel()
    tcloud = np.asarray(tcloud, dtype=int).ravel()
    if c.size == 0:
        return np.zeros(0, dtype=int)
    tcloud_remove = tcloud[c - 1]
    total = int(np.sum(tcloud_remove))
    b = np.zeros(total, dtype=int)
    c_enlarged = np.zeros(total, dtype=int)
    ptr = 0
    for ii in range(c.size):
        length = int(tcloud_remove[ii])
        b[ptr:ptr + length] = np.arange(1, length + 1)
        c_enlarged[ptr:ptr + length] = c[ii]
        ptr += length
    return bc2a(b, c_enlarged, tcloud)


def prob_logs_to_probabilities(prob_logs, clusters, clusters_card):
    """Port of ``probLogs2Probabilities.m``: per-cluster normalised probs.

    Returns a list (one entry per cluster) of probability arrays.
    """
    prob_logs = np.asarray(prob_logs, dtype=float).ravel()
    clusters = np.asarray(clusters, dtype=int).ravel()
    clusters_card = np.asarray(clusters_card, dtype=int).ravel()
    begs = tcloud_to_beg(clusters_card)
    ends = tcloud_to_end(clusters_card)
    out = []
    for ii in range(clusters_card.size):
        idx = clusters[begs[ii] - 1:ends[ii]] - 1  # 1-based hypo ids -> 0-based
        a = np.exp(prob_logs[idx])
        out.append(a / np.sum(a))
    return out


def track2cluster(tracks, hypos, hypos_card, clusters, clusters_card,
                  prob_log_hypos=None):
    """Port of ``track2Cluster.m``. Returns (cluster_numbers, hypos_col, probs).

    ``tracks`` are 1-based track numbers. ``hypos_col`` is a list (one per
    track) of B-level hypothesis indices within the cluster. When
    ``prob_log_hypos`` is given, also returns per-track summed probabilities.
    """
    tracks = np.atleast_1d(np.asarray(tracks, dtype=int)).ravel()
    hypos = np.asarray(hypos, dtype=int).ravel()
    hypos_card = np.asarray(hypos_card, dtype=int).ravel()
    clusters = np.asarray(clusters, dtype=int).ravel()
    clusters_card = np.asarray(clusters_card, dtype=int).ravel()

    nT = tracks.size
    cluster_numbers = np.zeros(nT)
    hypos_col = [None] * nT

    want_probs = prob_log_hypos is not None
    if want_probs:
        probabilities = np.zeros(nT)
        prob_cell = prob_logs_to_probabilities(prob_log_hypos, clusters, clusters_card)
        probabilities_h = np.zeros(hypos_card.size)
        begsC = tcloud_to_beg(clusters_card)
        endsC = tcloud_to_end(clusters_card)
        for iC in range(len(prob_cell)):
            hyp_ids = clusters[begsC[iC] - 1:endsC[iC]] - 1
            probabilities_h[hyp_ids] = prob_cell[iC]

    for ii in range(nT):
        hypos_ia = np.nonzero(hypos == tracks[ii])[0] + 1  # 1-based A-level
        _, hyp_i = a2bc(hypos_ia, hypos_card)  # B-level hypothesis indices
        hyp_i = hyp_i[~np.isnan(hyp_i)].astype(int)
        hypos_col[ii] = hyp_i

        if hyp_i.size > 0:
            clusters_i = np.zeros(hyp_i.size)
            for h in range(hyp_i.size):
                clusters_ia = np.nonzero(clusters == hyp_i[h])[0] + 1
                _, cc = a2bc(clusters_ia, clusters_card)
                clusters_i[h] = cc[0]
            if np.unique(clusters_i).size > 1:
                raise ValueError("Each track should only be in one cluster")
            cluster_numbers[ii] = np.unique(clusters_i)[0]
        else:
            cluster_numbers[ii] = np.nan

        if want_probs and hyp_i.size > 0:
            probabilities[ii] = np.sum(probabilities_h[hyp_i - 1])

    if want_probs:
        return cluster_numbers, hypos_col, probabilities
    return cluster_numbers, hypos_col


def cluster2tracks(c, hypos, hypos_card, clusters, clusters_card):
    """Port of ``cluster2Tracks.m``: 1-based track numbers in cluster ``c``."""
    hypos = np.asarray(hypos, dtype=int).ravel()
    hypos_card = np.asarray(hypos_card, dtype=int).ravel()
    clusters = np.asarray(clusters, dtype=int).ravel()
    clusters_card = np.asarray(clusters_card, dtype=int).ravel()
    begsC = tcloud_to_beg(clusters_card)
    endsC = tcloud_to_end(clusters_card)
    c_hypos = clusters[begsC[c - 1] - 1:endsC[c - 1]]
    a_in_cluster = pick_ind_c(c_hypos, hypos_card)
    if a_in_cluster.size == 0:
        return np.zeros(0, dtype=int)
    return np.unique(hypos[a_in_cluster - 1])


def union_sorted(a, b):
    """Port of ``union_sorted.m`` (Minka). Returns (c, a_match, b_match) where
    ``c`` is the sorted unique union and a_match/b_match map a,b into c (1-based).
    """
    a = np.asarray(a, dtype=int).ravel()
    b = np.asarray(b, dtype=int).ravel()
    c = np.union1d(a, b)  # sorted unique
    pos = {val: i + 1 for i, val in enumerate(c)}  # 1-based position in c
    a_match = np.array([pos[v] for v in a], dtype=int)
    b_match = np.array([pos[v] for v in b], dtype=int)
    return c, a_match, b_match
