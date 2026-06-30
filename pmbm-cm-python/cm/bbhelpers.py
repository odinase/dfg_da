"""Foundational helpers for Stage 2b (branch-and-bound hypothesis generation).

Deterministic, unit-testable ports of the MATLAB helpers that the cluster
carry-over and ``branchAndBoundExplore`` depend on:

* ``assign2D.m`` (David Crouse's rectangular Jonker-Volgenant solver) and its
  wrapper ``crouse2D.m`` (returns dual-variable upper bounds on rewards);
* ``reOrderC.m`` (at612/jmpdFunctions), ``insertElements.m`` /
  ``insertElementsInd.m``;
* ``sortHyposInCluster.m``, ``probLogs2ProbabilitiesForHypos.m``;
* the cluster/measurement invariant checks ``testClustersShareTracks.m``,
  ``testClustersShareMeasurements.m``, ``testRepeatedMeasurements.m``.

All public functions keep MATLAB's 1-based indexing convention at their
boundaries (``hypos``/``clusters`` store 1-based ids), matching ``cm/clouds.py``.
"""

import numpy as np

from .clouds import (a2bc, pick_ind_c, prob_logs_to_probabilities,
                     tcloud_to_beg, tcloud_to_end)


# ---------------------------------------------------------------------------
# Assignment solver (Crouse's assign2D + crouse2D wrapper)
# ---------------------------------------------------------------------------

def _shortest_path(cur_unass_col, u, v, C, col4row, row4col):
    """Shortest augmenting path (0-based). Port of the nested ``ShortestPath``
    in ``assign2D.m``. ``col4row``/``row4col`` use -1 for unassigned."""
    num_row, num_col = C.shape
    pred = np.full(num_row, -1, dtype=int)
    scanned_cols = np.zeros(num_col, dtype=bool)
    scanned_row = np.zeros(num_row, dtype=bool)
    row2scan = list(range(num_row))

    sink = -1
    delta = 0.0
    cur_col = cur_unass_col
    spc = np.full(num_row, np.inf)

    while sink == -1:
        scanned_cols[cur_col] = True
        min_val = np.inf
        closest_row_scan = -1
        for scan_idx, cur_row in enumerate(row2scan):
            reduced_cost = delta + C[cur_row, cur_col] - u[cur_col] - v[cur_row]
            if reduced_cost < spc[cur_row]:
                pred[cur_row] = cur_col
                spc[cur_row] = reduced_cost
            if spc[cur_row] < min_val:
                min_val = spc[cur_row]
                closest_row_scan = scan_idx
        if not np.isfinite(min_val):
            return -1, pred, u, v  # infeasible
        closest_row = row2scan[closest_row_scan]
        scanned_row[closest_row] = True
        del row2scan[closest_row_scan]
        delta = spc[closest_row]
        if col4row[closest_row] == -1:
            sink = closest_row
        else:
            cur_col = col4row[closest_row]

    u[cur_unass_col] += delta
    sel = scanned_cols.copy()
    sel[cur_unass_col] = False
    for c in np.nonzero(sel)[0]:
        u[c] += delta - spc[row4col[c]]
    for r in np.nonzero(scanned_row)[0]:
        v[r] += -delta + spc[r]
    return sink, pred, u, v


def assign2d(C, maximize=False):
    """Port of ``assign2D.m`` (Crouse). Solve the rectangular 2-D assignment.

    Returns ``(col4row, row4col, gain, u, v)`` where ``col4row`` is 0-based
    (column assigned to each row, -1 = unassigned), ``u`` is the dual for the
    columns and ``v`` the dual for the rows, both in the original orientation.
    If infeasible, ``col4row``/``row4col`` are empty and ``gain`` is -1.
    """
    C = np.array(C, dtype=float)
    num_row, num_col = C.shape

    did_flip = False
    if num_col > num_row:
        C = C.T
        num_row, num_col = num_col, num_row
        did_flip = True

    if maximize:
        c_delta = C.max()
        if c_delta < 0:
            c_delta = 0.0
        C = -C + c_delta
    else:
        c_delta = C.min()
        if c_delta > 0:
            c_delta = 0.0
        C = C - c_delta

    col4row = np.full(num_row, -1, dtype=int)
    row4col = np.full(num_col, -1, dtype=int)
    u = np.zeros(num_col)
    v = np.zeros(num_row)

    for cur_unass_col in range(num_col):
        sink, pred, u, v = _shortest_path(cur_unass_col, u, v, C, col4row, row4col)
        if sink == -1:
            empty = np.zeros(0, dtype=int)
            return empty, empty, -1.0, u, v
        j = sink
        while True:
            i = pred[j]
            col4row[j] = i
            h = row4col[i]
            row4col[i] = j
            j = h
            if i == cur_unass_col:
                break

    gain = 0.0
    for cur_col in range(num_col):
        gain += C[row4col[cur_col], cur_col]
    if maximize:
        gain = -gain + c_delta * num_col
        u = -u
        v = -v
    else:
        gain = gain + c_delta * num_col

    if did_flip:
        col4row, row4col = row4col, col4row
        u, v = v, u
    return col4row, row4col, gain, u, v


def crouse2d(reward_mat):
    """Port of ``crouse2D.m``. Solve the max-reward assignment and return
    dual-based per-cell upper bounds.

    Returns ``(person_to_item, opt_reward, upper_bounds_of_scores)`` where
    ``person_to_item`` is a 1-based column index per row (length n_rows),
    ``opt_reward`` is the optimal total reward and ``upper_bounds_of_scores`` is
    an (n_rows, n_cols) matrix bounding the reward attainable if that
    (row, col) pairing is enforced.
    """
    reward_mat = np.asarray(reward_mat, dtype=float)
    n_customers, n_items = reward_mat.shape
    if n_customers == 0 or n_items == 0:
        # Empty assignment problem (e.g. a branch-and-bound switch to an empty
        # parent hypothesis): no rows to assign, zero reward, empty bounds.
        return (np.zeros(0, dtype=int), 0.0, np.zeros((n_customers, n_items)))
    cost_mat = -reward_mat
    min_cost = cost_mat.min()
    p_mat = cost_mat - min_cost

    col4row, _, opt_cost_cr, u, v = assign2d(p_mat)
    person_to_item = col4row + 1  # 1-based

    bounds_mat = (opt_cost_cr + p_mat
                  - v.reshape(-1, 1) - u.reshape(1, -1))
    bounds_mat2 = bounds_mat + min_cost * n_customers
    opt_cost_cr2 = opt_cost_cr + min_cost * n_customers
    opt_reward = -opt_cost_cr2
    upper_bounds_of_scores = -bounds_mat2
    return person_to_item, opt_reward, upper_bounds_of_scores


# ---------------------------------------------------------------------------
# Partition re-ordering / insertion
# ---------------------------------------------------------------------------

def reorder_c(x_concat, t_concat, fwd_mapping):
    """Port of ``reOrderC.m``: re-order the partitions of a level-A cloud.

    ``t_concat`` are partition lengths; ``fwd_mapping`` is a 1-based permutation
    of partition indices (new partition ``k`` is taken from old partition
    ``fwd_mapping[k]``). ``x_concat`` is a (dim, n) array whose columns are the
    A-level elements (a 1-D array is treated as a single row). Returns
    ``(x_merged, t_merged)``.
    """
    t_concat = np.asarray(t_concat, dtype=int).ravel()
    fwd = np.asarray(fwd_mapping, dtype=int).ravel()
    x = np.asarray(x_concat)
    was_1d = x.ndim == 1
    if was_1d:
        x = x.reshape(1, -1)

    beg_c = tcloud_to_beg(t_concat)
    end_c = tcloud_to_end(t_concat)
    t_merged = t_concat[fwd - 1]
    beg_m = tcloud_to_beg(t_merged)
    end_m = tcloud_to_end(t_merged)

    x_merged = np.zeros_like(x)
    for k in range(fwd.size):
        src = slice(beg_c[fwd[k] - 1] - 1, end_c[fwd[k] - 1])
        dst = slice(beg_m[k] - 1, end_m[k])
        x_merged[:, dst] = x[:, src]

    if was_1d:
        x_merged = x_merged.ravel()
    return x_merged, t_merged


def insert_elements_ind(particles, tcloud):
    """Port of ``insertElementsInd.m``. ``particles`` are 1-based partition
    indices, each grown by one element. Returns ``(old_ind, insert_ind,
    tcloud_new)`` with 1-based positions into the new A-level cloud."""
    particles = np.atleast_1d(np.asarray(particles, dtype=int)).ravel()
    tcloud = np.asarray(tcloud, dtype=int).ravel()
    if particles.size and particles.max() > tcloud.size:
        raise ValueError("partition index exceeds number of partitions")
    if np.any(particles < 0):
        raise ValueError("negative partition index")

    tcloud_new = tcloud.copy()
    np.add.at(tcloud_new, particles - 1, 1)
    end_ind = np.cumsum(tcloud_new)
    insert_ind = end_ind[particles - 1]
    if end_ind.size:
        old_ind = np.arange(1, end_ind[-1] + 1)
        old_ind = np.delete(old_ind, insert_ind - 1)
    else:
        old_ind = np.zeros(0, dtype=int)
    return old_ind, insert_ind, tcloud_new


def insert_elements(particles, newborn, x_cloud, tcloud, x_dim):
    """Port of ``insertElements.m``: insert newborn A-level elements into the
    partitions named by ``particles``. Returns ``(x_cloud_new, tcloud_new)``."""
    particles = np.atleast_1d(np.asarray(particles, dtype=int)).ravel()
    newborn = np.asarray(newborn, dtype=float).reshape(x_dim, -1)
    if particles.size != newborn.shape[1]:
        raise ValueError("particles and newborn count mismatch")

    _, insert_ind, tcloud_new = insert_elements_ind(particles, tcloud)
    total = int(np.sum(tcloud_new))
    insert_marks = np.zeros(total, dtype=bool)
    insert_marks[insert_ind - 1] = True

    x_cloud_new = np.zeros((x_dim, total))
    x_cloud_new[:, insert_marks] = newborn
    x_cloud = np.asarray(x_cloud, dtype=float)
    if x_cloud.size:
        x_cloud_new[:, ~insert_marks] = x_cloud.reshape(x_dim, -1)

    if x_dim == 1:
        x_cloud_new = x_cloud_new.ravel()
    return x_cloud_new, tcloud_new


# ---------------------------------------------------------------------------
# Hypothesis sorting / probabilities
# ---------------------------------------------------------------------------

def sort_hypos_in_cluster(hypos, hypos_card, clusters, clusters_card, prob_logs):
    """Port of ``sortHyposInCluster.m``: sort hypotheses by descending log-prob
    inside each cluster and renumber.

    Returns ``(hypos_new, hypos_card_new, clusters_new, clusters_card_new,
    probabilities, prob_logs_new)``.
    """
    hypos = np.asarray(hypos, dtype=int).ravel()
    hypos_card = np.asarray(hypos_card, dtype=int).ravel()
    clusters = np.asarray(clusters, dtype=int).ravel()
    clusters_card = np.asarray(clusters_card, dtype=int).ravel()
    prob_logs = np.asarray(prob_logs, dtype=float).ravel()

    c_begs = tcloud_to_beg(clusters_card)
    c_ends = tcloud_to_end(clusters_card)
    clusters_new_temp = np.zeros(clusters.size, dtype=int)
    for iC in range(clusters_card.size):
        h_in_c = clusters[c_begs[iC] - 1:c_ends[iC]]
        p_in_c = prob_logs[h_in_c - 1]
        ix = np.argsort(-p_in_c, kind="stable")  # descend, stable like MATLAB
        clusters_new_temp[c_begs[iC] - 1:c_ends[iC]] = h_in_c[ix]

    hypos_new, hypos_card_new = reorder_c(hypos, hypos_card, clusters_new_temp)
    clusters_new = np.arange(1, clusters.size + 1, dtype=int)
    clusters_card_new = clusters_card.copy()
    prob_logs_new = prob_logs[clusters_new_temp - 1]

    prob_cell = prob_logs_to_probabilities(prob_logs_new, clusters_new,
                                           clusters_card_new)
    probabilities = np.zeros(clusters.size)
    for ii in range(clusters_card.size):
        probabilities[c_begs[ii] - 1:c_ends[ii]] = prob_cell[ii]

    return (hypos_new, hypos_card_new, clusters_new, clusters_card_new,
            probabilities, prob_logs_new)


def prob_logs_to_probabilities_for_hypos(hypos_selected, prob_logs, clusters,
                                         clusters_card, nH):
    """Port of ``probLogs2ProbabilitiesForHypos.m``: per-hypothesis normalised
    probability for a selected set of (1-based) hypothesis ids."""
    hypos_selected = np.atleast_1d(np.asarray(hypos_selected, dtype=int)).ravel()
    clusters = np.asarray(clusters, dtype=int).ravel()
    clusters_card = np.asarray(clusters_card, dtype=int).ravel()

    prob_cell = prob_logs_to_probabilities(prob_logs, clusters, clusters_card)
    a = np.full(nH, np.nan)
    a[clusters - 1] = np.arange(1, nH + 1)
    a_sel = a[hypos_selected - 1]
    b, c = a2bc(a_sel, clusters_card)

    probs = np.zeros(hypos_selected.size)
    for ii in range(probs.size):
        probs[ii] = prob_cell[int(c[ii]) - 1][int(b[ii]) - 1]
    return probs, a_sel, b, c


# ---------------------------------------------------------------------------
# Invariant checks (assertions used throughout the carry-over)
# ---------------------------------------------------------------------------

def test_clusters_share_tracks(clusters, clusters_card, hypos, hypos_card):
    """Port of ``testClustersShareTracks.m``. Returns ``(share_tracks,
    share_clusters)``; non-empty ``share_tracks`` means an invariant violation."""
    clusters = np.asarray(clusters, dtype=int).ravel()
    clusters_card = np.asarray(clusters_card, dtype=int).ravel()
    hypos = np.asarray(hypos, dtype=int).ravel()
    hypos_card = np.asarray(hypos_card, dtype=int).ravel()

    ch_begs = tcloud_to_beg(clusters_card)
    ch_ends = tcloud_to_end(clusters_card)
    share_tracks = []
    share_clusters = []
    n = clusters_card.size
    for iC in range(n):
        hyp_i = clusters[ch_begs[iC] - 1:ch_ends[iC]]
        tracks_i = hypos[pick_ind_c(hyp_i, hypos_card) - 1]
        for jC in range(iC + 1, n):
            hyp_j = clusters[ch_begs[jC] - 1:ch_ends[jC]]
            tracks_j = hypos[pick_ind_c(hyp_j, hypos_card) - 1]
            inter = np.intersect1d(tracks_i, tracks_j)
            if inter.size:
                share_tracks.extend(inter.tolist())
                share_clusters.append((iC + 1, jC + 1))
    return np.array(share_tracks, dtype=int), share_clusters


def test_clusters_share_measurements(clusters, clusters_card, hypos, hypos_card,
                                     mea_hist_col):
    """Port of ``testClustersShareMeasurements.m`` (last-measurement variant)."""
    clusters = np.asarray(clusters, dtype=int).ravel()
    clusters_card = np.asarray(clusters_card, dtype=int).ravel()
    hypos = np.asarray(hypos, dtype=int).ravel()
    hypos_card = np.asarray(hypos_card, dtype=int).ravel()
    mea_hist_col = np.asarray(mea_hist_col, dtype=float)

    ch_begs = tcloud_to_beg(clusters_card)
    ch_ends = tcloud_to_end(clusters_card)
    share_meas = []
    share_clusters = []
    n = clusters_card.size
    for iC in range(n):
        hyp_i = clusters[ch_begs[iC] - 1:ch_ends[iC]]
        tracks_i = hypos[pick_ind_c(hyp_i, hypos_card) - 1]
        mea_last_i = mea_hist_col[-1, tracks_i - 1].astype(float)
        mea_last_i[mea_last_i == 0] = np.nan
        for jC in range(iC + 1, n):
            hyp_j = clusters[ch_begs[jC] - 1:ch_ends[jC]]
            tracks_j = hypos[pick_ind_c(hyp_j, hypos_card) - 1]
            mea_last_j = mea_hist_col[-1, tracks_j - 1].astype(float)
            mea_last_j[mea_last_j == 0] = np.nan
            # intersect ignoring NaN (MATLAB intersect drops NaN)
            inter = np.intersect1d(mea_last_i[~np.isnan(mea_last_i)],
                                   mea_last_j[~np.isnan(mea_last_j)])
            if inter.size:
                share_meas.extend(inter.tolist())
                share_clusters.append((iC + 1, jC + 1))
    return np.array(share_meas, dtype=float), share_clusters


def test_repeated_measurements(hypos, hypos_card, mea_hist_col):
    """Port of ``testRepeatedMeasurements.m``. Returns ``(bools_h, bools_t)``;
    ``bools_h[i]`` true if hypothesis ``i`` claims the same current measurement
    twice."""
    hypos = np.asarray(hypos, dtype=int).ravel()
    hypos_card = np.asarray(hypos_card, dtype=int).ravel()
    mea_hist_col = np.asarray(mea_hist_col, dtype=float)

    begs = tcloud_to_beg(hypos_card)
    ends = tcloud_to_end(hypos_card)
    bools_h = np.zeros(hypos_card.size, dtype=bool)
    bools_t = np.zeros(mea_hist_col.shape[1], dtype=bool)
    for ii in range(hypos_card.size):
        h = hypos[begs[ii] - 1:ends[ii]]
        if h.size == 0:
            continue
        z = mea_hist_col[-1, h - 1]
        uni_z = np.unique(z)
        uni_z = uni_z[uni_z != 0]
        counts = np.array([np.sum(z == val) for val in uni_z])
        if np.any(counts > 1):
            bools_h[ii] = True
            z_reps = uni_z[counts > 1]
            for zr in z_reps:
                b_inds = np.nonzero(z == zr)[0]
                bools_t[h[b_inds] - 1] = True
    return bools_h, bools_t
