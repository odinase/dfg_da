"""Post-hypothesis-generation reduction stages of the CM PMBM carry-over.

Ports of ``pruningPmbmBid.m`` (+ ``pruningAdjustClusters2.m``),
``trackPruningPmbm.m``, ``nScanMerge.m`` (+ ``repeatedTracks.m``,
``checkSameTrackInDifferentClusters.m``), ``majorityTracks.m`` and
``clusterSplittingMajTraj.m`` (script_pmbm91.m:1847-2034).

Equivalent, not bit-exact (no MATLAB reference). ``pruningPmbmBid`` here enforces
the per-cluster hypothesis cap ``n_hypo_max`` as an n-best cut (the MATLAB "bid"
loop's effect): hypotheses beyond rank ``n_hypo_max`` within each cluster are
marked for removal after the descending-log-prob sort. It then also removes
tracks with no hypothesis support and simplifies two-hypothesis clusters that
contain an empty hypothesis.
"""

import numpy as np

from .bbhelpers import sort_hypos_in_cluster
from .clouds import (a2bc, cluster2tracks, pick_ind_c, prob_logs_to_probabilities,
                     remove_ind_a, tcloud_to_beg, tcloud_to_end, track2cluster)
from .mathutils import cov_mat_to_vec, cov_vec_to_mat, gm_reduce
from .validation import track_prob_accumulate_pure


def _pick_ind_c2(c, tcloud):
    """``pickIndC`` returning both the A-level indices and the picked partition
    lengths ``tcloud[c]`` (1-based ``c``)."""
    c = np.atleast_1d(np.asarray(c, dtype=int)).ravel()
    tcloud = np.asarray(tcloud, dtype=int).ravel()
    a = pick_ind_c(c, tcloud)
    cards = tcloud[c - 1] if c.size else np.zeros(0, dtype=int)
    return a, cards


# ---------------------------------------------------------------------------
# pruningAdjustClusters2
# ---------------------------------------------------------------------------

def pruning_adjust_clusters2(clusters, clusters_card, to_be_kept, to_be_removed,
                             hypos_prune, hypos_card_prune):
    """Port of ``pruningAdjustClusters2.m``."""
    clusters = np.asarray(clusters, dtype=int).ravel()
    clusters_card = np.asarray(clusters_card, dtype=int).ravel()
    to_be_kept = np.asarray(to_be_kept, dtype=int).ravel()
    to_be_removed = np.asarray(to_be_removed, dtype=bool).ravel()
    nH = clusters.size

    where_in_clusters = np.zeros(int(clusters.max()) if clusters.size else 0, dtype=int)
    where_in_clusters[clusters - 1] = np.arange(1, nH + 1)

    hypo_rem = clusters[to_be_removed]
    increments = np.ones(nH, dtype=int)
    if hypo_rem.size:
        increments[hypo_rem - 1] = 0
    new_ind_hypo = np.cumsum(increments)
    if hypo_rem.size:
        new_ind_hypo[hypo_rem - 1] = 0

    clusters_shifted = np.zeros(nH, dtype=int)
    clusters_shifted[where_in_clusters - 1] = new_ind_hypo
    clusters_prune = clusters_shifted[clusters_shifted > 0]

    _, clusters_card_prune, _, _ = remove_ind_a(to_be_kept, clusters_card)

    supported = np.nonzero(clusters_card_prune > 0)[0] + 1
    a_ind, clusters_card_out = _pick_ind_c2(supported, clusters_card_prune)
    clusters_out = clusters_prune[a_ind - 1] if a_ind.size else np.zeros(0, dtype=int)
    return clusters_out, clusters_card_out


# ---------------------------------------------------------------------------
# pruningPmbmBid
# ---------------------------------------------------------------------------

def pruning_pmbm_bid(hypos, hypos_card, clusters, clusters_card, prob_log_hypos,
                     track_file, track_file_shadow, mea_hist_col, incol,
                     n_hypo_total_max, n_hypo_max, k):
    """Port of ``pruningPmbmBid.m``.

    The per-cluster cap ``n_hypo_max`` is enforced as an n-best cut: after the
    hypotheses are sorted descending by log-prob within each cluster, the tail
    beyond rank ``n_hypo_max`` is marked for removal. ``n_hypo_total_max`` is
    unused (kept for signature stability)."""
    (hypos, hypos_card, clusters, clusters_card, _probs, prob_log_hypos) = \
        sort_hypos_in_cluster(hypos, hypos_card, clusters, clusters_card, prob_log_hypos)
    hypos = np.asarray(hypos, dtype=int).ravel()
    hypos_card = np.asarray(hypos_card, dtype=int).ravel()
    clusters = np.asarray(clusters, dtype=int).ravel()
    clusters_card = np.asarray(clusters_card, dtype=int).ravel()
    prob_log_hypos = np.asarray(prob_log_hypos, dtype=float).ravel()
    track_file = np.array(track_file, dtype=float)
    track_file_shadow = np.array(track_file_shadow, dtype=float)
    mea_hist_col = np.array(mea_hist_col, dtype=float)

    n_clusters_hyp = clusters.size
    # Per-cluster n-best cut: hypotheses are sorted descending by log-prob within
    # each cluster, so keep the first ``n_hypo_max`` and drop the tail.
    to_be_removed = np.zeros(n_clusters_hyp, dtype=bool)
    c_begs = tcloud_to_beg(clusters_card)
    c_ends = tcloud_to_end(clusters_card)
    for c in range(clusters_card.size):
        if clusters_card[c] > n_hypo_max:
            to_be_removed[c_begs[c] - 1 + n_hypo_max:c_ends[c]] = True
    to_be_kept = np.nonzero(~to_be_removed)[0] + 1

    hypos_ind_prune, hypos_card_prune = _pick_ind_c2(
        np.sort(clusters[to_be_kept - 1]), hypos_card)
    hypos_prune = hypos[hypos_ind_prune - 1] if hypos_ind_prune.size else np.zeros(0, dtype=int)
    prob_log_prune = prob_log_hypos[~to_be_removed]

    clusters_out, clusters_card_out = pruning_adjust_clusters2(
        clusters, clusters_card, to_be_kept, to_be_removed, hypos_prune, hypos_card_prune)

    # remove unclaimed tracks
    h_begs = tcloud_to_beg(hypos_card_prune)
    h_ends = tcloud_to_end(hypos_card_prune)
    claimed = np.zeros(track_file.shape[1], dtype=bool)
    for iH in range(hypos_card_prune.size):
        h = hypos_prune[h_begs[iH] - 1:h_ends[iH]]
        claimed[h - 1] = True
    tracks_keep = np.nonzero(claimed)[0] + 1
    new_for_old = np.zeros(track_file.shape[1], dtype=int)
    new_for_old[tracks_keep - 1] = np.arange(1, tracks_keep.size + 1)
    hypos_new = new_for_old[hypos_prune - 1] if hypos_prune.size else np.zeros(0, dtype=int)
    hypos_card_new = hypos_card_prune
    track_file = track_file[:, tracks_keep - 1]
    track_file_shadow = track_file_shadow[:, tracks_keep - 1, :]
    mea_hist_col = mea_hist_col[:, tracks_keep - 1]

    # simplify two-hypothesis clusters that contain an empty hypothesis
    existences = track_file[incol.exi, :].copy()
    two_some = np.nonzero(clusters_card_out == 2)[0] + 1
    empty_hypos = np.nonzero(hypos_card_new == 0)[0] + 1
    if empty_hypos.size:
        _, clusters_with_empty = a2bc(empty_hypos, clusters_card_out)
        clusters_with_empty = clusters_with_empty[~np.isnan(clusters_with_empty)].astype(int)
    else:
        clusters_with_empty = np.zeros(0, dtype=int)
    cand = np.intersect1d(two_some, clusters_with_empty)

    if cand.size:
        to_be_removed2 = np.zeros(int(np.sum(clusters_card_out)), dtype=bool)
        begsH = tcloud_to_beg(hypos_card_new)
        endsH = tcloud_to_end(hypos_card_new)
        begsC = tcloud_to_beg(clusters_card_out)
        endsC = tcloud_to_end(clusters_card_out)
        for c in cand:
            hC = clusters_out[begsC[c - 1] - 1:endsC[c - 1]]
            zero_card_h = int(np.argmin(hypos_card_new[hC - 1])) + 1  # 1-based in hC
            pL = prob_log_prune[hC - 1]
            probs = np.exp(pL - pL.max())
            probs = probs / probs.sum()
            other_h = 1 if zero_card_h == 2 else 2
            tracks_other = hypos_new[begsH[hC[other_h - 1] - 1] - 1:endsH[hC[other_h - 1] - 1]]
            existences[tracks_other - 1] = existences[tracks_other - 1] * probs[other_h - 1]
            to_be_removed2[hC[zero_card_h - 1] - 1] = True

        to_be_kept2 = np.nonzero(~to_be_removed2)[0] + 1
        hp_ind, hp_card = _pick_ind_c2(np.sort(clusters_out[to_be_kept2 - 1]), hypos_card_new)
        hypos_prune2 = hypos_new[hp_ind - 1] if hp_ind.size else np.zeros(0, dtype=int)
        prob_log_prune = prob_log_prune[~to_be_removed2]
        clusters_out, clusters_card_out = pruning_adjust_clusters2(
            clusters_out, clusters_card_out, to_be_kept2, to_be_removed2,
            hypos_prune2, hp_card)
        hypos_new = hypos_prune2
        hypos_card_new = hp_card
        track_file[incol.exi, :] = existences

    return (hypos_new, hypos_card_new, clusters_out, clusters_card_out,
            prob_log_prune, track_file, track_file_shadow, mea_hist_col, existences)


# ---------------------------------------------------------------------------
# trackPruningPmbm
# ---------------------------------------------------------------------------

def track_pruning_pmbm(hypos, track_file, track_file_shadow, mea_hist_col, k):
    """Port of ``trackPruningPmbm.m``: drop tracks with no hypothesis support
    and renumber. Returns (hypos_new, track_file, track_file_shadow,
    mea_hist_col)."""
    hypos = np.asarray(hypos, dtype=int).ravel()
    track_file = np.array(track_file, dtype=float)
    track_file_shadow = np.array(track_file_shadow, dtype=float)
    mea_hist_col = np.array(mea_hist_col, dtype=float)
    nT = track_file.shape[1]

    supported = np.intersect1d(np.arange(1, nT + 1), hypos)
    nTNew = supported.size
    ntn_in_old = np.full(nT, np.nan)
    ntn_in_old[supported - 1] = np.arange(1, nTNew + 1)
    hypos_new = ntn_in_old[hypos - 1].astype(int) if hypos.size else np.zeros(0, dtype=int)

    mea_hist_col = mea_hist_col[:, supported - 1]
    track_file = track_file[:, supported - 1]
    track_file_shadow = track_file_shadow[:, supported - 1, :]
    return hypos_new, track_file, track_file_shadow, mea_hist_col


# ---------------------------------------------------------------------------
# repeatedTracks / checkSameTrackInDifferentClusters / majorityTracks
# ---------------------------------------------------------------------------

def repeated_tracks(mea_hist_col):
    """Port of ``repeatedTracks.m``: single-linkage on identical measurement
    histories. Returns a (2, nT) array (master index 1-based; master flag)."""
    mh = np.array(mea_hist_col, dtype=float)
    mh[np.isnan(mh)] = -1
    nT = mh.shape[1]
    cols = mh.T
    repeated = np.zeros((2, nT), dtype=int)
    for t in range(nT):
        ia = np.all(cols == cols[t], axis=1)
        q = int(np.nonzero(ia)[0][0]) + 1
        repeated[0, ia] = q
        repeated[1, q - 1] = 1
    return repeated


def check_same_track_in_different_clusters(tracks, hypos, hypos_card, clusters,
                                           clusters_card):
    """Port of ``checkSameTrackInDifferentClusters.m``."""
    tracks = np.atleast_1d(np.asarray(tracks, dtype=int)).ravel()
    hypos = np.asarray(hypos, dtype=int).ravel()
    hypos_card = np.asarray(hypos_card, dtype=int).ravel()
    clusters = np.asarray(clusters, dtype=int).ravel()
    clusters_card = np.asarray(clusters_card, dtype=int).ravel()

    test_bool = np.zeros(tracks.size, dtype=bool)
    problematic = []
    for ii in range(tracks.size):
        hypos_ia = np.nonzero(hypos == tracks[ii])[0] + 1
        _, hyp_i = a2bc(hypos_ia, hypos_card)
        hyp_i = hyp_i[~np.isnan(hyp_i)].astype(int)
        if hyp_i.size:
            clusters_i = np.zeros(hyp_i.size, dtype=int)
            for h in range(hyp_i.size):
                clusters_ia = np.nonzero(clusters == hyp_i[h])[0] + 1
                _, cc = a2bc(clusters_ia, clusters_card)
                clusters_i[h] = int(cc[0])
            if np.unique(clusters_i).size > 1:
                test_bool[ii] = True
                problematic.append(clusters_i)
    return test_bool, problematic


def majority_tracks(hypos, hypos_card, clusters, clusters_card, prob_log_hypos,
                    sig, track_file, incol):
    """Port of ``majorityTracks.m``."""
    track_sum, _, _ = track_prob_accumulate_pure(
        track_file, incol, hypos, hypos_card, clusters, clusters_card, prob_log_hypos)
    return track_sum > 1 - sig, track_sum


# ---------------------------------------------------------------------------
# nScanMerge
# ---------------------------------------------------------------------------

def n_scan_merge(hypos, hypos_card, clusters, clusters_card, prob_log_hypos,
                 mea_hist_col, incol, track_file, track_file_shadow):
    """Port of ``nScanMerge.m`` (main path).

    The cross-cluster duplicate-recovery branch (rarely taken) is not ported;
    it raises :class:`NotImplementedError` if triggered.
    """
    hypos = np.asarray(hypos, dtype=int).ravel()
    hypos_card = np.asarray(hypos_card, dtype=int).ravel()
    clusters = np.asarray(clusters, dtype=int).ravel()
    clusters_card = np.asarray(clusters_card, dtype=int).ravel()
    prob_log_hypos = np.asarray(prob_log_hypos, dtype=float).ravel()
    mea_hist_col = np.array(mea_hist_col, dtype=float)
    track_file = np.array(track_file, dtype=float)
    track_file_shadow = np.array(track_file_shadow, dtype=float)

    max_lag = mea_hist_col.shape[0]
    repeated = repeated_tracks(mea_hist_col)
    new_track_numbers = np.cumsum(repeated[1, :])
    slave = repeated[1, :] == 0
    new_track_numbers[slave] = new_track_numbers[repeated[0, slave] - 1]
    n_new = int(new_track_numbers[-1]) if new_track_numbers.size else 0

    _, ttp, _ = track_prob_accumulate_pure(
        track_file, incol, hypos, hypos_card, clusters, clusters_card, prob_log_hypos)

    rows = track_file.shape[0]
    s_rows = track_file_shadow.shape[0]
    track_file_new = np.zeros((rows, n_new))
    mea_hist_col_new = np.zeros((max_lag, n_new))
    track_file_shadow_new = np.zeros((s_rows, n_new, max_lag))

    for ii in range(1, n_new + 1):
        first_i = int(np.nonzero(new_track_numbers == ii)[0][0]) + 1
        every_i = np.nonzero(repeated[0, :] == first_i)[0] + 1

        states = track_file[incol.tarX, every_i - 1]
        covs = cov_vec_to_mat(track_file[incol.tarP, every_i - 1])
        weights = ttp[every_i - 1].astype(float)
        weights = weights / weights.sum() if weights.sum() != 0 else np.ones_like(weights) / weights.size
        eta_ave, cov_ave = gm_reduce(states, covs, weights)
        track_file_new[incol.tarX, ii - 1] = eta_ave
        track_file_new[incol.tarP, ii - 1] = np.asarray(cov_mat_to_vec(cov_ave)).ravel()

        exi_pre = track_file[incol.exi, every_i - 1]
        ttp_w = ttp[every_i - 1].astype(float)
        ttp_w = ttp_w / ttp_w.sum() if ttp_w.sum() != 0 else np.ones_like(ttp_w) / ttp_w.size
        track_file_new[incol.exi, ii - 1] = exi_pre @ ttp_w

        strongest = int(np.argmax(ttp[every_i - 1]))
        src = every_i[strongest] - 1
        # incol.last is the row count (21); the "last" metadata row is index 20.
        last_row = incol.last - 1
        for f in (incol.label, last_row, incol.meaLast, incol.visi, incol.cost, incol.contrib):
            track_file_new[f, ii - 1] = track_file[f, src]
        mea_hist_col_new[:, ii - 1] = mea_hist_col[:, src]

        for lag in range(max_lag):
            s_states = track_file_shadow[incol.tarX, every_i - 1, lag]
            s_covs = cov_vec_to_mat(track_file_shadow[incol.tarP, every_i - 1, lag])
            s_eta, s_cov = gm_reduce(s_states, s_covs, weights)
            track_file_shadow_new[incol.tarX, ii - 1, lag] = s_eta
            track_file_shadow_new[incol.tarP, ii - 1, lag] = np.asarray(cov_mat_to_vec(s_cov)).ravel()
            s_exi = track_file_shadow[incol.exi, every_i - 1, lag]
            track_file_shadow_new[incol.exi, ii - 1, lag] = s_exi @ ttp_w
            for f in (incol.label, last_row, incol.meaLast, incol.visi, incol.cost, incol.contrib):
                track_file_shadow_new[f, ii - 1, lag] = track_file_shadow[f, src, lag]

    # merge hypotheses with identical (re-numbered) track sets, per cluster
    c_begs = tcloud_to_beg(clusters_card)
    c_ends = tcloud_to_end(clusters_card)
    h_begs = tcloud_to_beg(hypos_card)
    h_ends = tcloud_to_end(hypos_card)
    prob_cell = prob_logs_to_probabilities(prob_log_hypos, clusters, clusters_card)

    hypos_new, hypos_card_new, clusters_card_new, prob_logs_new = [], [], [], []
    for c in range(clusters_card.size):
        hypos_in_c = clusters[c_begs[c] - 1:c_ends[c]]
        heap = hypos_in_c.tolist()
        heap_probs = prob_cell[c].tolist()
        h_new_c, prob_logs_new_c = [], []
        while heap:
            master = heap[0]
            mt = new_track_numbers[hypos[h_begs[master - 1] - 1:h_ends[master - 1]] - 1]
            comb = [0]
            keep = [True] * len(heap)
            keep[0] = False
            for jj in range(1, len(heap)):
                cand = heap[jj]
                ct = new_track_numbers[hypos[h_begs[cand - 1] - 1:h_ends[cand - 1]] - 1]
                if (mt.size == ct.size and np.all(mt == ct)) or (mt.size == 0 and ct.size == 0):
                    comb.append(jj)
                    keep[jj] = False
            new_prob = sum(heap_probs[ci] for ci in comb)
            prob_logs_new_c.append(np.log(new_prob))
            h_new_c.append(master)
            heap = [heap[i] for i in range(len(heap)) if keep[i]]
            heap_probs = [heap_probs[i] for i in range(len(heap_probs)) if keep[i]]

        clusters_card_new.append(len(h_new_c))
        h_new_c = np.array(h_new_c, dtype=int)
        h_new_content = pick_ind_c(h_new_c, hypos_card)
        hypos_new.extend(new_track_numbers[hypos[h_new_content - 1] - 1].tolist())
        hypos_card_new.extend(hypos_card[h_new_c - 1].tolist())
        prob_logs_new.extend(prob_logs_new_c)

    hypos_new = np.array(hypos_new, dtype=int)
    hypos_card_new = np.array(hypos_card_new, dtype=int)
    clusters_card_new = np.array(clusters_card_new, dtype=int)
    clusters_new = np.arange(1, int(np.sum(clusters_card_new)) + 1)
    prob_logs_new = np.array(prob_logs_new, dtype=float)

    test_bool, _ = check_same_track_in_different_clusters(
        np.arange(1, n_new + 1), hypos_new, hypos_card_new, clusters_new, clusters_card_new)
    if np.any(test_bool):
        raise NotImplementedError(
            "nScanMerge cross-cluster duplicate-recovery branch not ported "
            "(same track ended up in multiple clusters after merge)")

    return (hypos_new, hypos_card_new, clusters_new, clusters_card_new,
            prob_logs_new, mea_hist_col_new, track_file_new, track_file_shadow_new)


# ---------------------------------------------------------------------------
# clusterSplittingMajTraj
# ---------------------------------------------------------------------------

def cluster_splitting_maj_traj(hypos, hypos_card, clusters, clusters_card,
                               prob_log_hypos, track_file, sig, mea_hist_col,
                               incol, splitting_threshold, maha_cs_thres,
                               non_split_lag, k):
    """Port of ``clusterSplittingMajTraj.m``."""
    hypos = np.asarray(hypos, dtype=int).ravel()
    hypos_card = np.asarray(hypos_card, dtype=int).ravel()
    clusters = np.asarray(clusters, dtype=int).ravel()
    clusters_card = np.asarray(clusters_card, dtype=int).ravel()
    prob_log_hypos = np.asarray(prob_log_hypos, dtype=float).ravel()
    track_file = np.array(track_file, dtype=float)
    mea_hist_col = np.array(mea_hist_col, dtype=float)

    n_tracks = mea_hist_col.shape[1]
    max_lag = mea_hist_col.shape[0]

    track_probs = np.zeros(n_tracks)
    for ii in range(n_tracks):
        _, _, pI = track2cluster(ii + 1, hypos, hypos_card, clusters, clusters_card,
                                 prob_log_hypos)
        track_probs[ii] = pI[0] * track_file[incol.exi, ii]

    last_consec = np.zeros(n_tracks, dtype=int)
    for ii in range(n_tracks):
        counter = 0
        for jj in range(max_lag):
            if mea_hist_col[max_lag - 1 - jj, ii] != 0:
                break
            counter += 1
        last_consec[ii] = counter

    splitting_tracks = np.nonzero((last_consec > 1) & (track_probs < splitting_threshold))[0] + 1
    if splitting_tracks.size:
        c_of_split, _ = track2cluster(splitting_tracks, hypos, hypos_card, clusters,
                                      clusters_card)
    else:
        c_of_split = np.zeros(0)

    prob_cell = prob_logs_to_probabilities(prob_log_hypos, clusters, clusters_card)
    c_begs = tcloud_to_beg(clusters_card)
    c_ends = tcloud_to_end(clusters_card)
    h_begs = tcloud_to_beg(hypos_card)
    h_ends = tcloud_to_end(hypos_card)
    nC = clusters_card.size

    selected, _ = majority_tracks(hypos, hypos_card, clusters, clusters_card,
                                  prob_log_hypos, sig, track_file, incol)
    maj_tracks = np.nonzero(selected)[0] + 1

    hypos_new, hypos_new_card, clusters_new, clusters_new_card, probs_as = [], [], [], [], []

    for c in range(1, nC + 1):
        splitting_in_c = splitting_tracks[c_of_split == c] if splitting_tracks.size \
            else np.zeros(0, dtype=int)
        tracks_in_c = cluster2tracks(c, hypos, hypos_card, clusters, clusters_card)
        active = np.setdiff1d(tracks_in_c, splitting_in_c)
        nTA = active.size
        _, ia_idx, _ = np.intersect1d(active, maj_tracks, return_indices=True) \
            if nTA else (np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int))
        ia = (ia_idx + 1).tolist()  # 1-based positions in active

        omega = np.zeros((nTA, nTA))
        for t in range(1, nTA + 1):
            xT = track_file[incol.tarX, active[t - 1] - 1]
            pT = cov_vec_to_mat(track_file[incol.tarP, active[t - 1] - 1])
            for jj in range(1, nTA + 1):
                if jj == t:
                    continue
                if (jj not in ia) and (t not in ia):
                    xJ = track_file[incol.tarX, active[jj - 1] - 1]
                    pJ = cov_vec_to_mat(track_file[incol.tarP, active[jj - 1] - 1])
                    d = xT - xJ
                    maha = d @ np.linalg.solve(pT + pJ, d)
                    if maha < maha_cs_thres ** 2:
                        omega[t - 1, jj - 1] = 1
                mea_now_t = mea_hist_col[-1, active[t - 1] - 1]
                mea_now_j = mea_hist_col[-1, active[jj - 1] - 1]
                if mea_now_t == mea_now_j and mea_now_t > 0:
                    omega[t - 1, jj - 1] = 1
                comp_t = mea_hist_col[max_lag - non_split_lag:max_lag, active[t - 1] - 1].copy()
                comp_j = mea_hist_col[max_lag - non_split_lag:max_lag, active[jj - 1] - 1].copy()
                comp_t[comp_t == 0] = np.nan
                comp_j[comp_j == 0] = np.nan
                if np.any(comp_t == comp_j):
                    omega[t - 1, jj - 1] = 1
        omega = (omega + omega.T) > 0

        assoc = np.zeros((2, nTA))
        assoc[1, :] = np.nan
        for p in range(1, nTA + 1):
            if np.isnan(assoc[1, p - 1]):
                assoc[1, p - 1] = p
                assoc[0, p - 1] = 1
            for i in list(range(1, p)) + list(range(p + 1, nTA + 1)):
                if omega[p - 1, i - 1] > 0:
                    if assoc[1, i - 1] > assoc[1, p - 1]:
                        slaves = np.nonzero(assoc[1, :] == assoc[1, i - 1])[0]
                        assoc[0, slaves] = 0
                        assoc[1, slaves] = assoc[1, p - 1]
                    elif assoc[1, i - 1] < assoc[1, p - 1]:
                        slaves = np.nonzero(assoc[1, :] == assoc[1, p - 1])[0]
                        assoc[0, slaves] = 0
                        assoc[1, slaves] = assoc[1, i - 1]
                    else:
                        assoc[1, i - 1] = assoc[1, p - 1]

        hypos_in_c = clusters[c_begs[c - 1] - 1:c_ends[c - 1]]
        masters = np.nonzero(assoc[0, :])[0] + 1
        nDC = masters.size
        probabilities_c = prob_cell[c - 1]

        for d in range(nDC):
            active_in_d = active[assoc[1, :] == masters[d]]
            heap_hypos, heap_card, heap_probs = [], [], []
            for iH in range(hypos_in_c.size):
                h = hypos[h_begs[hypos_in_c[iH] - 1] - 1:h_ends[hypos_in_c[iH] - 1]]
                active_in_h = np.intersect1d(active_in_d, h)
                ea_begs = tcloud_to_beg(np.array(heap_card, dtype=int))
                ea_ends = tcloud_to_end(np.array(heap_card, dtype=int))
                in_heap = False
                ea_numbers = []
                heap_flat = np.array(heap_hypos, dtype=int)
                for ea in range(len(heap_card)):
                    hEA = heap_flat[ea_begs[ea] - 1:ea_ends[ea]]
                    if (active_in_h.size and hEA.size == active_in_h.size
                            and np.all(hEA == active_in_h)) or \
                            (active_in_h.size == 0 and hEA.size == 0):
                        in_heap = True
                        ea_numbers.append(ea)
                if len(ea_numbers) > 1:
                    raise ValueError("more than one heap hypo matches in cluster splitting")
                if not in_heap:
                    heap_hypos.extend(active_in_h.tolist())
                    heap_card.append(active_in_h.size)
                    heap_probs.append(probabilities_c[iH])
                else:
                    heap_probs[ea_numbers[0]] += probabilities_c[iH]

            hypos_new.extend(heap_hypos)
            hypos_new_card.extend(heap_card)
            probs_as.extend(heap_probs)
            base = len(clusters_new)
            clusters_new.extend(range(base + 1, base + len(heap_card) + 1))
            clusters_new_card.append(len(heap_card))

    return (np.array(hypos_new, dtype=int), np.array(hypos_new_card, dtype=int),
            np.array(clusters_new, dtype=int), np.array(clusters_new_card, dtype=int),
            np.array(probs_as, dtype=float), splitting_tracks)
