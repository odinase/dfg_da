"""Carry-over: advance the CM state from the end of step k to the start of k+1.

Port of the back-half of the per-step loop in ``script_pmbm91.m`` (lines
1239-2055): per-cluster branch-and-bound hypothesis generation, newborn-cluster
creation, pruning, n-scan merging, cluster splitting and PHD recycling.

Two regimes:

* **No clusters yet** (the step-1 carry-over): ``branchAndBoundExplore`` is not
  called and the reductions are no-ops on the singleton-cluster state. This is
  the Stage 2a path, kept verbatim so the verified ``priorLikelihood2`` dump is
  unchanged.
* **Clusters exist** (k>=2 carry-over, Stage 2b): the full pipeline runs —
  per-supercluster ``branchAndBoundExplore`` -> newborn clustering ->
  ``pruningPmbmBid`` -> ``nScanMerge`` -> ``clusterSplittingMajTraj`` -> PHD
  recycling (``doRunnall=false``, ``doPostDMRecycle=false``).
"""

import numpy as np

from .bbhelpers import insert_elements
from .branchbound import branch_and_bound_explore
from .clouds import track2cluster
from .reductions import (cluster_splitting_maj_traj, n_scan_merge,
                         pruning_pmbm_bid, track_pruning_pmbm)


def carry_over(state, dump, extra, config=None):
    """Advance ``state`` in place to the next step using this step's outputs.

    ``config`` is only required for the clustered (k>=2) carry-over; the step-1
    (no-cluster) carry-over ignores it.
    """
    masters = np.asarray(extra["masters"], dtype=int).ravel()
    if masters.size == 0:
        _carry_over_no_clusters(state, dump, extra)
    else:
        if config is None:
            raise ValueError("config is required for the clustered carry-over")
        _carry_over_with_clusters(state, dump, extra, config)


# ---------------------------------------------------------------------------
# Stage 2a: no pre-existing clusters (step-1 carry-over)
# ---------------------------------------------------------------------------

def _carry_over_no_clusters(state, dump, extra):
    m = dump["measurements"].shape[1]
    indices_newborn = np.asarray(dump["indicesOfNewbornTracks"], dtype=int).ravel()
    mea_hist_col_new = np.asarray(extra["mea_hist_col_new"], dtype=float)

    new_hypos = []
    new_hypos_card = []
    new_prob_logs = []
    cW = []
    cCardW = []
    for jj in range(1, m + 1):
        t_index = int(indices_newborn[jj - 1])
        claimed = mea_hist_col_new[-1, np.array(new_hypos, dtype=int) - 1] \
            if new_hypos else np.zeros(0)
        if jj in claimed:
            continue
        new_hypos_card.append(1)
        new_hypos.append(t_index)
        new_prob_logs.append(0.0)
        cW.append(len(new_hypos_card))
        cCardW.append(1)

    state.hypos = np.array(new_hypos, dtype=int)
    state.hypos_card = np.array(new_hypos_card, dtype=int)
    state.clusters = np.array(cW, dtype=int)
    state.clusters_card = np.array(cCardW, dtype=int)
    state.prob_log_hypos = np.array(new_prob_logs, dtype=float)

    state.track_file = np.asarray(extra["track_file_new"], dtype=float)
    state.track_file_shadow = np.asarray(extra["track_file_shadow_new"], dtype=float)
    state.mea_hist_col = mea_hist_col_new

    assert np.all(state.clusters_card == 1), \
        "non-singleton clusters: pruningPmbmBid / clusterSplitting not a no-op"
    last = mea_hist_col_new[-1, :]
    assert np.unique(last).size == last.size, \
        "repeated measurement histories: nScanMerge not a no-op"


# ---------------------------------------------------------------------------
# Stage 2b: clusters exist (k>=2 carry-over)
# ---------------------------------------------------------------------------

def _carry_over_with_clusters(state, dump, extra, config):
    incol = state.incol
    k = int(dump["k"])
    n_hypo_total_max = int(config["nHypoTotalMax"])

    hypos = np.asarray(state.hypos, dtype=int).ravel()
    hypos_card = np.asarray(state.hypos_card, dtype=int).ravel()
    clusters = np.asarray(state.clusters, dtype=int).ravel()
    clusters_card = np.asarray(state.clusters_card, dtype=int).ravel()
    prob_log_hypos = np.asarray(state.prob_log_hypos, dtype=float).ravel()

    assoc_local = np.asarray(dump["assocLocal"], dtype=float)
    gain_mat_post_c = np.asarray(dump["gainMatPostC"], dtype=float)
    indices_newborn = np.asarray(dump["indicesOfNewbornTracks"], dtype=int).ravel()
    track_number_lookup = np.asarray(dump["trackNumberLookup"], dtype=float)

    track_file_new = np.asarray(extra["track_file_new"], dtype=float)
    track_file_shadow_new = np.asarray(extra["track_file_shadow_new"], dtype=float)
    mea_hist_col_new = np.asarray(extra["mea_hist_col_new"], dtype=float)

    n_masters = int(np.sum(assoc_local[1, :] == 1))
    m = indices_newborn.size
    n_tracks_new = mea_hist_col_new.shape[1]
    tf_cluster_membership = np.full(n_tracks_new, np.nan)

    # --- Per-supercluster branch-and-bound (script 1239-1602) ----------------
    new_hypos = []
    new_hypos_card = []
    new_prob_logs = []
    cW = []
    cCardW = []
    new_hypos_count = 0
    for iC in range(1, n_masters + 1):
        hl, hcl, pll = branch_and_bound_explore(
            hypos, hypos_card, clusters, clusters_card, prob_log_hypos, iC,
            assoc_local, gain_mat_post_c, indices_newborn, n_hypo_total_max,
            track_number_lookup, k)
        new_hypos.extend(hl.tolist())
        new_hypos_card.extend(hcl.tolist())
        new_prob_logs.extend(pll.tolist())
        new_hypos_old = new_hypos_count
        new_hypos_count += int(hcl.size)
        cW.extend(range(new_hypos_old + 1, new_hypos_count + 1))
        cCardW.append(int(hcl.size))
        tracks_in_cluster = np.unique(hl)
        if tracks_in_cluster.size:
            tf_cluster_membership[tracks_in_cluster - 1] = iC

    # --- Newborn separate-hypothesis clustering (script 1616-1637) -----------
    for jj in range(1, m + 1):
        t_index = int(indices_newborn[jj - 1])
        claimed = mea_hist_col_new[-1, np.array(new_hypos, dtype=int) - 1] \
            if new_hypos else np.zeros(0)
        if jj in claimed:
            continue
        new_hypos_card.append(1)
        new_hypos.append(t_index)
        new_prob_logs.append(0.0)
        hypo_number = len(new_hypos_card)
        t_cluster = tf_cluster_membership[t_index - 1]
        if np.isnan(t_cluster):
            cW.append(len(new_hypos_card))
            cCardW.append(1)
        else:
            cW_arr, cCardW_arr = insert_elements(
                int(t_cluster), hypo_number, np.array(cW, dtype=int),
                np.array(cCardW, dtype=int), 1)
            cW = cW_arr.astype(int).tolist()
            cCardW = cCardW_arr.astype(int).tolist()

    state.hypos = np.array(new_hypos, dtype=int)
    state.hypos_card = np.array(new_hypos_card, dtype=int)
    state.clusters = np.array(cW, dtype=int)
    state.clusters_card = np.array(cCardW, dtype=int)
    state.prob_log_hypos = np.array(new_prob_logs, dtype=float)
    state.track_file = track_file_new
    state.track_file_shadow = track_file_shadow_new
    state.mea_hist_col = mea_hist_col_new

    # --- Pruning (script 1865-1885) ------------------------------------------
    if state.clusters.size:
        (hypos, hypos_card, clusters, clusters_card, prob_log_hypos, track_file,
         track_file_shadow, mea_hist_col, _exist) = pruning_pmbm_bid(
            state.hypos, state.hypos_card, state.clusters, state.clusters_card,
            state.prob_log_hypos, state.track_file, state.track_file_shadow,
            state.mea_hist_col, incol, n_hypo_total_max, k)
        state.hypos, state.hypos_card = hypos, hypos_card
        state.clusters, state.clusters_card = clusters, clusters_card
        state.prob_log_hypos = prob_log_hypos
        state.track_file, state.track_file_shadow = track_file, track_file_shadow
        state.mea_hist_col = mea_hist_col

    # --- N-scan track merging (script 1926-1927) -----------------------------
    (hypos, hypos_card, clusters, clusters_card, prob_log_hypos, mea_hist_col,
     track_file, track_file_shadow) = n_scan_merge(
        state.hypos, state.hypos_card, state.clusters, state.clusters_card,
        state.prob_log_hypos, state.mea_hist_col, incol, state.track_file,
        state.track_file_shadow)
    state.hypos, state.hypos_card = hypos, hypos_card
    state.clusters, state.clusters_card = clusters, clusters_card
    state.prob_log_hypos = prob_log_hypos
    state.track_file, state.track_file_shadow = track_file, track_file_shadow
    state.mea_hist_col = mea_hist_col

    # --- Cluster splitting (script 1951) -------------------------------------
    (hypos_new, hypos_new_card, clusters_new, clusters_new_card, probs_as,
     splitting_tracks) = cluster_splitting_maj_traj(
        state.hypos, state.hypos_card, state.clusters, state.clusters_card,
        state.prob_log_hypos, state.track_file, config["sig"], state.mea_hist_col,
        incol, config["splitting_threshold"], config["maha_cs_thres"],
        config["non_split_lag"], k)

    # --- PHD recycling (script 1979-2024; doRunnall/doPostDMRecycle false) ----
    skipped = np.setdiff1d(state.hypos, hypos_new)
    assert skipped.size == splitting_tracks.size, \
        "skipped tracks must equal splitting tracks"
    if skipped.size:
        _, _, probabilities = track2cluster(
            skipped, state.hypos, state.hypos_card, state.clusters,
            state.clusters_card, state.prob_log_hypos)
        wS = state.track_file[incol.exi, skipped - 1] * probabilities
        phd_recycle = state.track_file[:, skipped - 1].copy()
        phd_recycle[incol.exi, :] = wS
        # incol.last is the row count (21); the "last" metadata row is index 20.
        for f in (incol.cost, incol.contrib, incol.visi, incol.label, incol.last - 1):
            phd_recycle[f, :] = np.nan
        state.phd_tracks = np.hstack([state.phd_tracks, phd_recycle])
    keep = state.phd_tracks[incol.exi, :] >= 1e-5
    state.phd_tracks = state.phd_tracks[:, keep]

    # --- Remove split-off tracks, finalise state (script 2034-2055) ----------
    hypos_new, track_file, track_file_shadow, mea_hist_col = track_pruning_pmbm(
        hypos_new, state.track_file, state.track_file_shadow, state.mea_hist_col, k)

    state.prob_log_hypos = np.log(probs_as)
    state.hypos = hypos_new
    state.hypos_card = hypos_new_card
    state.clusters = clusters_new
    state.clusters_card = clusters_new_card
    state.track_file = track_file
    state.track_file_shadow = track_file_shadow
    state.mea_hist_col = mea_hist_col
