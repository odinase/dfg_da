"""Multicluster, multihypothesis Murty marginals and normalization constant.

Python port of the MATLAB driver ``compute_results.m`` (the "Murty" baseline of the
thesis), which is itself built on ``branchAndBoundExplore.m`` -- an outer/inner Murty
branch-and-bound over (prior-hypothesis combinations x track->measurement assignments)
for one supercluster.

The branch-and-bound itself is **not** re-implemented here: ``pmbm-cm-python`` already
carries a faithful port (``cm.branchbound.branch_and_bound_explore``), and
``cm.carryover._carry_over_with_clusters`` already ports the supercluster loop and the
newborn-hypothesis fixup. What this module adds is the half of ``compute_results.m``
that was never ported: the accumulation of the k-best global hypotheses into track
association marginals (``trackProbAccumulatePureOdin`` + ``margs2distrs``) and the
normalization constant (per-cluster ``logsumexp``, summed in the log domain).

Layout of the returned marginals matches every other computer in this repo and
``MatFileParser.reward_matrix_lc``::

    (num_tracks, num_measurements + 2)
    column 0          misdetection
    columns 1..m      measurement j
    column -1         non-existence

so it can be compared elementwise with ``MulticlusterExactOutput.exact_marginals``.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

# ``pmbm-cm-python`` is a plain package directory (no setup.py); put it on the path so
# ``cm`` imports resolve. Done at import time so it survives the ``spawn`` start method
# used by multiprocessing.Pool on macOS.
_CM_ROOT = Path(__file__).resolve().parents[1] / "pmbm-cm-python"
if str(_CM_ROOT) not in sys.path:
    sys.path.insert(0, str(_CM_ROOT))

from cm.bbhelpers import insert_elements  # noqa: E402
from cm.branchbound import branch_and_bound_explore  # noqa: E402
from cm.clouds import tcloud_to_beg, tcloud_to_end, track2cluster  # noqa: E402


# The nHypoTotalMax sweep of compute_results.m:28.
K_SWEEP: Tuple[int, ...] = (10, 20, 50, 100, 150)


@dataclass
class MulticlusterMurtyOutput:
    """Result of one Murty run at a single truncation depth ``n_hypo_total_max``.

    ``marginals``/``likelihood`` are named to match ``MulticlusterConditionendLBPOutput``
    so ``plot_convergence_stats.METHODS`` can address them the same way.
    """

    marginals: np.ndarray
    likelihood: float
    n_hypotheses: int
    runtime: float
    n_hypo_total_max: int


def _posterior_hypothesis_cloud(ws: Mapping[str, Any], n_hypo_total_max: int):
    """Run the branch-and-bound over every supercluster and rebuild the posterior
    hypothesis cloud.

    Port of ``compute_results.m:58-122``, structurally identical to
    ``cm.carryover._carry_over_with_clusters:116-159``.

    Returns ``(hypos, hypos_card, clusters, clusters_card, prob_log_hypos)``.
    """
    ravel = lambda key, dtype: np.asarray(ws[key], dtype=dtype).ravel()

    hypos = ravel("hypos", int)
    hypos_card = ravel("hyposCard", int)
    clusters = ravel("clusters", int)
    clusters_card = ravel("clustersCard", int)
    prob_log_hypos = ravel("probLogHypos", float)

    assoc_local = np.asarray(ws["assocLocal"], dtype=float)
    gain_mat_post_c = np.asarray(ws["gainMatPostC"], dtype=float)
    indices_newborn = ravel("indicesOfNewbornTracks", int)
    track_number_lookup = np.asarray(ws["trackNumberLookup"], dtype=float)
    mea_hist_col_new = np.asarray(ws["meaHistColNew"], dtype=float)
    k = int(np.asarray(ws["k"]).ravel()[0])

    m = indices_newborn.size
    n_masters = int(np.sum(assoc_local[1, :])) if assoc_local.size else 0

    new_hypos: list = []
    new_hypos_card: list = []
    new_prob_logs: list = []
    cW: list = []
    cCardW: list = []
    new_hypos_count = 0
    membership = np.full(mea_hist_col_new.shape[1], np.nan)

    # Every supercluster is solved against the *prior* cloud arrays; the rebind to the
    # posterior cloud happens only after the loop (as in the MATLAB).
    for iC in range(1, n_masters + 1):
        hl, hcl, pll = branch_and_bound_explore(
            hypos, hypos_card, clusters, clusters_card, prob_log_hypos, iC,
            assoc_local, gain_mat_post_c, indices_newborn, n_hypo_total_max,
            track_number_lookup, k)
        new_hypos.extend(hl.tolist())
        new_hypos_card.extend(hcl.tolist())
        new_prob_logs.extend(pll.tolist())
        old_count = new_hypos_count
        new_hypos_count += int(hcl.size)
        cW.extend(range(old_count + 1, new_hypos_count + 1))
        cCardW.append(int(hcl.size))
        if hl.size:
            membership[np.unique(hl) - 1] = iC

    # A newborn track whose measurement no other cluster claimed becomes its own
    # single-hypothesis cluster (compute_results.m:95-116).
    for jj in range(1, m + 1):
        t_index = int(indices_newborn[jj - 1])
        claimed = (mea_hist_col_new[-1, np.array(new_hypos, dtype=int) - 1]
                   if new_hypos else np.zeros(0))
        if jj in claimed:
            continue
        new_hypos_card.append(1)
        new_hypos.append(t_index)
        new_prob_logs.append(0.0)  # log(1)
        hypo_number = len(new_hypos_card)
        t_cluster = membership[t_index - 1]
        if np.isnan(t_cluster):
            cW.append(len(new_hypos_card))
            cCardW.append(1)
        else:
            cW_arr, cCardW_arr = insert_elements(
                int(t_cluster), hypo_number, np.array(cW, dtype=int),
                np.array(cCardW, dtype=int), 1)
            cW = cW_arr.astype(int).tolist()
            cCardW = cCardW_arr.astype(int).tolist()

    return (np.array(new_hypos, dtype=int), np.array(new_hypos_card, dtype=int),
            np.array(cW, dtype=int), np.array(cCardW, dtype=int),
            np.array(new_prob_logs, dtype=float))


def margs2distrs(track_probs: np.ndarray, track_number_lookup: np.ndarray,
                 num_tracks: int, num_measurements: int) -> np.ndarray:
    """Port of the local function ``margs2distrs`` in ``compute_results.m:165-188``.

    ``track_probs[t-1]`` is the total posterior probability of *posterior* track ``t``;
    ``track_number_lookup[row, col]`` maps a cell of ``gainMatPostC`` to the posterior
    track id it creates. Rows are prior tracks, columns ``0..m-1`` are measurements and
    columns ``>= m`` are the misdetection/existence block.
    """
    distrs = np.zeros((num_tracks, num_measurements + 2))
    for t in range(1, track_probs.size + 1):
        rows, cols = np.nonzero(track_number_lookup == t)
        if rows.size == 0:
            continue
        row, col = int(rows[0]), int(cols[0])
        if row >= num_tracks:  # a newborn track, not a row of the prior reward matrix
            continue
        is_misdetection = col >= num_measurements
        distrs[row, 0 if is_misdetection else col + 1] += track_probs[t - 1]
    distrs[:, -1] = 1.0 - distrs[:, :-1].sum(axis=1)
    return distrs


def murty_marginals_likelihood(ws: Mapping[str, Any], num_tracks: int,
                               num_measurements: int, n_hypo_total_max: int
                               ) -> MulticlusterMurtyOutput:
    """Marginals + normalization constant from the ``n_hypo_total_max``-best global
    hypotheses of every supercluster in one scan.

    ``ws`` is the raw ``scipy.io.loadmat`` dict (i.e. ``MatFileParser.ws``).
    """
    start = time.time()

    hypos, hypos_card, clusters, clusters_card, prob_log_hypos = \
        _posterior_hypothesis_cloud(ws, n_hypo_total_max)

    track_number_lookup = np.asarray(ws["trackNumberLookup"], dtype=float)
    n_posterior_tracks = int(np.sum(~np.isnan(track_number_lookup)))

    # track2cluster's third return value is trackProbAccumulatePureOdin's trackSumProbs:
    # the per-cluster-normalized posterior mass of every hypothesis containing the track.
    if n_posterior_tracks > 0 and hypos.size:
        _, _, track_probs = track2cluster(
            np.arange(1, n_posterior_tracks + 1), hypos, hypos_card, clusters,
            clusters_card, prob_log_hypos)
    else:
        track_probs = np.zeros(n_posterior_tracks)

    marginals = margs2distrs(track_probs, track_number_lookup, num_tracks,
                             num_measurements)

    # Z = prod over posterior clusters of sum over that cluster's hypotheses of exp(score)
    # (compute_results.m:139-149), accumulated in the log domain.
    begs = tcloud_to_beg(clusters_card)
    ends = tcloud_to_end(clusters_card)
    log_z = 0.0
    for iC in range(clusters_card.size):
        hypo_ids = clusters[begs[iC] - 1:ends[iC]] - 1
        log_z += float(logsumexp(prob_log_hypos[hypo_ids]))

    return MulticlusterMurtyOutput(
        marginals=marginals,
        likelihood=float(np.exp(log_z)),
        n_hypotheses=int(hypos_card.size),
        runtime=time.time() - start,
        n_hypo_total_max=int(n_hypo_total_max),
    )


def murty_sweep(ws: Mapping[str, Any], num_tracks: int, num_measurements: int,
                ks: Sequence[int] = K_SWEEP) -> Dict[int, MulticlusterMurtyOutput]:
    """``murty_marginals_likelihood`` at every truncation depth in ``ks``."""
    return {int(k): murty_marginals_likelihood(ws, num_tracks, num_measurements, int(k))
            for k in ks}
