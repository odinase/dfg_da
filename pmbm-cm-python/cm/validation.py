"""Validation phase: gating of PHD (PPP) and Bernoulli tracks.

Ports of ``phdValidation.m``, ``bernoulliValidation.m``,
``trackProbAccumulatePure.m`` and ``pruneSelectedTracks.m``.
"""

import numpy as np

from .clouds import a2bc, remove_ind_a, track2cluster
from .mathutils import find_colmajor, normpdf_log, v2m


def phd_validation(phd_tracks, measurements, pred_z_lambdau, pred_s_lambdau,
                   gamma_gate, incol, pD, muPPP):
    """Port of ``phdValidation.m``.

    Returns (lambdau_inner_prods (m,), gain_mat_lambdau ((nTPHD+1, m))).
    """
    measurements = np.asarray(measurements, dtype=float)
    if measurements.ndim == 1:
        measurements = measurements.reshape(2, -1)
    m = measurements.shape[1]
    nTPHD = phd_tracks.shape[1]

    gain = np.full((nTPHD + 1, m), -np.inf)
    for t in range(nTPHD):
        zBar = pred_z_lambdau[:, t]
        sMat = pred_s_lambdau[:, :, t]
        exi = phd_tracks[incol.exi, t]
        for jj in range(m):
            innov = measurements[:, jj] - zBar
            gate = innov @ np.linalg.solve(sMat, innov)
            if gate < gamma_gate:
                gain[t, jj] = np.log(pD) + np.log(exi) + normpdf_log(
                    measurements[:, jj], zBar, sMat)[0]
    gain[nTPHD, :] = np.log(pD) + np.log(muPPP)

    lambdau_inner = np.zeros(m)
    for jj in range(m):
        col = gain[:, jj]
        lambdau_inner[jj] = np.sum(np.exp(col[~np.isinf(col)]))
    return lambdau_inner, gain


def track_prob_accumulate_pure(track_file, incol, hypos, hypos_card,
                               clusters, clusters_card, prob_log_hypos):
    """Port of ``trackProbAccumulatePure.m``.

    Returns (track_sum_probs (nTracks,), track_total_probs (nTracks,),
    probabilities_cell list).
    """
    from .clouds import prob_logs_to_probabilities

    nTracks = track_file.shape[1]
    prob_cell = prob_logs_to_probabilities(prob_log_hypos, clusters, clusters_card)
    track_sum = np.zeros(nTracks)
    track_total = np.zeros(nTracks)

    for t in range(nTracks):
        cluster_numbers, hypos_col, _ = track2cluster(
            t + 1, hypos, hypos_card, clusters, clusters_card, prob_log_hypos)
        cn = cluster_numbers[0]
        if not np.isnan(cn):
            h_list = hypos_col[0]  # B-level hypothesis indices in the cluster
            in_clusters_a = []
            for hv in h_list:
                in_clusters_a.extend((np.nonzero(
                    np.asarray(clusters, dtype=int).ravel() == hv)[0] + 1).tolist())
            b, _ = a2bc(np.array(in_clusters_a), clusters_card)
            probs_this = prob_cell[int(cn) - 1]
            track_sum[t] = np.sum(probs_this[b.astype(int) - 1])
            track_total[t] = track_sum[t] * track_file[incol.exi, t]
    return track_sum, track_total, prob_cell


def prune_selected_tracks(hypos, hypos_card, tracks_to_prune, track_file,
                          track_file_shadow, mea_hist_col, k):
    """Port of ``pruneSelectedTracks.m``. ``tracks_to_prune`` is 1-based.

    Returns (hypos_new, hypos_card_new, track_file, track_file_shadow,
    mea_hist_col, to_be_kept (1-based)).
    """
    hypos = np.asarray(hypos, dtype=int).ravel()
    nTracks = track_file.shape[1]
    tracks_to_prune = np.asarray(tracks_to_prune, dtype=int).ravel()

    new_numbers = np.zeros(nTracks + 1, dtype=int)  # 1-based indexing
    counter = 1
    for ii in range(1, nTracks + 1):
        if ii not in tracks_to_prune:
            new_numbers[ii] = counter
            counter += 1

    to_be_kept = np.setdiff1d(np.arange(1, nTracks + 1), tracks_to_prune)
    track_file = track_file[:, to_be_kept - 1]
    track_file_shadow = track_file_shadow[:, to_be_kept - 1, :]
    mea_hist_col = mea_hist_col[:, to_be_kept - 1]

    hypos_new_a = np.array([new_numbers[h] for h in hypos], dtype=int)
    a_remove = np.nonzero(hypos_new_a == 0)[0] + 1
    _, _, a_remain, hypos_card_new = remove_ind_a(a_remove, hypos_card)
    hypos_new = hypos_new_a[a_remain - 1] if a_remain.size else np.zeros(0, dtype=int)
    return hypos_new, hypos_card_new, track_file, track_file_shadow, mea_hist_col, to_be_kept


def bernoulli_validation(track_file, mea_hist_col, measurements, lambdau_inner,
                         pD, incol, do_semi_two_point, pred_x, pred_z, pred_p,
                         pred_s, prev_p, gamma_gate, lambda_fa, hypos, hypos_card,
                         clusters, clusters_card, prob_log_hypos,
                         track_file_shadow, k, adole_thres):
    """Port of ``bernoulliValidation.m``.

    Returns (mea_hist_col_old, track_number_lookup, pred_x, pred_z, pred_p,
    pred_s, gain_mat_full, hypos, hypos_card, track_file, track_file_shadow,
    mea_hist_col, new_track_subs (2, nTracksTenta, 1-based), nTracksTenta).
    """
    measurements = np.asarray(measurements, dtype=float)
    if measurements.ndim == 1:
        measurements = measurements.reshape(2, -1)

    track_sum, track_total, _ = track_prob_accumulate_pure(
        track_file, incol, hypos, hypos_card, clusters, clusters_card, prob_log_hypos)

    nTracks = track_file.shape[1]
    m = measurements.shape[1]
    mea_hist_col_old = mea_hist_col.copy()

    gain = np.full((nTracks + m, m + nTracks), -np.inf)
    for t in range(nTracks):
        predExi = track_file[incol.exi, t]
        zBar = pred_z[:, t]
        sMat = pred_s[:, :, t]
        for jj in range(m):
            innov = measurements[:, jj] - zBar
            gate = innov @ np.linalg.solve(sMat, innov)
            if gate < gamma_gate:
                gain[t, jj] = (-np.log(lambda_fa + pD * lambdau_inner[jj])
                               + normpdf_log(measurements[:, jj], zBar, sMat)[0]
                               + np.log(pD) + np.log(predExi))
        gain[t, m + t] = np.log(1 - predExi + predExi * (1 - pD))
    for jj in range(m):
        gain[nTracks + jj, jj] = 0.0

    if do_semi_two_point:
        # Adolescent (semi-two-point) suppression of weak new-track assignments.
        not_nan = ~np.isnan(mea_hist_col)
        adolescents = np.nonzero(np.sum(not_nan, axis=0) == 1)[0]  # 0-based
        for ai in adolescents:
            last_row = mea_hist_col[-1, :]
            non_adole = np.nonzero((np.sum(not_nan, axis=0) > 1)
                                   & (last_row == mea_hist_col[-1, ai]))[0]
            if non_adole.size > 0:
                alpha = track_total[non_adole]
                beta = np.full(non_adole.size, -np.inf)
                for jx, na in enumerate(non_adole):
                    row = gain[jx, :m] if jx < gain.shape[0] else np.array([-np.inf])
                    finite = row[~np.isinf(row)]
                    if finite.size > 0:
                        beta[jx] = np.max(finite)
                alpha_beta_max = np.max(alpha * np.exp(beta))
                ch = np.nonzero(gain[ai, :m])[0]
                ch = ch[gain[ai, ch] != 0]  # MATLAB find() of nonzero entries
                for c in np.nonzero(gain[ai, :m] != 0)[0]:
                    adole_prod = track_total[ai] * np.exp(gain[ai, c])
                    if alpha_beta_max > adole_thres * adole_prod:
                        gain[ai, c] = -np.inf

        infeasible = np.all(np.isinf(gain[:nTracks, :]), axis=1)
        if np.any(infeasible):
            tracks_to_prune = np.nonzero(infeasible)[0] + 1
            (hypos, hypos_card, track_file, track_file_shadow, mea_hist_col,
             to_be_kept) = prune_selected_tracks(
                hypos, hypos_card, tracks_to_prune, track_file,
                track_file_shadow, mea_hist_col, k)
            keep0 = to_be_kept - 1
            pred_x = pred_x[:, keep0]
            pred_p = pred_p[:, :, keep0]
            prev_p = prev_p[:, :, keep0]
            pred_s = pred_s[:, :, keep0]
            pred_z = pred_z[:, keep0]
            gain = np.delete(gain, np.nonzero(infeasible)[0], axis=0)
            gain = np.delete(gain, m + (tracks_to_prune - 1), axis=1)
            nTracks = nTracks - int(np.sum(infeasible))

    if m > 0 and np.any(np.all(np.isinf(gain[:, m:]), axis=0)):
        raise ValueError("Entire columns in misdetection part are Inf")

    gain_copy = gain.copy()
    gain_copy[nTracks:, nTracks + m:] = -np.inf
    gain_copy_t = gain_copy.T
    indices = find_colmajor(gain_copy_t > -np.inf)
    new_track_subs = np.flipud(v2m(indices, gain_copy_t.shape[0]))

    nTracksTenta = new_track_subs.shape[1]
    track_number_lookup = np.full(gain.shape, np.nan)
    for ii in range(nTracksTenta):
        track_number_lookup[new_track_subs[0, ii] - 1,
                            new_track_subs[1, ii] - 1] = ii + 1

    return (mea_hist_col_old, track_number_lookup, pred_x, pred_z, pred_p,
            pred_s, gain, hypos, hypos_card, track_file, track_file_shadow,
            mea_hist_col, new_track_subs, nTracksTenta)
