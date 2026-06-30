"""Track filtering phase: KF updates for existing tracks and initialisation of
new (PPP-born) tracks.

Port of ``trackFiltering.m``. Some literal quirks of the MATLAB are reproduced
deliberately (the goal is to reproduce the exact dumped variables):

* ``xStuff(inCol.cost) = gainMatFull(newTrackSubs(1),newTrackSubs(2))`` uses
  linear indices 1 and 2, i.e. always the *first* tentative track's gain — so
  every new track stores the same cost. Reproduced as written.
* ``lambdauParents = find(isinf(gainMatLambdau(:,jAss)))`` selects the
  *non-gated* PHD components (whose weights are ``exp(-inf)=0``); only the
  ``muPPP`` component then carries weight in the mixture. Reproduced as written.
"""

import numpy as np
from scipy.linalg import block_diag

from .mathutils import c2p, cm_alternative, cov_mat_to_vec, gm_reduce


def track_filtering(pred_x, pred_z, pred_p, pred_s, new_track_subs, measurements,
                    system, incol, pD, track_file, gain_mat_full, gain_mat_lambdau,
                    muPPP, do_mechi, params, k, lambdau_inner, label_gen, max_lag,
                    track_file_shadow, mea_hist_col, pred_z_lambdau,
                    pred_s_lambdau, pred_p_lambdau):
    """Port of ``trackFiltering.m``.

    Returns (track_file_new, track_file_shadow_new, mea_hist_col_new,
    track_parents, indices_of_newborn_tracks). ``label_gen`` is consumed by
    value (matching the MATLAB, which does not return it).
    """
    measurements = np.asarray(measurements, dtype=float)
    if measurements.ndim == 1:
        measurements = measurements.reshape(2, -1)
    m = measurements.shape[1]
    n_tenta = new_track_subs.shape[1]
    n_tracks = track_file.shape[1]
    dim_tar = pred_x.shape[0]
    n_rows = incol.last

    indices_newborn = np.zeros(m, dtype=int)
    track_file_new = np.zeros((n_rows, n_tenta))
    track_file_shadow_new = np.zeros((n_rows, n_tenta, max_lag))
    mea_hist_col_new = np.zeros((max_lag, n_tenta))
    track_parents = np.full(n_tenta, np.nan)

    hMat = np.asarray(system["hMat"], dtype=float)
    rCart = np.asarray(system["rCart"], dtype=float)
    rPol = np.asarray(system["rPol"], dtype=float)
    own = np.asarray(params["stateFullOwn"], dtype=float)
    p_init_vel = np.asarray(params["pInitVel"], dtype=float)
    lambda_fa = float(params["lambdaFa"])

    # Constant "cost" bug: gainMatFull(newTrackSubs(1),newTrackSubs(2)).
    const_cost = gain_mat_full[new_track_subs[0, 0] - 1, new_track_subs[1, 0] - 1] \
        if n_tenta > 0 else 0.0

    for t in range(n_tenta):
        jAss = new_track_subs[1, t]            # 1-based column
        parent1 = new_track_subs[0, t]         # 1-based row

        if parent1 <= n_tracks:
            parent = parent1
            xBar = pred_x[:, parent - 1]
            zBar = pred_z[:, parent - 1]
            pBar = pred_p[:, :, parent - 1]
            sMat = pred_s[:, :, parent - 1]

            if jAss <= m:  # measurement assigned -> KF update
                z = measurements[:, jAss - 1]
                innov = z - zBar
                K = pBar @ hMat.T @ np.linalg.inv(sMat)
                xHat = xBar + K @ innov
                pHat = (np.eye(dim_tar) - K @ hMat) @ pBar
                pHat = (pHat + pHat.T) / 2.0
                jZ = jAss
                exi = 1.0
            else:          # misdetection
                xHat = xBar
                pHat = pBar
                jZ = 0
                exi_old = track_file[incol.exi, parent - 1]
                one_minus_pd = 1 - pD
                exi = exi_old * one_minus_pd / (1 - exi_old + exi_old * one_minus_pd)
            parent_for_label = parent
        else:  # parentless -> initialise from PPP/PHD
            z = measurements[:, jAss - 1]
            col = gain_mat_lambdau[:, jAss - 1]
            lambdau_parents = np.nonzero(np.isinf(col))[0]  # literal: isinf
            w_other = np.exp(col[lambdau_parents])
            w0 = muPPP
            wGM = np.concatenate([w_other, [w0]])
            wGM = wGM / np.sum(wGM)
            mu0 = np.concatenate([z, np.zeros(2)])

            if do_mechi:
                own2tar = z - own[0:2, k - 1]
                _, Ra = cm_alternative(c2p(own2tar), rPol)
                rMat = rCart + Ra
            else:
                rMat = rCart
            p_init = block_diag(rMat, p_init_vel)
            cov0 = p_init

            mu_other = np.zeros((dim_tar, w_other.size))
            cov_other = np.zeros((dim_tar, dim_tar, w_other.size))
            for ii in range(w_other.size):
                lp = lambdau_parents[ii]
                zBarL = pred_z_lambdau[:, lp]
                sMatL = pred_s_lambdau[:, :, lp]
                pBarL = pred_p_lambdau[:, :, lp]
                innov = z - zBarL
                K = pBarL @ hMat.T @ np.linalg.inv(sMatL)
                # NB: MATLAB uses xBar here, which is undefined for parentless
                # tracks at this point; reproduced only via the (zero-weight)
                # mixture so it never affects the result.
                mu_other[:, ii] = K @ innov
                cov_other[:, :, ii] = (np.eye(dim_tar) - K @ hMat) @ pBarL

            mu_gm = np.column_stack([mu_other, mu0]) if w_other.size else mu0[:, None]
            cov_gm = np.concatenate([cov_other, cov0[:, :, None]], axis=2)
            xHat, pHat = gm_reduce(mu_gm, cov_gm, wGM)

            r_numer = lambdau_inner[jAss - 1]
            exi = r_numer / (lambda_fa + r_numer)
            jZ = jAss
            label_gen = label_gen + 1
            parent = np.nan
            parent_for_label = None

        # Store the track-file column.
        x_stuff = np.zeros(n_rows)
        x_stuff[incol.tarX] = xHat
        x_stuff[incol.tarP] = cov_mat_to_vec(pHat).ravel()
        x_stuff[incol.meaLast] = jZ
        x_stuff[incol.cost] = const_cost
        x_stuff[incol.contrib] = x_stuff[incol.cost]
        x_stuff[incol.exi] = exi
        x_stuff[incol.visi] = 1
        if parent1 <= n_tracks:
            x_stuff[incol.label] = track_file[incol.label, parent_for_label - 1]
        else:
            x_stuff[incol.label] = label_gen

        track_parents[t] = parent
        track_file_new[:, t] = x_stuff

        # trackFileShadow update.
        if parent1 <= n_tracks:
            track_file_shadow_new[:, t, :max_lag - 1] = \
                track_file_shadow[:, parent - 1, 1:max_lag]
            track_file_shadow_new[:, t, max_lag - 1] = x_stuff
        else:
            track_file_shadow_new[:, t, :max_lag - 1] = np.nan
            track_file_shadow_new[:, t, max_lag - 1] = x_stuff

        if parent1 > n_tracks and jZ <= m:
            indices_newborn[jZ - 1] = t + 1  # 1-based tentative track number

        if parent1 <= n_tracks:
            prev_part = mea_hist_col[1:max_lag, parent - 1]
        else:
            prev_part = np.full(max_lag - 1, np.nan)
        mea_hist_col_new[:, t] = np.concatenate([prev_part, [jZ]])

    return (track_file_new, track_file_shadow_new, mea_hist_col_new,
            track_parents, indices_newborn)
