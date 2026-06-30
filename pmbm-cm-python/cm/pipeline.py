"""Per-step prior/likelihood pipeline of the CM PMBM filter.

Port of the prediction + validation + filtering + clustering phases of
``script_pmbm91.m:966-1166``. Produces the bundle of variables dumped at
``script_pmbm91.m:1236-1237`` (``priorLikelihood{k}.mat``).

Stage 1 covers the prior/likelihood computation only; the branch-and-bound
hypothesis generation and multi-step carry-over (which set ``hypos``/``clusters``
for the next step) are out of scope here.
"""

import numpy as np

from .clustering import clustering_preprocess
from .filtering import track_filtering
from .mathutils import c2p, cm_alternative, cov_mat_to_vec, cov_vec_to_mat
from .validation import bernoulli_validation, phd_validation


def _measurement_cov(do_mechi, pred_pos, own_k, rCart, rPol):
    if do_mechi:
        own2tar = pred_pos - own_k[0:2]
        _, Ra = cm_alternative(c2p(own2tar), rPol)
        return rCart + Ra
    return rCart


def run_prior_likelihood(state, scn, k, config):
    """Run one step of the prior/likelihood pipeline.

    Parameters
    ----------
    state : CMState (mutated: prediction updates muPPP / phd_tracks / existence).
    scn   : MatScenarioCM.
    k     : 1-based time step.
    config: dict with gamma_gate, pD, do_mechi, do_semi_two_point, adole_thres,
            nHypoMax, nHypoTotalMax, pre_cluster_threshold, label_gen.

    Returns
    -------
    dump : dict of the 24 variables for priorLikelihood{k}.mat.
    """
    incol = state.incol
    system = scn.system
    params = scn.params
    F = system["fMat"]
    Q = system["qMat"]
    H = system["hMat"]
    rCart = system["rCart"]
    rPol = system["rPol"]
    dim_tar = F.shape[0]
    dim_z = H.shape[0]

    pD = config["pD"]
    pS = float(params["pS"])
    do_mechi = config["do_mechi"]
    gamma_gate = config["gamma_gate"]
    lambda_fa = scn.lambda_fa
    own = np.asarray(params["stateFullOwn"], dtype=float)

    measurements = scn.measurements_at(k)
    m = measurements.shape[1]
    mu_birth = scn.mu_birth(k)

    # --- Prediction (script 972-1035) ------------------------------------
    state.muPPP = state.muPPP * (1 - pD) * pS + mu_birth
    muPPP = state.muPPP

    phd = state.phd_tracks
    nTPHD = phd.shape[1]
    phd[incol.exi, :] = pS * (1 - pD) * phd[incol.exi, :]
    pred_x_lambdau = F @ phd[incol.tarX, :]
    pred_z_lambdau = H @ pred_x_lambdau
    pred_p_lambdau = np.zeros((dim_tar, dim_tar, nTPHD))
    pred_s_lambdau = np.zeros((dim_z, dim_z, nTPHD))
    prev_p_phd = cov_vec_to_mat(phd[incol.tarP, :]) if nTPHD > 0 \
        else np.zeros((dim_tar, dim_tar, 0))
    for ii in range(nTPHD):
        pred_p_lambdau[:, :, ii] = F @ prev_p_phd[:, :, ii] @ F.T + Q
        rMat = _measurement_cov(do_mechi, pred_x_lambdau[0:2, ii], own[:, k - 1], rCart, rPol)
        pred_s_lambdau[:, :, ii] = H @ pred_p_lambdau[:, :, ii] @ H.T + rMat
    if nTPHD > 0:
        phd[incol.tarX, :] = pred_x_lambdau
        phd[incol.tarP, :] = cov_mat_to_vec(pred_p_lambdau)

    track_file = state.track_file
    n_tracks = track_file.shape[1]
    pred_x = F @ track_file[incol.tarX, :]
    pred_p = np.zeros((dim_tar, dim_tar, n_tracks))
    prev_p = cov_vec_to_mat(track_file[incol.tarP, :]) if n_tracks > 0 \
        else np.zeros((dim_tar, dim_tar, 0))
    pred_s = np.zeros((dim_z, dim_z, n_tracks))
    pred_z = H @ pred_x
    for ii in range(n_tracks):
        pred_p[:, :, ii] = F @ prev_p[:, :, ii] @ F.T + Q
        rMat = _measurement_cov(do_mechi, pred_x[0:2, ii], own[:, k - 1], rCart, rPol)
        pred_s[:, :, ii] = H @ pred_p[:, :, ii] @ H.T + rMat
    # Predict existence in the MBM component.
    track_file[incol.exi, :] = pS * track_file[incol.exi, :]

    # --- Validation (script 1061-1090) -----------------------------------
    lambdau_inner, gain_mat_lambdau = phd_validation(
        phd, measurements, pred_z_lambdau, pred_s_lambdau, gamma_gate, incol, pD, muPPP)

    (mea_hist_col_old, track_number_lookup, pred_x, pred_z, pred_p, pred_s,
     gain_mat_full, hypos, hypos_card, track_file, track_file_shadow,
     mea_hist_col, new_track_subs, n_tracks_tenta) = bernoulli_validation(
        track_file, state.mea_hist_col, measurements, lambdau_inner, pD, incol,
        config["do_semi_two_point"], pred_x, pred_z, pred_p, pred_s, prev_p,
        gamma_gate, lambda_fa, state.hypos, state.hypos_card, state.clusters,
        state.clusters_card, state.prob_log_hypos, state.track_file_shadow, k,
        config["adole_thres"])

    state.hypos = hypos
    state.hypos_card = hypos_card
    state.track_file = track_file
    state.track_file_shadow = track_file_shadow
    state.mea_hist_col = mea_hist_col

    # --- Track filtering (script 1111) -----------------------------------
    (track_file_new, track_file_shadow_new, mea_hist_col_new, track_parents,
     indices_newborn) = track_filtering(
        pred_x, pred_z, pred_p, pred_s, new_track_subs, measurements, system,
        incol, pD, track_file, gain_mat_full, gain_mat_lambdau, muPPP, do_mechi,
        params, k, lambdau_inner, config["label_gen"], state.max_lag,
        track_file_shadow, mea_hist_col, pred_z_lambdau, pred_s_lambdau,
        pred_p_lambdau)

    # --- Clustering preprocess (script 1166) -----------------------------
    assoc_local, gain_mat_post_c, masters = clustering_preprocess(
        state.hypos, state.hypos_card, state.clusters, state.clusters_card,
        gain_mat_full, mea_hist_col, m, config["pre_cluster_threshold"])

    dump = {
        "hypos": np.asarray(state.hypos, dtype=float).reshape(1, -1),
        "hyposCard": np.asarray(state.hypos_card, dtype=float).reshape(1, -1),
        "clusters": np.asarray(state.clusters, dtype=float).reshape(1, -1),
        "clustersCard": np.asarray(state.clusters_card, dtype=float).reshape(1, -1),
        "probLogHypos": np.asarray(state.prob_log_hypos, dtype=float).reshape(1, -1),
        "assocLocal": np.asarray(assoc_local, dtype=float),
        "gainMatPostC": np.asarray(gain_mat_post_c, dtype=float),
        "indicesOfNewbornTracks": np.asarray(indices_newborn, dtype=float).reshape(1, -1),
        "nHypoMax": float(config["nHypoMax"]),
        "nHypoTotalMax": float(config["nHypoTotalMax"]),
        "trackNumberLookup": np.asarray(track_number_lookup, dtype=float),
        "k": float(k),
        "trackFile": np.asarray(track_file, dtype=float),
        "inCol": incol.to_matlab_struct(),
        "hTrue": np.asarray(scn.hTrue, dtype=float),
        "trackFileShadow": np.asarray(track_file_shadow, dtype=float),
        "measurements": np.asarray(measurements, dtype=float),
        "predX": np.asarray(pred_x, dtype=float),
        "predZ": np.asarray(pred_z, dtype=float),
        "predP": np.asarray(pred_p, dtype=float),
        "predS": np.asarray(pred_s, dtype=float),
        "pD": float(pD),
        "meaHistCol": np.asarray(mea_hist_col, dtype=float),
        "meaHistColNew": np.asarray(mea_hist_col_new, dtype=float),
    }

    # Intermediate outputs that later stages (hypothesis generation) consume.
    extra = {
        "track_file_new": track_file_new,
        "track_file_shadow_new": track_file_shadow_new,
        "track_parents": track_parents,
        "gain_mat_full": gain_mat_full,
        "gain_mat_lambdau": gain_mat_lambdau,
        "new_track_subs": new_track_subs,
        "n_tracks_tenta": n_tracks_tenta,
        "masters": np.asarray(masters, dtype=int).ravel(),
        "mea_hist_col_new": mea_hist_col_new,
    }
    return dump, extra
