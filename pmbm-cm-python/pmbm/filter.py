"""Poisson Multi-Bernoulli Mixture (PMBM) filter.

Direct port of the reference MATLAB implementation by Angel F. Garcia-Fernandez
(``PoissonMBMtarget_pred.m``, ``PoissonMBMtarget_update.m``,
``PoissonMBMtarget_pruning.m`` and ``PoissonMBMtarget_estimate{1,2,3}.m``),
following the paper

    A. F. Garcia-Fernandez, J. L. Williams, K. Granstrom and L. Svensson,
    "Poisson Multi-Bernoulli Mixture Filter: Direct Derivation and
    Implementation," IEEE TAES, vol. 54, no. 4, pp. 1883-1901, Aug. 2018.

State representation
--------------------
The filter state is a ``dict`` with keys:

* ``weightPois``   : list of Poisson (PPP) component weights.
* ``meanPois``     : list of (Nx,) PPP component means.
* ``covPois``      : list of (Nx, Nx) PPP component covariances.
* ``tracks``       : list of Bernoulli "tracks" (see below).
* ``globHyp``      : int ndarray (n_glob, n_tracks). Entry (h, i) is the
                     0-based single-target hypothesis index used for track i in
                     global hypothesis h, or -1 if track i is absent there.
* ``globHypWeight``: 1D ndarray of global-hypothesis weights.

Each track is a ``dict`` with keys ``meanB`` (list of means), ``covB`` (list of
covariances), ``eB`` (existence probabilities), ``t_ini`` (birth time),
``aHis`` (list of association histories; 0 = misdetection, m>=1 = measurement
index), ``weightBLog`` (log weights) and, transiently after an update,
``weightBLog_k`` (single-step log weights).

Indexing differs from MATLAB only in being 0-based: single-target hypotheses
within a track are laid out as ``index = j + Nhyp_i * (t + 1)`` where ``j`` is
the parent hypothesis (0..Nhyp_i-1) and ``t+1`` selects the misdetection
(t = -1, i.e. index ``j``) or measurement ``t`` (0-based).
"""

import numpy as np

from .cardinality import cardinality_mb
from .murty import murty


def _gauss_pdf(x, mean, cov):
    """Multivariate normal density (replacement for MATLAB ``mvnpdf``)."""
    d = np.asarray(x, dtype=float) - np.asarray(mean, dtype=float)
    n = d.size
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, d)
    maha = sol @ sol
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return np.exp(-0.5 * (maha + logdet + n * np.log(2.0 * np.pi)))


def predict(flt_upd, F, Q, p_s, weights_b, means_b, covs_b):
    """Prediction step. Port of ``PoissonMBMtarget_pred.m``."""
    flt = {}

    Ncom = len(flt_upd["weightPois"])
    births_w = list(np.asarray(weights_b, dtype=float).ravel())
    births_m = [means_b[:, j].astype(float).copy() for j in range(means_b.shape[1])]
    births_c = [np.asarray(c, dtype=float).copy() for c in covs_b]

    if Ncom > 0:
        flt["weightPois"] = [p_s * w for w in flt_upd["weightPois"]]
        flt["meanPois"] = [F @ m for m in flt_upd["meanPois"]]
        flt["covPois"] = [F @ c @ F.T + Q for c in flt_upd["covPois"]]
        flt["weightPois"] += births_w
        flt["meanPois"] += births_m
        flt["covPois"] += births_c
    else:
        flt["weightPois"] = list(births_w)
        flt["meanPois"] = list(births_m)
        flt["covPois"] = list(births_c)

    Ntracks = len(flt_upd["tracks"])
    if Ntracks > 0:
        flt["globHyp"] = flt_upd["globHyp"].copy()
        flt["globHypWeight"] = np.asarray(flt_upd["globHypWeight"], dtype=float).copy()

        tracks = []
        for i in range(Ntracks):
            t = flt_upd["tracks"][i]
            Nhyp_i = len(t["eB"])
            nt = {
                "t_ini": t["t_ini"],
                "meanB": [F @ t["meanB"][j] for j in range(Nhyp_i)],
                "covB": [F @ t["covB"][j] @ F.T + Q for j in range(Nhyp_i)],
                "eB": p_s * np.asarray(t["eB"], dtype=float),
                "aHis": [np.array(a, dtype=int) for a in t["aHis"]],
                "weightBLog": np.asarray(t["weightBLog"], dtype=float).copy(),
            }
            tracks.append(nt)
        flt["tracks"] = tracks
    else:
        # Faithful to the MATLAB reference: with no tracks the predicted PPP is
        # reset to the birth intensity only.
        flt["weightPois"] = list(births_w)
        flt["meanPois"] = list(births_m)
        flt["covPois"] = list(births_c)
        flt["tracks"] = []
        flt["globHyp"] = np.zeros((0, 0), dtype=int)
        flt["globHypWeight"] = np.zeros(0)

    return flt


def update(flt_pred, z, H, R, p_d, k, gating_threshold, intensity_clutter, Nhyp_max):
    """Update step. Port of ``PoissonMBMtarget_update.m``."""
    Nz, Nx = H.shape
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(Nz, -1)
    Nmeas = z.shape[1]
    Nprev = len(flt_pred["tracks"])

    flt = {}
    # ---- Poisson (PPP) update: undetected components survive with weight (1-p_d).
    flt["weightPois"] = [(1.0 - p_d) * w for w in flt_pred["weightPois"]]
    flt["meanPois"] = [m.copy() for m in flt_pred["meanPois"]]
    flt["covPois"] = [c.copy() for c in flt_pred["covPois"]]

    tracks = [None] * (Nprev + Nmeas)

    # ---- New track generation (one potential track per measurement).
    for m in range(Nmeas):
        z_m = z[:, m]

        indices_new = []
        for i in range(len(flt_pred["weightPois"])):
            cov_i = flt_pred["covPois"][i]
            S = H @ cov_i @ H.T + R
            nu = z_m - H @ flt_pred["meanPois"][i]
            maha = nu @ np.linalg.solve(S, nu)
            if maha < gating_threshold:
                indices_new.append(i)

        trk = {}
        if indices_new:
            meanB = np.zeros(Nx)
            covB = np.zeros((Nx, Nx))
            weightB = 0.0
            for i in indices_new:
                mean_i = flt_pred["meanPois"][i]
                cov_i = flt_pred["covPois"][i]
                S = H @ cov_i @ H.T + R
                z_pred = H @ mean_i
                K = cov_i @ H.T @ np.linalg.inv(S)
                P_u = (np.eye(Nx) - K @ H) @ cov_i
                P_u = (P_u + P_u.T) / 2.0
                x_u = mean_i + K @ (z_m - z_pred)
                w_i = p_d * _gauss_pdf(z_m, z_pred, S) * flt_pred["weightPois"][i]
                weightB += w_i
                meanB += w_i * x_u
                covB += w_i * P_u + w_i * np.outer(x_u, x_u)

            meanB = meanB / weightB
            covB = covB / weightB - np.outer(meanB, meanB)
            eB = weightB
            weightB = weightB + intensity_clutter
            eB = eB / weightB

            trk["meanB"] = [meanB]
            trk["covB"] = [covB]
            trk["eB"] = np.array([eB])
            trk["t_ini"] = k
            trk["aHis"] = [np.array([m + 1], dtype=int)]
            trk["weightBLog"] = np.array([np.log(weightB)])
        else:
            # Existence probability zero (pure clutter); removed by pruning.
            weightB = intensity_clutter
            trk["meanB"] = [np.zeros(Nx)]
            trk["covB"] = [np.zeros((Nx, Nx))]
            trk["eB"] = np.array([0.0])
            trk["t_ini"] = k
            trk["aHis"] = [np.array([m + 1], dtype=int)]
            trk["weightBLog"] = np.array([np.log(weightB)])

        tracks[Nprev + m] = trk

    # ---- Update of all previous tracks with the measurements.
    for i in range(Nprev):
        pt = flt_pred["tracks"][i]
        Nhyp_i = len(pt["eB"])
        total_len = Nhyp_i * (Nmeas + 1)

        meanB = [None] * total_len
        covB = [None] * total_len
        eB = np.zeros(total_len)
        aHis = [None] * total_len
        weightBLog = np.full(total_len, -np.inf)
        weightBLog_k = np.full(total_len, -np.inf)

        for j in range(Nhyp_i):
            mean_j = pt["meanB"][j]
            cov_j = pt["covB"][j]
            eB_j = pt["eB"][j]
            aHis_j = np.asarray(pt["aHis"][j], dtype=int)
            wlog_j = pt["weightBLog"][j]

            # Misdetection hypothesis (kept at index j).
            denom = 1.0 - eB_j + eB_j * (1.0 - p_d)
            meanB[j] = mean_j
            covB[j] = cov_j
            eB[j] = eB_j * (1.0 - p_d) / denom
            aHis[j] = np.append(aHis_j, 0)
            weightBLog[j] = wlog_j + np.log(denom)
            weightBLog_k[j] = np.log(denom)

            # Kalman-filter moments shared by all measurement hypotheses.
            S = H @ cov_j @ H.T + R
            z_pred = H @ mean_j
            L = np.linalg.cholesky(S)
            log_det_S = 2.0 * np.sum(np.log(np.diag(L)))
            inv_S = np.linalg.inv(S)
            K = cov_j @ H.T @ inv_S
            P_u = (np.eye(Nx) - K @ H) @ cov_j
            P_u = (P_u + P_u.T) / 2.0

            for t in range(Nmeas):
                idx = j + Nhyp_i * (t + 1)
                nu = z[:, t] - z_pred
                maha = nu @ inv_S @ nu
                if maha < gating_threshold:
                    meanB[idx] = mean_j + K @ nu
                    covB[idx] = P_u
                    eB[idx] = 1.0
                    aHis[idx] = np.append(aHis_j, t + 1)
                    quad = -0.5 * maha
                    base = np.log(eB_j * p_d) + quad - 0.5 * log_det_S - Nz * np.log(2.0 * np.pi) / 2.0
                    weightBLog[idx] = wlog_j + base
                    weightBLog_k[idx] = base
                # else: weights remain -inf.

        tracks[i] = {
            "meanB": meanB,
            "covB": covB,
            "eB": eB,
            "t_ini": pt["t_ini"],
            "aHis": aHis,
            "weightBLog": weightBLog,
            "weightBLog_k": weightBLog_k,
        }

    # ---- Update of global hypotheses.
    if Nprev == 0:
        flt["globHypWeight"] = np.array([1.0])
        flt["globHyp"] = np.zeros((1, Nmeas), dtype=int)  # all hypotheses index 0
        if Nmeas == 0:
            tracks = []
            flt["globHyp"] = np.zeros((1, 0), dtype=int)
        flt["tracks"] = tracks
        return flt

    glob_weight_log = []
    glob_hyp_rows = []

    for p in range(len(flt_pred["globHypWeight"])):
        cost_matrix = np.full((Nprev + Nmeas, Nmeas), -np.inf)
        cost_misdet = np.zeros(Nprev)

        for i in range(Nprev):
            index_hyp = flt_pred["globHyp"][p, i]
            Nhyp_i = len(flt_pred["tracks"][i]["eB"])
            if index_hyp != -1:
                wk = tracks[i]["weightBLog_k"]
                ref = wk[index_hyp]
                for t in range(Nmeas):
                    idx = index_hyp + Nhyp_i * (t + 1)
                    cost_matrix[i, t] = wk[idx] - ref
                cost_misdet[i] = ref

        for i in range(Nprev, Nprev + Nmeas):
            m = i - Nprev
            cost_matrix[i, m] = tracks[i]["weightBLog"][0]

        # Trim -inf rows/columns; columns with a single finite entry are fixed.
        finite = np.isfinite(cost_matrix)
        col_counts = finite.sum(axis=0)
        stay_columns = np.where(col_counts > 1)[0]
        fixed_columns = np.where(col_counts == 1)[0]

        if stay_columns.size > 0:
            stay_rows = np.where(finite[:, stay_columns].sum(axis=1) > 0)[0]
        else:
            stay_rows = np.array([], dtype=int)

        fixed_rows = np.array(
            [np.where(finite[:, c])[0][0] for c in fixed_columns], dtype=int
        )
        cost_fixed = float(
            sum(cost_matrix[fixed_rows[a], fixed_columns[a]] for a in range(fixed_columns.size))
        )

        glob_weight_log_pred = np.log(flt_pred["globHypWeight"][p])

        if stay_rows.size == 0 or stay_columns.size == 0:
            # No assignment ambiguity: a single resulting global hypothesis.
            opt_trans = -np.ones((1, Nmeas), dtype=int)
            for a in range(fixed_columns.size):
                opt_trans[0, fixed_columns[a]] = fixed_rows[a]
            glob_weight_log.append(cost_misdet.sum() + cost_fixed + glob_weight_log_pred)
        else:
            trimmed = cost_matrix[np.ix_(stay_rows, stay_columns)]
            kbest = int(np.ceil(Nhyp_max * flt_pred["globHypWeight"][p]))
            assigns, nlcosts = murty(-trimmed.T, kbest)
            n_sol = assigns.shape[0]

            opt_trans = -np.ones((n_sol, Nmeas), dtype=int)
            for s in range(n_sol):
                for pcol in range(stay_columns.size):
                    track_local = assigns[s, pcol]
                    opt_trans[s, stay_columns[pcol]] = stay_rows[track_local]
                for a in range(fixed_columns.size):
                    opt_trans[s, fixed_columns[a]] = fixed_rows[a]
                glob_weight_log.append(
                    -nlcosts[s] + cost_misdet.sum() + cost_fixed + glob_weight_log_pred
                )

        # Translate assignments into single-target hypothesis indices.
        n_sol = opt_trans.shape[0]
        glob_hyp_prov = np.zeros((n_sol, Nprev + Nmeas), dtype=int)

        for i in range(Nprev):
            index_hyp = flt_pred["globHyp"][p, i]
            Nhyp_i = len(flt_pred["tracks"][i]["eB"])
            for s in range(n_sol):
                assigned = np.where(opt_trans[s, :] == i)[0]
                if assigned.size == 0:
                    glob_hyp_prov[s, i] = index_hyp
                else:
                    t = assigned[0]
                    glob_hyp_prov[s, i] = index_hyp + Nhyp_i * (t + 1)

        for i in range(Nprev, Nprev + Nmeas):
            for s in range(n_sol):
                assigned = np.where(opt_trans[s, :] == i)[0]
                glob_hyp_prov[s, i] = 0 if assigned.size > 0 else -1

        glob_hyp_rows.append(glob_hyp_prov)

    flt["globHyp"] = np.vstack(glob_hyp_rows) if glob_hyp_rows else np.zeros((0, Nprev + Nmeas), dtype=int)
    gwl = np.asarray(glob_weight_log, dtype=float)
    gw = np.exp(gwl - np.max(gwl))
    flt["globHypWeight"] = gw / np.sum(gw)
    flt["tracks"] = tracks
    return flt


def prune(flt, T_pruning, T_pruningPois, Nhyp_max, existence_threshold):
    """Pruning / hypothesis reduction. Port of ``PoissonMBMtarget_pruning.m``."""
    out = {}

    # ---- Poisson pruning.
    wP = np.asarray(flt["weightPois"], dtype=float)
    keep_p = wP >= T_pruningPois
    out["weightPois"] = [flt["weightPois"][i] for i in range(len(keep_p)) if keep_p[i]]
    out["meanPois"] = [flt["meanPois"][i] for i in range(len(keep_p)) if keep_p[i]]
    out["covPois"] = [flt["covPois"][i] for i in range(len(keep_p)) if keep_p[i]]

    tracks = [dict(t) for t in flt["tracks"]]
    weights = np.asarray(flt["globHypWeight"], dtype=float)
    glob_hyp = np.asarray(flt["globHyp"], dtype=int)

    if weights.size == 0 or glob_hyp.size == 0:
        out["tracks"] = tracks
        out["globHyp"] = glob_hyp
        out["globHypWeight"] = weights
        return out

    # ---- Prune global hypotheses by weight threshold.
    pruned = weights > T_pruning
    weights_pruned = weights[pruned]
    glob_hyp_pruned = glob_hyp[pruned, :]

    # Cap the number of global hypotheses.
    if weights_pruned.size > Nhyp_max:
        order = np.argsort(weights_pruned)[::-1]
        sel = order[:Nhyp_max]
    else:
        sel = np.arange(weights_pruned.size)
    weights_pruned = weights_pruned[sel] / np.sum(weights_pruned[sel])
    glob_hyp_pruned = glob_hyp_pruned[sel, :]

    # ---- Remove tracks not present in any global hypothesis.
    present = np.any(glob_hyp_pruned != -1, axis=0)
    keep_tracks = np.where(present)[0]
    tracks = [tracks[i] for i in keep_tracks]
    glob_hyp_pruned = glob_hyp_pruned[:, keep_tracks]

    # ---- Remove single-target hypotheses not used by any global hypothesis,
    #      and reindex.
    for i in range(len(tracks)):
        Nhyp_i = len(tracks[i]["eB"])
        used = glob_hyp_pruned[:, i]
        used_valid = np.unique(used[used != -1])
        index_remove = np.ones(Nhyp_i, dtype=bool)
        index_remove[used_valid] = False

        if index_remove.any():
            keep_idx = np.where(~index_remove)[0]
            tracks[i]["meanB"] = [tracks[i]["meanB"][j] for j in keep_idx]
            tracks[i]["covB"] = [tracks[i]["covB"][j] for j in keep_idx]
            tracks[i]["eB"] = np.asarray(tracks[i]["eB"])[keep_idx]
            tracks[i]["aHis"] = [tracks[i]["aHis"][j] for j in keep_idx]
            tracks[i]["weightBLog"] = np.asarray(tracks[i]["weightBLog"])[keep_idx]

            removed_before = np.cumsum(index_remove)
            for r in range(glob_hyp_pruned.shape[0]):
                h = glob_hyp_pruned[r, i]
                if h != -1:
                    glob_hyp_pruned[r, i] = h - removed_before[h]

    # ---- Remove tracks whose existence is below threshold in all hypotheses.
    index_remove_tracks = [
        i for i in range(len(tracks))
        if np.all(np.asarray(tracks[i]["eB"]) < existence_threshold)
    ]
    if index_remove_tracks:
        keep = [i for i in range(len(tracks)) if i not in index_remove_tracks]
        tracks = [tracks[i] for i in keep]
        glob_hyp_pruned = glob_hyp_pruned[:, keep]

        # Merge duplicate global hypotheses produced by the track removal.
        uniq, inverse = np.unique(glob_hyp_pruned, axis=0, return_inverse=True)
        if uniq.shape[0] != glob_hyp_pruned.shape[0]:
            new_weights = np.zeros(uniq.shape[0])
            for idx in range(uniq.shape[0]):
                new_weights[idx] = np.sum(weights_pruned[inverse.ravel() == idx])
            glob_hyp_pruned = uniq
            weights_pruned = new_weights

    out["tracks"] = tracks
    out["globHyp"] = glob_hyp_pruned
    out["globHypWeight"] = weights_pruned / np.sum(weights_pruned)
    return out


def _has_hypotheses(flt):
    gh = np.asarray(flt["globHyp"])
    return gh.size > 0 and gh.shape[0] > 0 and len(flt["globHypWeight"]) > 0


def estimate1(flt, existence_estimation_threshold):
    """Estimator 1: highest-weight global hypothesis, threshold on existence.

    Port of ``PoissonMBMtarget_estimate1.m``.
    """
    parts = []
    if _has_hypotheses(flt):
        idx = int(np.argmax(flt["globHypWeight"]))
        hyp_max = flt["globHyp"][idx, :]
        for i in range(hyp_max.size):
            h = hyp_max[i]
            if h >= 0:
                e = flt["tracks"][i]["eB"][h]
                if e > existence_estimation_threshold:
                    parts.append(np.asarray(flt["tracks"][i]["meanB"][h]).ravel())
    return np.concatenate(parts) if parts else np.zeros(0)


def estimate2(flt):
    """Estimator 2: MAP cardinality, then best hypothesis of that cardinality.

    Port of ``PoissonMBMtarget_estimate2.m``.
    """
    if not _has_hypotheses(flt):
        return np.zeros(0)

    glob_hyp = flt["globHyp"]
    glob_w = np.asarray(flt["globHypWeight"], dtype=float)
    Nhyp = glob_w.size
    N_tracks = len(flt["tracks"])

    pcard_tot = np.zeros(N_tracks + 1)
    eB_tot = np.zeros((Nhyp, N_tracks))
    for j in range(Nhyp):
        eB_hyp = np.zeros(N_tracks)
        for i in range(N_tracks):
            h = glob_hyp[j, i]
            if h != -1:
                eB_hyp[i] = flt["tracks"][i]["eB"][h]
        eB_tot[j, :] = eB_hyp
        pcard_tot += glob_w[j] * cardinality_mb(eB_hyp)

    card_estimate = int(np.argmax(pcard_tot))
    if card_estimate == 0:
        return np.zeros(0)

    weight_hyp_card = np.zeros(Nhyp)
    indices_sort_hyp = np.zeros((Nhyp, N_tracks), dtype=int)
    for j in range(Nhyp):
        eB_hyp = eB_tot[j, :]
        order = np.argsort(eB_hyp)[::-1]
        indices_sort_hyp[j, :] = order
        eB_sorted = eB_hyp[order]
        vec = np.concatenate([eB_sorted[:card_estimate], 1.0 - eB_sorted[card_estimate:]])
        weight_hyp_card[j] = glob_w[j] * np.prod(vec)

    index_f = int(np.argmax(weight_hyp_card))
    order = indices_sort_hyp[index_f, :]
    hyp = glob_hyp[index_f, :]
    parts = []
    for i in range(card_estimate):
        target_i = order[i]
        h = hyp[target_i]
        parts.append(np.asarray(flt["tracks"][target_i]["meanB"][h]).ravel())
    return np.concatenate(parts) if parts else np.zeros(0)


def estimate3(flt):
    """Estimator 3: best fixed-cardinality (MBM01) hypothesis.

    Port of ``PoissonMBMtarget_estimate3.m``.
    """
    if not _has_hypotheses(flt):
        return np.zeros(0)

    glob_hyp = flt["globHyp"]
    glob_w = np.asarray(flt["globHypWeight"], dtype=float).copy()
    Nhyp = glob_w.size
    N_tracks = len(flt["tracks"])

    weight_tot = glob_w.copy()
    for j in range(Nhyp):
        for i in range(N_tracks):
            h = glob_hyp[j, i]
            if h != -1:
                e = flt["tracks"][i]["eB"][h]
                weight_tot[j] *= e if e > 0.5 else (1.0 - e)

    index_hyp = int(np.argmax(weight_tot))
    parts = []
    for i in range(N_tracks):
        h = glob_hyp[index_hyp, i]
        if h != -1:
            e = flt["tracks"][i]["eB"][h]
            if e > 0.5:
                parts.append(np.asarray(flt["tracks"][i]["meanB"][h]).ravel())
    return np.concatenate(parts) if parts else np.zeros(0)
