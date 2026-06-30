"""One-time precomputation of ``preClusterThreshold``.

Port of the "pure track" score recursion in ``script_pmbm91.m:629-688``. The
threshold is the absolute difference between the accumulated log-scores of a
consistently-detected vs. a consistently-misdetected track over
``allowedCountMisdet + 2`` steps. Used only by ``clusteringPreprocess`` once
clusters exist.
"""

import numpy as np
from scipy.linalg import block_diag

from .mathutils import normpdf_log

ALLOWED_COUNT_MISDET = 3  # script_pmbm91.m:629


def compute_pre_cluster_threshold(system, params, pD, allowed_count_misdet=ALLOWED_COUNT_MISDET):
    F = np.asarray(system["fMat"], dtype=float)
    Q = np.asarray(system["qMat"], dtype=float)
    H = np.asarray(system["hMat"], dtype=float)
    rCart = np.asarray(system["rCart"], dtype=float)
    p_init_vel = np.asarray(params["pInitVel"], dtype=float)
    dim_tar = F.shape[0]

    nKPT = allowed_count_misdet + 2
    p_init = block_diag(rCart, p_init_vel)

    pPTDet = np.zeros((dim_tar, dim_tar, nKPT))
    pPTMisdet = np.zeros((dim_tar, dim_tar, nKPT))
    pPTDet[:, :, 0] = p_init
    pPTMisdet[:, :, 0] = p_init

    pPTPred = F @ pPTDet[:, :, 0] @ F.T + Q
    sMat = H @ pPTPred @ H.T + rCart
    K = pPTPred @ H.T @ np.linalg.inv(sMat)
    pPTDet[:, :, 1] = (np.eye(dim_tar) - K @ H) @ pPTPred
    pPTMisdet[:, :, 0] = pPTDet[:, :, 1]

    exi_misdet = np.zeros(nKPT)
    exi_misdet[0:2] = 1.0

    score_det = np.zeros(nKPT)
    score_misdet = np.zeros(nKPT)

    area = float(params["areaCircle"])
    lambda_fa = float(params["faRate"]) / area
    mu_birth = float(np.asarray(params["birthRateHistory"]).ravel()[0]) / area
    pred_exi = float(params["pS"])

    z0 = np.zeros((2, 1))
    base = -np.log(lambda_fa + pD * mu_birth) + normpdf_log(z0, z0, sMat)[0] \
        + np.log(pD) + np.log(pred_exi)
    score_det[1] = base
    score_misdet[1] = base

    for kPT in range(2, nKPT):  # 0-based index for MATLAB kPT=3..nKPT
        # With detection
        pPTPred = F @ pPTDet[:, :, kPT - 1] @ F.T + Q
        sMat = H @ pPTPred @ H.T + rCart
        K = pPTPred @ H.T @ np.linalg.inv(sMat)
        pPTDet[:, :, kPT] = (np.eye(dim_tar) - K @ H) @ pPTPred
        pred_exi = float(params["pS"])
        score_det[kPT] = score_det[kPT - 1] - np.log(lambda_fa + pD * mu_birth) \
            + normpdf_log(z0, z0, sMat)[0] + np.log(pD) + np.log(pred_exi)

        # Without detection
        pPTPred = F @ pPTMisdet[:, :, kPT - 1] @ F.T + Q
        pPTMisdet[:, :, kPT] = pPTPred
        pred_exi = exi_misdet[kPT - 1] * float(params["pS"])
        exi_misdet[kPT] = pred_exi
        score_misdet[kPT] = score_misdet[kPT - 1] \
            + np.log(1 - pred_exi + pred_exi * (1 - pD))

    return abs(score_det[-1] - score_misdet[-1])
