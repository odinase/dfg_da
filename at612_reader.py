from scipy.io import loadmat
from dfg_da.marginal_association_Odin import exact_marginal, lbp_marginal_nonexistence, logsumexp, lbp_marginal
import dfg_da.stats_logger as sl
import dfg_da.marginals_computers as mc
import numpy as np

import matplotlib.pyplot as plt

DATA_PATH = "./data/at612"
MAT_FILE = "priorLikelihood612.mat"

if __name__ == "__main__":
    parser = sl.MatFileParser("./data/at612/priorLikelihood612.mat")
    hypos_in_clusters = parser.prior_hypotheses_per_cluster

    mhlbp = mc.LBPMarginalsFullAssociation()
    R_LC = parser.reward_matrix_lc

    for k, hypos in enumerate([hypos_in_clusters[6]]):
        # return (asso_prob, (it, msg_it, converged)) + extra
        out = mhlbp(R_LC, hypos)
        tracks_in_cluster = frozenset(tt for t,_ in hypos for tt in t)
        t_idx = np.sort(np.fromiter(tracks_in_cluster, dtype=int)) - 1
        print(t_idx)
        np.set_printoptions(precision=6, suppress=True, linewidth=150)
        print(out[0][t_idx])