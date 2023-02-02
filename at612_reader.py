from scipy.io import loadmat
from dfg_da.marginal_association_Odin import exact_marginal, lbp_marginal_nonexistence, logsumexp, lbp_marginal
import dfg_da.stats_logger as sl
import numpy as np

import matplotlib.pyplot as plt

DATA_PATH = "./data/at612"
MAT_FILE = "priorLikelihood612.mat"

if __name__ == "__main__":
    parser = sl.MatFileParser("./data/at612/priorLikelihood612.mat")
    hypos_in_clusters = parser.prior_hypotheses_per_cluster

    for k, hypos in enumerate(hypos_in_clusters[:1]):
        print(f"Cluster {k+1}")
        for i, (t, p) in enumerate(hypos):
            print(f"Hypothesis {i+1}: {p} {t}\n")

    print(parser.reward_matrix_edmund[:3, :3])