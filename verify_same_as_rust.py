from dfg_da.cluster_bayes_tree_extra_term import (
    test_case,
    test_case2,
    edmund_to_lc,
    MulticlusterEfficientMarginals,
)
from dfg_da.cluster_bayes_tree_extra_term2 import MulticlusterEfficientMarginals2
from cluster_partition.partitioning import MulticlusterPartitionedMarginals
import numpy as np
from copy import deepcopy
import py_dfg_da as pdd
from dfg_da.marginals_computers import LBPMarginalsByTotalProbBethe

if __name__ == "__main__":
    R_LC = np.array([
        [0.0, 1.5, 0.3],
        [0.0, 0.2, 1.1],
    ])

    prior_hypotheses = pdd.hypothesis.Hypotheses([
        pdd.hypothesis.Hypothesis([1, 2], np.log(0.7)),
        pdd.hypothesis.Hypothesis([1], np.log(0.3))
    ])

    # R_LC = edmund_to_lc(R)

    computer = LBPMarginalsByTotalProbBethe()

    lbp_marginals, lbp_theta_posterior, lbp_likelihood = computer.compute_marginals(R_LC, prior_hypotheses)

    with np.printoptions(suppress=True, precision=7, linewidth=180):
        print(lbp_marginals)
        print(lbp_theta_posterior)
        print(lbp_likelihood)
