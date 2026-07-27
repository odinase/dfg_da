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
from dfg_da.cluster_conditioning_lbp_ie import MulticlusterEfficientMarginalsLBPInclusionExclusion


def test_case3():
    R = np.array([
        [    3.0, -np.inf, -np.inf, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
        [    3.2, -np.inf, -np.inf, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
        [   -3.0,     2.0,     1.2, -np.inf, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf, -np.inf, -np.inf],
        [-np.inf, -np.inf,     3.0, -np.inf, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf, -np.inf, -np.inf],
        [-np.inf,    -0.4,    -1.8, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55, -np.inf, -np.inf],
        [-np.inf,     0.5, -np.inf,    -0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
        [-np.inf, -np.inf, -np.inf,     0.8, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    ], order='F')

    prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([1, 2], np.log(0.5)),
            pdd.hypothesis.Hypothesis([1, 3], np.log(0.5))
        ]),
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([4], np.log(0.5)),
            pdd.hypothesis.Hypothesis([5], np.log(0.5))
        ]),
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([6, 7], np.log(0.2)),
            pdd.hypothesis.Hypothesis([6],    np.log(0.8))
        ])
    ])

    assocLocal = np.array([
        [1, 1, 1],
        [1, 0, 0]
    ])

    return R, prior_hypotheses_per_cluster, assocLocal


if __name__ == "__main__":
    R, prior_hypotheses_per_cluster, assocLocal = test_case3()
    R_LC = edmund_to_lc(R)

    mc_lbp_ie = MulticlusterEfficientMarginalsLBPInclusionExclusion(R_LC=R_LC, prior_hypotheses_per_cluster=deepcopy(prior_hypotheses_per_cluster), assocLocal=assocLocal.copy(), lbp_solver=LBPMarginalsByTotalProbBethe())

    computers = [
        mc_lbp_ie
        # MulticlusterEfficientMarginals(
        #     deepcopy(R_LC), deepcopy(prior_hypotheses_per_cluster), deepcopy(assocLocal)
        # ),
        # MulticlusterEfficientMarginals2(
        #     deepcopy(R_LC), deepcopy(prior_hypotheses_per_cluster), deepcopy(assocLocal)
        # ),
        # MulticlusterPartitionedMarginals(
        #     R_LC=deepcopy(R_LC),
        #     prior_hypotheses_per_cluster=deepcopy(prior_hypotheses_per_cluster),
        #     assocLocal=deepcopy(assocLocal),
        # ),
    ]

    for computer in computers:
        out = computer.compute_marginals_likelihood()
        # @dataclass
        # class MulticlusterConditionendLBPOutput:
        #     marginals: np.ndarray
        #     likelihood: float
        #     theta_posteriors: Optional[Dict[int, np.ndarray]] = None
        #     raised_warning: bool = False


        exact_marginals, exact_likelihood = out.marginals, out.likelihood

        with np.printoptions(suppress=True, precision=7, linewidth=180):
            print(exact_marginals)
            print(exact_likelihood)

    # exact_marginals2, exact_likelihood2 = marginal_computer_exact2.compute_marginals_likelihood()

    # with np.printoptions(suppress=True, precision=7, linewidth=180):
    #     print(exact_marginals2)
    #     print(exact_likelihood2)

    # exact_marginals2, exact_likelihood2 = marginal_computer_exact2.compute_marginals_likelihood()

    # with np.printoptions(suppress=True, precision=7, linewidth=180):
    #     print(exact_marginals2)
    #     print(exact_likelihood2)
