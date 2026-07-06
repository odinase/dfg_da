from dfg_da.cluster_bayes_tree_extra_term import test_case, edmund_to_lc, MulticlusterEfficientMarginals
from dfg_da.cluster_bayes_tree_extra_term2 import MulticlusterEfficientMarginals2
import numpy as np


if __name__ == "__main__":
    R, prior_hypotheses_per_cluster, assocLocal = test_case()
    R_LC = edmund_to_lc(R)

    marginal_computer_exact = MulticlusterEfficientMarginals(R_LC, prior_hypotheses_per_cluster, assocLocal)

    marginal_computer_exact2 = MulticlusterEfficientMarginals2(R_LC, prior_hypotheses_per_cluster, assocLocal)

    exact_marginals, exact_likelihood = marginal_computer_exact.compute_marginals_likelihood()

    with np.printoptions(suppress=True, precision=7, linewidth=180):
        print(exact_marginals)
        print(exact_likelihood)


    exact_marginals2, exact_likelihood2 = marginal_computer_exact2.compute_marginals_likelihood()

    with np.printoptions(suppress=True, precision=7, linewidth=180):
        print(exact_marginals2)
        print(exact_likelihood2)