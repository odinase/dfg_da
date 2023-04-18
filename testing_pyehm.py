from pyehm.core import EHM2
from dfg_da.marginal_association_Odin import exact_marginal
import dfg_da.marginals_computers as mc
import numpy as np
from cluster_data_asso import edmund_to_lc, lc_to_edmund
import py_dfg_da


if __name__ == "__main__":
    R = np.array([
        [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
        [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
        [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
        [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
        [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    ], order='F')

    prior_hypotheses_per_cluster: py_dfg_da.hypothesis.HypothesesList = py_dfg_da.hypothesis.HypothesesList([
        py_dfg_da.hypothesis.Hypotheses([
            py_dfg_da.hypothesis.Hypothesis([1, 2], np.log(0.5)),
            py_dfg_da.hypothesis.Hypothesis([1, 3], np.log(0.5))
        ]),
        py_dfg_da.hypothesis.Hypotheses([
            py_dfg_da.hypothesis.Hypothesis([4], np.log(0.5)),
            py_dfg_da.hypothesis.Hypothesis([5], np.log(0.5)),
        ])
    ])

    assocLocal = np.array([
        [1, 1],
        [1, 0]
    ])

    R_LC = edmund_to_lc(R)
    likelihood_matrix = np.asfortranarray(np.exp(R_LC))
    validation_matrix = np.asfortranarray((likelihood_matrix > 0.0).astype(np.int32))

    EHM2_margs, EHM2_likelihood = EHM2.run_and_likelihood(validation_matrix, likelihood_matrix)
    JPDAprobs, hyp_prob_log, loglikelihood = exact_marginal(R_LC, False)

    np.set_printoptions(suppress=True)
    print(EHM2_margs)
    print(EHM2_likelihood)
    
    print(JPDAprobs)
    print(np.exp(loglikelihood))

    naive = mc.MulticlusterExact()
    ehm2 = mc.MulticlusterExactEHM2()
    out_ehm2: mc.MulticlusterExactOutput = ehm2(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal.copy())
    out_exact: mc.MulticlusterExactOutput = naive(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal.copy())

    print(out_ehm2.exact_marginals)
    print(out_ehm2.compute_theta_posteriors())

    print(out_exact.exact_marginals)
    print(out_exact.compute_theta_posteriors())
