from pyehm.core import EHM2
from dfg_da.marginal_association_Odin import exact_marginal
import numpy as np
from cluster_data_asso import edmund_to_lc, lc_to_edmund


if __name__ == "__main__":
    R = np.array([
        [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
        [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
        [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
        [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
        [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    ], order='F')

    R_LC = edmund_to_lc(R)
    likelihood_matrix = np.asfortranarray(np.exp(R_LC))
    validation_matrix = np.asfortranarray((likelihood_matrix > 0.0).astype(int))

    assoc_matrix_ehm2 = EHM2.run(validation_matrix, likelihood_matrix)
    JPDAprobs, hyp_prob_log, loglikelihood = exact_marginal(R_LC, False)

    np.set_printoptions(suppress=True)
    print(assoc_matrix_ehm2)
    
    print(JPDAprobs)
    print(np.exp(loglikelihood))
