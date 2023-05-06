import numpy as np
from dfg_da.marginal_association_Odin import exact_marginal
from dfg_da.cluster_bayes_tree import conditioned_reward_matrix_bversion, R_LC_to_validation_likelihood_matrix, b_probs_likelihood_to_a_probs_likelihood, edmund_to_lc, lc_to_edmund
import dfg_da.marginals_computers as mc
from pyehm.core import EHM2



def conditioned_exact(R_LC, meas_existence_mapping):
    meas_idxs, mask = meas_existence_mapping.T
    R_conditioned_bversion = conditioned_reward_matrix_bversion(R_LC, meas_existence_mapping)
    forced_meas = meas_idxs[mask==1]
    R_sub_exact = R_conditioned_bversion[forced_meas]
    b_forced_probs, (hypProb, JPDA_hyp_mat), loglikelihood = exact_marginal(R_sub_exact, False)
    R_LC_temp = np.empty_like(R_LC)
    for hyp, prob in zip(hypProb, JPDA_hyp_mat.T):
        R_LC_temp[...] = R_LC
        # Should now be able to loop over each hypothesis we used to construct
        # Hyp should be what track a forced measurement is associated with, delete them from R_LC
        R_LC_temp[hyp] = -np.inf
        R_LC_temp[:, forced_meas] = -np.inf
        R_LC[np.ix_(hyp, forced_meas)] = 0.0
        

if __name__ == "__main__":
    R_LC = np.array([
        [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.1,     1.0,     1.0, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.1, -np.inf,     1.0, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.1, -np.inf, -np.inf,     1.0, -np.inf, -np.inf, -np.inf],
        [0.1, -np.inf, -np.inf,     1.0,     1.0, -np.inf, -np.inf],
        [0.1, -np.inf, -np.inf, -np.inf,     1.0, -np.inf, -np.inf],
        [0.1, -np.inf, -np.inf,     1.0,     1.0,     1.0, -np.inf],
        [0.1, -np.inf, -np.inf, -np.inf, -np.inf,     1.0, -np.inf],
        [0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.1,     1.0, -np.inf, -np.inf, -np.inf,     1.0, -np.inf],
        [0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,     1.0],
    ], order='F')

    n, mp1 = R_LC.shape
    m = mp1 - 1

    np.set_printoptions(suppress=False, linewidth=150)
    meas_existence_mapping = np.empty((m, 2), dtype=np.int64)
    meas_existence_mapping[:, 0] = np.arange(m) + 1
    meas_existence_mapping[:, 1] = 0
    meas_existence_mapping[::2, 1] = 1
    print(meas_existence_mapping)

    R_conditioned_bversion = conditioned_reward_matrix_bversion(R_LC, meas_existence_mapping)
    validation_matrix, likelihood_matrix = R_LC_to_validation_likelihood_matrix(R_conditioned_bversion)
    meas_is_gated = validation_matrix.any(axis=1)
    b_probs, b_likelihood = EHM2.run_and_likelihood(validation_matrix, likelihood_matrix)
    JPDAprobs, likelihood = b_probs_likelihood_to_a_probs_likelihood(b_probs, b_likelihood, R_LC)

    print(b_probs)

    conditioned_exact(R_LC, meas_existence_mapping)