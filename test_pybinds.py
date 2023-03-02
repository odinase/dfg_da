import py_dfg_da
import numpy as np
import dfg_da.marginal_association_Odin as ma
import dfg_da.marginals_computers as mc

if __name__ == "__main__":
    R = np.array([
        [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
        [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
        [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
        [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
        [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    ], order='F')

    # R = np.array([
    #     [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
    #     [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
    #     [   -3.0, -np.inf, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
    #     [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
    #     [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    # ], order='F')

    n, mpn = R.shape
    m = mpn - n

    R_LC = np.hstack((np.diag(R[:, m:])[:,None], R[:,:m]))

    prior_hypotheses_per_cluster: py_dfg_da.hypothesis.HypothesesList = py_dfg_da.hypothesis.HypothesesList([
        py_dfg_da.hypothesis.Hypotheses([
            py_dfg_da.hypothesis.Hypothesis([1, 2], np.log(0.5)),
            py_dfg_da.hypothesis.Hypothesis([1, 3], np.log(0.5))
        ]),
        py_dfg_da.hypothesis.Hypotheses([
            py_dfg_da.hypothesis.Hypothesis([4], np.log(0.5)),
            py_dfg_da.hypothesis.Hypothesis([5], np.log(0.5))
        ])
    ])

    output = py_dfg_da.lbp.lbp_multicluster(R, prior_hypotheses_per_cluster)
    np.set_printoptions(suppress=True)
    print(output.track_association_marginals().T)
    print(output.bethe_pseudodual_normalization_constant())
    exact_marginals, exact_normalization_constant = py_dfg_da.factor_graph.exact_marginals_and_normalization_constant(R, prior_hypotheses_per_cluster)
    print(exact_marginals.T)
    print(exact_normalization_constant)
    print(dir(prior_hypotheses_per_cluster))

    merged_hypos = prior_hypotheses_per_cluster[0].combine(prior_hypotheses_per_cluster[1])
    exact_margs, exact_norm = py_dfg_da.hypothesis.association_marginal_posteriors_normalization_constant(R, merged_hypos)

    print(exact_margs.T)
    print(exact_norm)

    # print("Multicluster!!!")
    # exact_margs_mc, exact_norm_mc = py_dfg_da.hypothesis.association_marginal_posteriors_normalization_constant_multicluster(R, prior_hypotheses_per_cluster)
    # print(exact_margs_mc.T)
    # print(exact_norm_mc)


    # lc_margs = np.empty((n, m + 2))
    # _0 = np.zeros((n, 1))
    # all_tracks_idx = np.arange(n)
    # Z = np.empty(len(prior_hypotheses_per_cluster))
    # for c, prior_hypotheses in enumerate(prior_hypotheses_per_cluster):
    #     t_idx = np.fromiter(prior_hypotheses.tracks(), dtype=int) - 1
    #     nc = len(t_idx)
    #     conditioned_marginals = np.zeros((n, m + 2))
    #     marginal_total = np.zeros((n, m + 1 + 1))
    #     for k in range(len(prior_hypotheses)):
    #         hh = prior_hypotheses[k]
    #         tracks = np.array(hh.tracks())
    #         hypo_prob = hh.probability()
    #         R_sub = R_LC[tracks-1, :]
    #         JPDAprobs, _, loglikelihood = ma.exact_marginal(R_sub, False)

    #         # We need to concatenate the JPDAprobs with all tracks and existence probs
    #         existing_tracks_idx = tracks - 1
    #         non_existing_tracks_idx = np.delete(all_tracks_idx, existing_tracks_idx)

    #         existing_probs = np.hstack((JPDAprobs, _0[:len(tracks)]))
    #         nonexisting_probs = np.hstack((np.zeros((len(non_existing_tracks_idx), m + 1)), np.ones((len(non_existing_tracks_idx), 1))))

    #         conditioned_marginals[existing_tracks_idx] = existing_probs
    #         conditioned_marginals[non_existing_tracks_idx] =  nonexisting_probs

    #         normalizing_constant = np.exp(loglikelihood)

    #         marginal_total += conditioned_marginals * normalizing_constant * hypo_prob

    #     Z_cluster = marginal_total.sum(axis=1, keepdims=True)
    #     Z[c] = Z_cluster.ravel()[0]
    #     marginal_total = marginal_total / Z_cluster
    #     lc_margs[t_idx] = marginal_total[t_idx]

    # print(f"\nLC margs")
    # print(lc_margs)
    # print(Z)
    # print(np.prod(Z))
    # print(np.prod(Z) / exact_norm)
    # print(exact_norm / Z)

        # exact_marginals(Rcluster, hh)


