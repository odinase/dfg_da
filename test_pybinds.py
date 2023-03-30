import py_dfg_da
import py_dfg_da as pdd
import numpy as np
import dfg_da.marginal_association_Odin as ma
import dfg_da.marginals_computers as mc
import pickle

if __name__ == "__main__":
    R = np.array([
        [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
        [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
        [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
        [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
        [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    ], order='F')

    # R = np.array([
    #     [    3.0,     2.9,   -40.60, -np.inf, -np.inf, -np.inf, -np.inf],
    #     [    3.2,     2.5, -np.inf,   -40.56, -np.inf, -np.inf, -np.inf],
    #     [    3.0,     3.2, -np.inf, -np.inf,   -30.46, -np.inf, -np.inf],
    #     [    3.2,     3.0, -np.inf, -np.inf, -np.inf,   -50.62, -np.inf],
    #     [    3.1,     2.4, -np.inf, -np.inf, -np.inf, -np.inf,   -40.55],
    # ], order='F')

    # R = np.array([
    #     [    3.0,     2.9,     -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
    #     [    3.2,     2.5, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
    #     [    3.0,     3.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
    #     [    3.2,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
    #     [    3.1,     2.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    # ], order='F')

    # R = np.array([
    #     [    3.0, -np.inf,   -20.60, -np.inf, -np.inf, -np.inf, -np.inf],
    #     [    3.0, -np.inf, -np.inf,   -20.60, -np.inf, -np.inf, -np.inf],
    #     [   -3.0,     1.2, -np.inf, -np.inf,   -20.46, -np.inf, -np.inf],
    #     [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -25.62, -np.inf],
    #     [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -25.55],
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
            py_dfg_da.hypothesis.Hypothesis([5], np.log(0.5)),
        ])
    ])

    output = py_dfg_da.lbp.lbp_multicluster(R, prior_hypotheses_per_cluster)
    # np.set_printoptions(suppress=True)
    # print("LBP")
    # print(output.track_association_marginals().T)
    # print(output.bethe_pseudodual_normalization_constant())
    # print(output.num_iters)

    # path = "./test_cpp_pickle"
    # with open(path, "wb") as f:
    #     pickle.dump(output, f)

    # with open(path, "rb") as f:
    #     mhlbp_from_pickle = pickle.load(f)

    # print("LBP from pickle")
    # print(mhlbp_from_pickle.track_association_marginals().T)
    # print(mhlbp_from_pickle.bethe_pseudodual_loglikelihood())
    # print(mhlbp_from_pickle.bethe_pseudodual_normalization_constant())
    # print(mhlbp_from_pickle.num_iters)

    for ph in prior_hypotheses_per_cluster:
        for h in ph:
            print(h.tracks())
            print(h.probability())

    path = "./test_cpp_pickle"
    with open(path, "wb") as f:
        pickle.dump(prior_hypotheses_per_cluster, f)

    with open(path, "rb") as f:
        prior_hypotheses_per_cluster_from_pickle = pickle.load(f)
    print("Pickle")
    for ph in prior_hypotheses_per_cluster_from_pickle:
        for h in ph:
            print(h.tracks())
            print(h.probability())


    # merged_hypos = prior_hypotheses_per_cluster[0].combine(prior_hypotheses_per_cluster[1])
    # exact_margs, exact_norm = py_dfg_da.hypothesis.association_marginal_posteriors_normalization_constant(R, merged_hypos)

    # margs, const = py_dfg_da.factor_graph.exact_marginals_and_normalization_constant(R, prior_hypotheses_per_cluster)
    # # print(margs.T)
    # # print(const)

    # # print("\nExact")
    # print(exact_margs.T)
    # print(exact_norm)


    # R = np.array([
    #     [    3.0, -np.inf,  -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
    #     [    3.2, -np.inf,  -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
    #     [   -3.0,     1.2,  -np.inf, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf, -np.inf, -np.inf],
    #     [-np.inf, -np.inf,      1.7, -np.inf, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf, -np.inf],
    #     [-np.inf, -np.inf,      2.3, -np.inf, -np.inf, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
    #     [-np.inf,     3.0,  -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
    #     [-np.inf,    -0.4,  -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    # ], order='F')


    # prior_hypotheses_per_cluster: py_dfg_da.hypothesis.HypothesesList = py_dfg_da.hypothesis.HypothesesList([
    #     py_dfg_da.hypothesis.Hypotheses([
    #         py_dfg_da.hypothesis.Hypothesis([1, 2], np.log(0.5)),
    #         py_dfg_da.hypothesis.Hypothesis([1, 3], np.log(0.5))
    #     ]),
    #     py_dfg_da.hypothesis.Hypotheses([
    #         py_dfg_da.hypothesis.Hypothesis([4], np.log(0.5)),
    #         py_dfg_da.hypothesis.Hypothesis([5], np.log(0.5)),
    #     ]),
    #     py_dfg_da.hypothesis.Hypotheses([
    #         py_dfg_da.hypothesis.Hypothesis([6], np.log(0.2)),
    #         py_dfg_da.hypothesis.Hypothesis([7], np.log(0.8)),
    #     ]),
    # ])

    # margs, const = py_dfg_da.factor_graph.exact_marginals_and_normalization_constant(R, prior_hypotheses_per_cluster)
    # print("\nExact")
    # print(margs.T)
    # print(const)


    # R = np.array([
    #     [    3.0, -np.inf,  -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
    #     [    3.2, -np.inf,  -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
    #     [   -3.0,     1.2,  -np.inf, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf, -np.inf, -np.inf],
    #     [-np.inf,     3.0,  -np.inf, -np.inf, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf, -np.inf],
    #     [-np.inf,    -0.4,  -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
    #     [-np.inf, -np.inf,      1.7, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
    #     [-np.inf, -np.inf,      2.3, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    # ], order='F')

    # prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
    #     pdd.hypothesis.Hypotheses([
    #         pdd.hypothesis.Hypothesis([1, 2], np.log(0.2)),
    #         pdd.hypothesis.Hypothesis([1, 3], np.log(0.8))
    #     ]),
    #     pdd.hypothesis.Hypotheses([
    #         pdd.hypothesis.Hypothesis([4], np.log(0.3)),
    #         pdd.hypothesis.Hypothesis([5], np.log(0.7))
    #     ]),
    #     pdd.hypothesis.Hypotheses([
    #         pdd.hypothesis.Hypothesis([6], np.log(0.5)),
    #         pdd.hypothesis.Hypothesis([7], np.log(0.5))
    #     ])
    # ])

    # margs, const = py_dfg_da.factor_graph.exact_marginals_and_normalization_constant(R, prior_hypotheses_per_cluster)
    # print("\nExact")
    # print(margs.T)
    # print(const)
