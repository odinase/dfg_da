import py_dfg_da
import py_dfg_da as pdd
import numpy as np
import dfg_da.marginal_association_Odin as ma
import dfg_da.marginals_computers as mc
import pickle
import dfg_da as dd
from typing import List, Optional
import matplotlib.pyplot as plt


def make_clusters(llr):
        g = (llr > -np.inf)
        pr_a = np.zeros_like(llr)
        pr_new = np.zeros(llr.shape[1] - 1, float)
        usedtr = np.zeros(llr.shape[0], bool)
        usedm = np.zeros(llr.shape[1] - 1, bool)
        gm = g[:, 1:]
        c = []
        for i in range(len(llr)):
            if not usedtr[i]:
                ctr = np.zeros_like(usedtr)
                ca = np.zeros(llr.shape[1], bool)
                ca[0] = True
                cm = ca[1:]
                ctr[i] = True
                m_in_same = gm[i]
                cm[m_in_same] = True
                changed = True
                while changed:
                    tr_in_same = np.any(gm[:, cm], axis=1)
                    changed &= ~np.all(ctr[tr_in_same])
                    ctr[tr_in_same] = True
                    m_in_same = np.any(gm[tr_in_same, :], axis=0)
                    changed &= ~np.all(cm[m_in_same])
                    cm[m_in_same] = True

                print(ctr, ca)
        #         pr_a[np.ix_(ctr, ca)], pr_new[cm] = exact_marginal(
        #             llr[np.ix_(ctr, ca)], False)
        #         usedtr[ctr] = True
        #         usedm[cm] = True
        # pr_new[~usedm] = 1
        # return pr_a, pr_new


def theta_posterior_correlation(true_posteriors: List[np.ndarray], lbp_posteriors: List[np.ndarray], ax: Optional[plt.Axes] = None):
    if ax is None:
        fig, ax = plt.subplots()

    true_posteriors = np.hstack(true_posteriors)
    lbp_posteriors = np.hstack(lbp_posteriors)

    ax.plot(lbp_posteriors, true_posteriors, 'bx')



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

    assocLocal = np.array([
        [1, 1],
        [1, 0]
    ])

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

    # output = py_dfg_da.lbp.lbp_multicluster(R, prior_hypotheses_per_cluster)

    exact_computer: mc.MulticlusterExact = mc.MulticlusterExact()
    exact_output: mc.MulticlusterExactOutput = exact_computer(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal)
    np.set_printoptions(suppress=True)
    print(exact_output.exact_marginals)
    print(exact_output.compute_theta_posteriors())


    # .def("track_association_marginals",  &lbp::MHLBPMulticlusterOutput::track_association_marginals)
    # .def("measurement_association_marginals",  &lbp::MHLBPMulticlusterOutput::measurement_association_marginals)
    # .def("hypotheses_marginals",  &lbp::MHLBPMulticlusterOutput::hypotheses_marginals)
    # .def("bethe_pseudodual_loglikelihood",  &lbp::MHLBPMulticlusterOutput::bethe_pseudodual_loglikelihood)
    # .def("bethe_pseudodual_normalization_constant",  &lbp::MHLBPMulticlusterOutput::bethe_pseudodual_normalization_constant)


    mcmhlbp: pdd.lbp.MHLBPMulticlusterOutput = pdd.lbp.lbp_multicluster(R, prior_hypotheses_per_cluster)
    print(mcmhlbp.track_association_marginals().T)
    hp = mcmhlbp.hypotheses_marginals()
    print(np.hstack(hp))

    print()

    track_marginals, meas_marginals, theta_marginals, exact_normalization_constant = pdd.factor_graph.all_exact_marginals_and_normalization_constant(R, prior_hypotheses_per_cluster)
    print(track_marginals)
    print(meas_marginals)
    print(theta_marginals)
    print(exact_normalization_constant)

    fig, ax = plt.subplots()

    theta_posterior_correlation(exact_output.compute_theta_posteriors(), mcmhlbp.hypotheses_marginals(), ax=ax)
    plt.show()

    # R_LC2 = np.log(np.array([
    #     [0.2, 1.0]
    # ]))

    # phs: py_dfg_da.hypothesis.HypothesesList = py_dfg_da.hypothesis.HypothesesList([
    #     py_dfg_da.hypothesis.Hypotheses([
    #         py_dfg_da.hypothesis.Hypothesis([1], np.log(0.9)),
    #         py_dfg_da.hypothesis.Hypothesis([], np.log(0.1))
    #     ])
    # ])

    # assocLocal = np.array([
    #      [1],
    #      [1]
    # ])

    # exact_output: mc.MulticlusterExactOutput = exact_computer(R_LC2, phs, assocLocal=assocLocal)
    # np.set_printoptions(suppress=True)
    # print(exact_output.exact_marginals)
    # consts = exact_output.hypo_cond_normalization_constants_per_cluster[0]
    # print(consts / consts.sum())

    # print(exact_output.compute_theta_posteriors())

    # for c, ph in prior_hypotheses_per_cluster:
    #     for k, h in enumerate(ph):
    #         pass
    #         # At this point we need to find all cluster 

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

    # for ph in prior_hypotheses_per_cluster:
    #     for h in ph:
    #         print(h.tracks())
    #         print(h.probability())

    # path = "./test_cpp_pickle"
    # with open(path, "wb") as f:
    #     pickle.dump(prior_hypotheses_per_cluster, f)

    # with open(path, "rb") as f:
    #     prior_hypotheses_per_cluster_from_pickle = pickle.load(f)
    # print("Pickle")
    # for ph in prior_hypotheses_per_cluster_from_pickle:
    #     for h in ph:
    #         print(h.tracks())
    #         print(h.probability())


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
