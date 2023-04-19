import numpy as np
import py_dfg_da as pdd


if __name__ == "__main__":
    R = np.array([
        [    3.0,   -0.60, -np.inf],
        [    3.2, -np.inf,   -0.56]
    ], order='F')

    n, mpn = R.shape
    m = mpn - n

    R_LC = np.hstack((np.diag(R[:, m:])[:,None], R[:,:m]))

    prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([1], np.log(0.5)),
            # pdd.hypothesis.Hypothesis([1, 2], np.log(1.0)),
            pdd.hypothesis.Hypothesis([2], np.log(0.5)),
        ]),
        # pdd.hypothesis.Hypotheses([
        #     pdd.hypothesis.Hypothesis([2], np.log(1.0))
        # ])
    ])

    mhlbp = pdd.lbp.lbp_multicluster(R, prior_hypotheses_per_cluster)

    # prior_hypotheses_posterior = prior_hypotheses_per_cluster[0].combine(prior_hypotheses_per_cluster[1])
    prior_hypotheses_posterior = prior_hypotheses_per_cluster[0]
    for ph in prior_hypotheses_posterior:
        print(ph.tracks())
        print(ph.probability())

    np.set_printoptions(suppress=True)

    exact_margs, exact_norm = pdd.hypothesis.association_marginal_posteriors_normalization_constant(R, prior_hypotheses_posterior)

    print(mhlbp.track_association_marginals().T)
    print(mhlbp.bethe_pseudodual_normalization_constant())

    print(exact_margs.T)
    print(exact_norm)