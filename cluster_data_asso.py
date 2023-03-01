import py_dfg_da as pdd
import numpy as np


def cluster_reward_matrix(R, tracks_in_cluster):
    n, mpn = R.shape
    m = mpn - n
    Rd = R[:, :m]
    Rmd = R[:, m:]
    Rd = Rd[tracks_in_cluster - 1]
    Rmd = Rmd[np.ix_(tracks_in_cluster - 1, tracks_in_cluster - 1)]
    return np.hstack((Rd, Rmd))


if __name__ == "__main__":
    R = np.array([
        [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
        [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
        [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
        [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
        [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    ], order='F')

    n, mpn = R.shape
    m = mpn - n

    R_LC = np.hstack((np.diag(R[:, m:])[:,None], R[:,:m]))

    prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([1, 2], np.log(0.5)),
            pdd.hypothesis.Hypothesis([1, 3], np.log(0.5))
        ]),
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([4], np.log(0.5)),
            pdd.hypothesis.Hypothesis([5], np.log(0.5))
        ])
    ])

    prior_hypotheses_posterior = prior_hypotheses_per_cluster[0].combine(prior_hypotheses_per_cluster[1])

    for k, h in enumerate(prior_hypotheses_posterior):
        ts = np.array(h.tracks())
        t_idxs = ts - 1
        print(f"Hypothesis {k+1}: {ts}")
        hypos = np.array([pdd.hypothesis.mo_to_to_hypothesis(hh, n) for hh in pdd.hypothesis.hypothesis_enumeration(R, h)])
        print(f" {ts}")
        print(hypos[:, t_idxs])


    exact_margs, exact_norm = pdd.hypothesis.association_marginal_posteriors_normalization_constant(R, prior_hypotheses_posterior)
    print(exact_margs)
    print(exact_norm)

    tracks_c1 = np.sort(np.fromiter(prior_hypotheses_per_cluster[0].tracks(), dtype=int))
    tracks_c2 = np.sort(np.fromiter(prior_hypotheses_per_cluster[1].tracks(), dtype=int))
    Rc1 = cluster_reward_matrix(R, tracks_c1)
    Rc2 = cluster_reward_matrix(R, tracks_c2)

    meas_gated_by_track3 = np.where(np.isfinite(R[2, :m]))[0]
    print(meas_gated_by_track3)

    # We can condition on track 3, to separate the two clusters
    # For misdetection and non-existence, the other tracks are free to associate to whatever.
    track3_log_md = R[2, m + 2]
    track3_log_nonexistence = 0

    # We sum over the two clusters individually. For both clusters we need in this case to only compute two different sums - One where track 3 claims the common measurement in the cluster, and one where it does not
    
