import py_dfg_da as pdd
import numpy as np
from scipy.special import logsumexp


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
    track3_md = np.exp(R[2, m + 2])
    track3_nonexistence = 1.0
    track3_j1 = np.exp(R[2, 0])
    track3_j2 = np.exp(R[2, 1])

    f3 = np.array([track3_md, track3_j1, track3_j2, track3_nonexistence])

    # We sum over the two clusters individually. We need to handle the cases of existence and nonexistence on their own.
    # First sum over existence

    print()
    print()
    Z = 0.0
    for a3, f in enumerate(f3):
        Zc1 = 0.0
        # Cluster 1:
        ph_c1 = prior_hypotheses_per_cluster[0]
        ts = np.sort(np.fromiter(ph_c1.tracks(), dtype=int))
        n = ts.shape[0]
        # Loop over prior hypotheses in cluster 1. If we condition on a3 existing we skip hypotheses not containing the track and vice-versa(?) Should be correct because of compatibility factor making those terms zero
        existing = 0 <= a3 <= m
        for h in ph_c1:
            contained_in_hypo = h.contains(3)
            should_continue = (existing and not contained_in_hypo) or (not existing and contained_in_hypo)
            if should_continue:
                continue

            prior_prob = h.probability()

            hypos = np.array([pdd.hypothesis.mo_to_to_hypothesis(hh, n) for hh in pdd.hypothesis.hypothesis_enumeration(Rc1, h)])
            if existing:
                valid_hypos = np.where(hypos[:, 2] == a3)[0]
                hypos[valid_hypos, :]

            scores = np.array([
                pdd.hypothesis.prior_hypothesis_conditional_association_probability(hh, h, Rc1) for hh in hypos
            ])

            Zc1 += (np.exp(scores)*prior_prob).sum()

        Zc2 = 0.0
        # Cluster 1:
        ph_c2 = prior_hypotheses_per_cluster[1]
        ts = np.sort(np.fromiter(ph_c2.tracks(), dtype=int))
        n = ts.shape[0]
        # Loop over prior hypotheses in cluster 2.
        # Things are probably different here as the track is not contained in any of the prior hypotheses. Instead I assume that the logic is to 
        # Remove illegal associations to a measurement claimed by track 3? This logic should hold above
        existing = 0 <= a3 <= m
        for h in ph_c2:
            contained_in_hypo = h.contains(3)
            should_continue = (existing and not contained_in_hypo) or (not existing and contained_in_hypo)
            if should_continue:
                continue

            prior_prob = h.probability()

            hypos = np.array([pdd.hypothesis.mo_to_to_hypothesis(hh, n) for hh in pdd.hypothesis.hypothesis_enumeration(Rc2, h)])
            if existing:
                valid_hypos = np.where(hypos[:, 2] == a3)[0]
                hypos[valid_hypos, :]

            scores = np.array([
                pdd.hypothesis.prior_hypothesis_conditional_association_probability(hh, h, Rc1) for hh in hypos
            ])

            Zc2 += (np.exp(scores)*prior_prob).sum()

        Z += f * Zc1 * Zc2


    print(Z)