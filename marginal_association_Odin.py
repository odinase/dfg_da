"""Functions for calculating the marginal track to measurement association probabilities.

Copyright (C) Lars-Christian Ness Tokle - All Rights Reserved.

Functions
---------
lbp_marginal:
    Calculate marginal association probabilities using loopy belief propagation (fast).
exact_rec_marginal:
    Calculate marginal probabilities by naive recursions (slow!).
exact_rec_marginal_inner:
    Calculate the total sum of valid associations in measurement masked lr.
    Meant for internal use, but can be of value in other circumstances.
cluster:
    Clusters tracks and measurements so that no associations between clusters are possible.
exact_marginal:
    Calculate marginal association probabilities by enumeration (OK for smaller problems).

Author: Lars-Christian Ness Tokle (lars-christian.n.tokle@ntnu.no), last modified 21.09.22.
"""
import numpy as np
from scipy.special import logsumexp
import time

# controls some extra (potentially costly) checks done in asserts
DEBUG: bool = True


def lbp_marginal(llr: np.ndarray, max_prob_diff_from_conv: float = 1e-3, max_iter: int = 300, iter_per_check: int = 5, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Calculate marginal association probabilities using loopy belief propagation [1].

    Parameters
    ----------
    llr : np.ndarray[(N, M + 1)]
        log likelihood ratios. llr[:, 0] is the misseded detection.
    max_prob_diff_from_conv : float, optional
        when to deem the iterations as converged, by default 1e-3
    max_iter : int, optional
        upper bound on the number of iterations to do, by default 300
    iter_per_check : int, optional
        how often to check for convergence, by default 5

    Returns
    -------
    track_to_meas_probability: np.ndarray[float, (N, M + 1)]
    new_track_probability: np.ndarray[float, (M,)]

    References
    ----------
    [1] Williams, J., Lau, R. (2014).
        Approximate evaluation of marginal association probabilities with belief propagation.
        IEEE Transactions on Aerospace and Electronic Systems, 50(4), 2942-2959.
        https://doi.org/10.1109/TAES.2014.120568

    """
    n, mp1 = llr.shape
    m = mp1 - 1
    if n == 0 or m == 0:
        return np.zeros((n, mp1)), np.ones(m)

    llr = llr - llr[:, [0]]
    w_nmd = np.exp(llr[:, 1:])

    w_star = np.max(w_nmd.sum(axis=1))
    log1p_w_star = np.log(1 + w_star)
    stop_crit = 0.5 * np.log(1 + max_prob_diff_from_conv)

    it = 0
    conv_val = np.inf

    # note parenthesis for underflow problems

    # NOTE(odin): Add a misdetection term in bottom sum and make 1 for nonexistence?
    a2b_msg = w_nmd / (1 + (w_nmd.sum(axis=1, keepdims=True) - w_nmd))
    b2a_msg = 1 / (1 + (a2b_msg.sum(axis=0, keepdims=True) - a2b_msg))

    while conv_val >= stop_crit and it < max_iter:
        for k in range(iter_per_check):
            assert DEBUG or np.isfinite(a2b_msg).all(), 'a2b not finite'
            assert DEBUG or np.isfinite(b2a_msg).all(), 'b2a not finite'

            w_times_msg = w_nmd * b2a_msg
            # note parenthesis for underflow problems
            a2b_msg = w_nmd / \
                (1 + (w_times_msg.sum(axis=1, keepdims=True) - w_times_msg))

            if k == iter_per_check - 1:
                prevb2a = np.copy(b2a_msg)

            # note parenthesis for underflow problems
            b2a_msg = 1 / (1 + (a2b_msg.sum(axis=0, keepdims=True) - a2b_msg))

            it = it + 1

        bmsg_ratio = b2a_msg / prevb2a
        max_ratio = bmsg_ratio.max()
        min_ratio = bmsg_ratio.min()
        max_abs = max(max_ratio, 1 / min_ratio)
        d = np.log(max_abs)
        if d == 0:
            conv_val = 0
        else:
            alpha = (np.log(1 + w_star * d) - log1p_w_star) / np.log(d)
            conv_val = alpha * (d + stop_crit)

    prob = np.empty(llr.shape)
    w_times_msg = w_nmd * b2a_msg
    s = 1 + w_times_msg.sum(axis=1, keepdims=True)
    # NOTE(odin): Change nominator to misdetection for misdetection and add extra row with one for for nonexistence?
    prob[:, 1:] = w_times_msg / s
    prob[:, [0]] = 1 / s

    not_track_prob: np.ndarray = 1 / (1 + a2b_msg.sum(axis=0))

    assert DEBUG or (np.all(np.isfinite(prob)) and np.all(np.isfinite(not_track_prob))),\
        'not finite probs'

    return prob, not_track_prob


def lbp_marginal_nonexistence(llr: np.ndarray, prior_hypotheses: list[tuple[list[int], float]], max_prob_diff_from_conv: float = 1e-3, max_iter: int = 300, iter_per_check: int = 5, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Calculate marginal association probabilities using loopy belief propagation [1].

    Parameters
    ----------
    llr : np.ndarray[(N, M + 1)]
        log likelihood ratios. llr[:, 0] is the misseded detection.
    max_prob_diff_from_conv : float, optional
        when to deem the iterations as converged, by default 1e-3
    max_iter : int, optional
        upper bound on the number of iterations to do, by default 300
    iter_per_check : int, optional
        how often to check for convergence, by default 5

    Returns
    -------
    track_to_meas_probability: np.ndarray[float, (N, M + 1)]
    new_track_probability: np.ndarray[float, (M,)]

    References
    ----------
    [1] Williams, J., Lau, R. (2014).
        Approximate evaluation of marginal association probabilities with belief propagation.
        IEEE Transactions on Aerospace and Electronic Systems, 50(4), 2942-2959.
        https://doi.org/10.1109/TAES.2014.120568

    """
    n, mp1 = llr.shape
    m = mp1 - 1
    if n == 0 or m == 0:
        return np.zeros((n, mp1)), np.ones(m)

    tracks = range(1, n + 1)

    # Normalize with misdetection to make psi(0) = 1. We skip this step for nonexistence
    # llr = llr - llr[:, [0]]
    w_nmd = np.exp(llr[:, 1:])
    w_0 = np.exp(llr[:, [0]])
    print(w_nmd)
    print(w_0)
    w_N = 1 #np.ones((w_nmd.shape[0], 1))
    # We instead want psi such that psi(0) = m, psi(1, 2, ..., mk) = l and psi(N) = 1
    # w_nmd = np.hstack((w_nmd, w_N))

    w_star = np.max((w_nmd / m).sum(axis=1)) # Scale by m to keep the math the same? convergence criteria??
    log1p_w_star = np.log(1 + w_star)
    stop_crit = 0.5 * np.log(1 + max_prob_diff_from_conv)

    it = 0
    conv_val = np.inf

    # note parenthesis for underflow problems

    # NOTE(odin): Add a misdetection term in bottom sum and make 1 for nonexistence?
    # The nominator should only be over actual measurements, denominator however is over all values of ai / bj, so it should include misdetection and nonexistence


    # Initialize useful data structures for later: - We'll need index lists that broadcast arrays to correct sizes
    # List over each track what hypotheses it exists in. I.e., each row is a track, and that row is true or false for all hypotheses
    t2h_idx = np.array([
        [t in hypo[0] for hypo in prior_hypotheses] for t in tracks
    ])
    t2noth_idx = ~t2h_idx
    h2t_idx = t2h_idx.T

    # Assume sigma(ai = N) = 1 for initialization

    # tracks x measurements
    a2b_msg = w_nmd / (w_0 + (w_nmd.sum(axis=1, keepdims=True) - w_nmd) + w_N)
    # The one in the numerator is due to no ai = 0, so no messages are compatible and the product is just 1
    b2a_msg = 1 / (1 + (a2b_msg.sum(axis=0, keepdims=True) - a2b_msg))

    phi = np.array([hypo[1] for hypo in prior_hypotheses])

    # we need messages from a to theta and theta to a
    # Let's do this carefully. Sigma should be nmber of tracks long, as we only store 1 number per track

    # Multiply psi by nu and sum out measurements
    rho = w_0.ravel() + (w_nmd*b2a_msg).sum(axis=1)

    # The message from theta to track is a little convoluted to compute
    # We need, for each track, to know what hypotheses it's present in and what it's not, and do two sums for each track
    # However, the sum involves doing a product over all other tracks for the rho message, where the product changes

    # We do both sums for each target, such that we slice the necessary 
    def compute_sigma(rho):
        rho_prods = (rho * h2t_idx + t2noth_idx.T).prod(axis=1)
        a = (rho_prods*t2noth_idx*phi).sum(axis=1)
        b = (rho_prods*t2h_idx*phi).sum(axis=1) / rho + 1e-16 # Add small epsilon to avoid divide by zero

        return a / b

    sigma = compute_sigma(rho)

    while conv_val >= stop_crit and it < max_iter:
        for k in range(iter_per_check):
            assert DEBUG or np.isfinite(a2b_msg).all(), 'a2b not finite'
            assert DEBUG or np.isfinite(b2a_msg).all(), 'b2a not finite'

            # We only multiply w_nmd by b2a_msg as for ai = 0 and ai = N all messages multiply to 1 due to normalization
            w_times_msg = w_nmd * b2a_msg

            # tracks x measurements
            a2b_msg = w_nmd / (w_0 + (w_times_msg.sum(axis=1, keepdims=True) - w_times_msg) + w_N*sigma[:,None])

            b2a_msg = 1 / (1 + (a2b_msg.sum(axis=0, keepdims=True) - a2b_msg))

            rho = w_0.ravel() + (w_nmd*b2a_msg).sum(axis=1)

            sigma = compute_sigma(rho)

            if k == iter_per_check - 1:
                prevb2a = np.copy(b2a_msg)

            it = it + 1

        bmsg_ratio = b2a_msg / prevb2a
        max_ratio = bmsg_ratio.max()
        min_ratio = bmsg_ratio.min()
        max_abs = max(max_ratio, 1 / min_ratio)
        d = np.log(max_abs)
        if d == 0:
            conv_val = 0
        else:
            alpha = (np.log(1 + w_star * d) - log1p_w_star) / np.log(d)
            conv_val = alpha * (d + stop_crit)

    print(f"Converged in {it} iters")

    asso_prob = np.empty((n, m + 2))
    # The incoming messages are either from misdetection, which we define as 1, or from measurements, indicating association, or from theta, indicating nonexistence

    # Misdetection, only the prior factor in misdetection
    asso_prob[:, [0]] = w_0
    # Association, use the messages from b
    asso_prob[:, 1:-1] = w_nmd * b2a_msg
    # Nonexistence, use sigma
    asso_prob[:, -1] = w_N * sigma

    asso_prob = asso_prob / asso_prob.sum(axis=1, keepdims=True)

    theta_probs = phi * np.array([np.prod(rho[h]) for h in h2t_idx])
    theta_probs = theta_probs / theta_probs.sum()

    meas_probs = np.empty((m, 1 + n))
    meas_probs[:, 0] = 1
    meas_probs[:, 1:] = a2b_msg.T

    meas_probs = meas_probs / meas_probs.sum(axis=1, keepdims=True)

    return asso_prob, theta_probs, meas_probs



def exact_rec_marginal(llr_or_lr: np.ndarray, is_log: bool = True, **kwargs,
                       ) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate marginal probabilities by naive recursions.

    Parameters
    ----------
    llr_or_lr : np.ndarray[N, M + 1]
        the likelihood ratios, if `is_log = True` it is the logarithm of these.
        llr_or_lr[:, 0] is for missed detection.
    is_log : bool, optional
        indicates if the logarithm is given or not, by default True.

    Returns
    -------
    track_to_meas_probs: np.ndarray[N, M + 1]
    new_track_probs: np.ndarray[M]
    normalization_const: float
    """
    if len(llr_or_lr) == 0:
        pr_a = np.empty_like(llr_or_lr)
        pr_new = np.ones(llr_or_lr.shape[1] - 1)
        return pr_a, pr_new
    if len(llr_or_lr) == 1:
        lr = np.exp(llr_or_lr) if is_log else llr_or_lr
        pr_a = lr / lr.sum()
        pr_new = 1 - pr_a[1:]
        return pr_a, pr_new

    lr = np.exp(llr_or_lr) if is_log else llr_or_lr

    m_mask = np.ones(lr.shape[1], dtype=bool)
    tot_sum = exact_rec_marginal_inner(lr, m_mask)
    pr_a = np.empty_like(lr)
    for i, lri in enumerate(lr):
        lr_removed_i = np.delete(lr, i, axis=0)
        for j, lrij in enumerate(lri):
            if lrij > 0:
                m_mask[j] = (j == 0)  # set to false when not missed detection
                pr_a[i, j] = lrij * \
                    exact_rec_marginal_inner(lr_removed_i, m_mask)
                m_mask[j] = True
            else:
                pr_a[i, j] = 0

    pr_new = np.empty_like(lr[0, 1:])
    for j in range(1, lr.shape[1]):
        m_mask[j] = False
        pr_new[j - 1] = exact_rec_marginal_inner(lr, m_mask)
        m_mask[j] = True

    sums = pr_a.sum(axis=1, keepdims=True)
    assert not DEBUG or np.allclose(
        sums, tot_sum), "Nope, they were not equal..."

    pr_a /= tot_sum
    pr_new /= tot_sum

    return pr_a, pr_new, tot_sum


def exact_rec_marginal_inner(lr: np.ndarray, m_mask) -> float:
    """Calculate the total sum of valid associations in measurement masked lr."""
    if len(lr) == 1:
        return lr[0, m_mask].sum()

    tot = 0
    for j in np.flatnonzero(m_mask):
        if lr[0, j] > 0:
            m_mask[j] = (j == 0)  # set to false when not missed detection
            tot += lr[0, j] * exact_rec_marginal_inner(lr[1:], m_mask)
            m_mask[j] = True
    assert not DEBUG or np.isfinite(tot),\
        "non finite number encounterd in calculation"
    return tot


def cluster(llr: np.ndarray) -> list[tuple[set[int], set[int]]]:
    """Clusters tracks and measurements so that no associations between clusters are possible.

    Parameters
    ----------
    llr : np.ndarray[N, M + 1]
        the log likelihood ratios. llr[:, 0] are for missed detections.
        -inf is used  for impossible assos.

    Returns
    -------
    clusters: list[tuple[set[int], set[int]]]
        list of set of tracks and measurements in the clusters.
    """
    g = llr[:, 1:] > -np.inf
    roots: list[tuple[set[int], set[int]]] = []
    for i, gi in enumerate(g):
        Mset = set(np.flatnonzero(gi) + 1)
        Trset = {i}
        no_longer_root = []
        for ii, r in enumerate(roots):
            if len(Mset & r[1]) > 0:
                Mset |= r[1]
                Trset |= r[0]
                no_longer_root.append(ii)
        for ii in reversed(no_longer_root):
            del roots[ii]
        roots.append((Trset, Mset))
    return roots


def exact_marginal(llr: np.ndarray, do_cluster: bool = True, **kwargs) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate marginal association probabilities by enumeration.

    Parameters
    ----------
    llr : np.ndarray[N, M + 1]
        log likelihood ratios for the track to measurement assignments. meas 0 is missed.
    do_cluster : bool, optional
        decide to do clustering before calculation, by default True

    Returns
    -------
    track_to_meas_probs: np.ndarray[N, M + 1]
    new_track_prob: np.ndarray[M]
    log_normalizing_const: float

    Raises
    ------
    ValueError:
        when the problem is too large
    """
    g = (llr > -np.inf)
    if do_cluster:
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

                pr_a[np.ix_(ctr, ca)], pr_new[cm] = exact_marginal(
                    llr[np.ix_(ctr, ca)], False)
                usedtr[ctr] = True
                usedm[cm] = True
        pr_new[~usedm] = 1
        return pr_a, pr_new

    w = np.exp(llr)

    n, mp1 = w.shape
    m = mp1 - 1

    # indices of possible measurement associations after gating per track
    # including no detection
    tracks_possible_associations = [np.nonzero(gi)[0] for gi in g]
    # Number of all possible hypotheses when only considering tracks after i
    # inclusive (both valid and invalid hypotheses).
    Nhypotheses = np.concatenate((np.flip(np.cumprod(
        [pai.shape[0] for pai in reversed(tracks_possible_associations)], dtype=np.uint)),
        np.ones(1, dtype=np.uint)))

    if n * mp1 * Nhypotheses[0] > 10 ** 9:
        raise ValueError("Too complex problem")
    # Generate a matrix holding the track hypothesis for the joint hypothesis
    index_matrix = np.empty((n, Nhypotheses[0]), dtype=np.int16)
    numbers = np.arange(Nhypotheses[0])  # hypoteses zero indexed
    for i in range(n):
        # No. of hyps among remaining tracks
        mode_number = Nhypotheses[(i + 1)]
        # same i hyp for all hyps amomg remaining tracks, but varying for prev
        # tracks. rounds the counting system so that target 1 gets
        # 0,0,0,0,1,1,1,1..., target 2 gets 0,0,1,1,0,0,1,1 etc.
        index_matrix[i] = np.floor(numbers / mode_number)
        # make the indeces roll faster for next track
        numbers = np.fmod(numbers, mode_number)

    # Hypotheses matrix. Each row represents a track. Each column
    # represents a hypothesis.
    # Elements keep the index of the measurement associated to the track for
    # the hypothesis.
    JPDA_hyp_mat = np.empty((n, Nhypotheses[0]), dtype=np.int16)

    # Hypotheses probabilities #! SHOULD BE ZEROS??
    hyp_prob_log = np.ones((Nhypotheses[0],))

    # calculate "extended alphabet" association hypothesis, unnormalized probability.
    for i in range(n):
        # Fill in the associations corresponding to the tracks
        JPDA_hyp_mat[i] = tracks_possible_associations[i][index_matrix[i]]
        # Update the hypothesis probabilities
        hyp_prob_log += llr[i, JPDA_hyp_mat[i]]

    # remove infeasible hypotheses from "alphabet" (ie. set prob to zero) and normalize.
    JPDA_hyp_mat_is_j = JPDA_hyp_mat[None] == np.arange(m + 1)[:, None, None]

    # times the jth measurement appears in each hypothesis
    num_meas_j_in_hyp = JPDA_hyp_mat_is_j[1:].sum(axis=1)
    # meas j used more than once ==> infeasible
    unfeasible_hyps_for_j = num_meas_j_in_hyp > 1
    # any meas used more than once ==> infeasible
    unfeasible_hyps = np.any(unfeasible_hyps_for_j, axis=0)
    hyp_prob_log[unfeasible_hyps] = -np.inf  # -inf ==> zero prob

    # normalize and get probabilities
    loglikelihood = logsumexp(hyp_prob_log)  # TODO: Verify that this is ll
    hyp_prob_log -= loglikelihood  # Normalize the probabilities
    hypProb = np.exp(hyp_prob_log)

    # calculate the marginalization
    # marginal probability matrix: Rows as tracks and Columns as associations.
    JPDAprobs = (hypProb[None, None] * JPDA_hyp_mat_is_j).sum(axis=2).T

    j_new_track_hyps = np.logical_not(np.any(JPDA_hyp_mat_is_j[1:], axis=1))
    # array of probabilities for j not coming from existing track
    notTrackProb = (hypProb[None] * j_new_track_hyps).sum(axis=1)

    # calculate number of hypotheses
    # Nhyp = (hypProbLog >= -np.inf).sum()
    return JPDAprobs, notTrackProb, loglikelihood
