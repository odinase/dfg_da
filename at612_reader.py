from scipy.io import loadmat
from marginal_association_Odin import exact_marginal, lbp_marginal_nonexistence, logsumexp, lbp_marginal
import numpy as np

import matplotlib.pyplot as plt


DATA_PATH = "./data/at612"
MAT_FILE = "priorLikelihood612.mat"


def ws_to_prior_hypotheses(ws):
    hypos = ws["hypos"].ravel()
    hyposCard = ws["hyposCard"].ravel()

    probLogHypos = ws["probLogHypos"].ravel()

    # TODO(odin): Generalize later to use all clusters. For now, only use largest cluster

    clustersCard = ws["clustersCard"].ravel()
    # Ugly hardcoding, we know that the largest cluster is the first at 149 hypotheses
    hypotheses_to_consider = clustersCard[0]

    hyposCard = hyposCard[:hypotheses_to_consider]


    # Compute normalizing constant for cluster hypothesis probabilities
    probLogHypos = probLogHypos[:hypotheses_to_consider]
    hypo_probs = np.exp(probLogHypos - logsumexp(probLogHypos))

    assert abs(hypo_probs.sum() - 1) < 1e-6

    prior_hypotheses = []

    start = 0

    for hc, prob in zip(hyposCard, hypo_probs):
        stop = start + hc
        tracks = hypos[start:stop]

        prior_hypotheses.append((tracks, prob))

        start = stop

    return prior_hypotheses


def tot_existence_prob(prior_hypotheses, num_tracks):
    existence_probs = np.zeros((num_tracks, 2))
    for tracks, prob in prior_hypotheses:
        existence_probs[tracks - 1, 1] += prob

    existence_probs[:, 0] = 1.0 - existence_probs[:, 1]

    return existence_probs
        


if __name__ == "__main__":
    ws = loadmat(DATA_PATH + "/" + MAT_FILE)
    R_wrapping = ws["gainMatPostC"] # This R has a strange shape...

    track_file = ws["trackFile"]
    measurements = ws["measurements"]
    
    n = track_file.shape[1]
    m = measurements.shape[1]

    R = R_wrapping[:n, :]

    print(R.shape)

    R_LC = np.hstack((np.diag(R[:,m:])[:,None], R[:,:m]))

    print(R_LC.shape)

    prior_hypotheses = ws_to_prior_hypotheses(ws)
    print(len(prior_hypotheses))

    existing_tracks = np.unique([t for tracks, p in prior_hypotheses for t in tracks])

    print(existing_tracks)
    print(existing_tracks.shape)

    marginal_total = np.zeros((n, m + 1 + 1))

    all_tracks_idx = np.arange(n)

    conditioned_marginals = np.empty((n, m + 2))

    exact_normalizing_constant = np.empty(len(prior_hypotheses))

    for k, (tracks, p) in enumerate(prior_hypotheses):
        R_sub = R_LC[tracks-1, :]
        JPDAprobs, notTrackProb, loglikelihood = exact_marginal(R_sub, False)

        # We need to concatenate the JPDAprobs with all tracks and existence probs
        existing_tracks_idx = tracks - 1
        non_existing_tracks_idx = np.delete(all_tracks_idx, existing_tracks_idx)

        existing_probs = np.hstack((JPDAprobs, np.zeros((JPDAprobs.shape[0], 1))))
        nonexisting_probs = np.hstack((np.zeros((len(non_existing_tracks_idx), m + 1)), np.ones((len(non_existing_tracks_idx), 1))))

        conditioned_marginals[existing_tracks_idx] = existing_probs
        conditioned_marginals[non_existing_tracks_idx] =  nonexisting_probs

        normalizing_constant = np.exp(loglikelihood)

        exact_normalizing_constant[k] = normalizing_constant

        marginal_total += conditioned_marginals*np.exp(loglikelihood)*p

    marginal_total = marginal_total / marginal_total.sum(axis=1).reshape(-1, 1)

    lbp_marginal_total = np.zeros((n, m + 1 + 1))

    conditioned_marginals = np.empty((n, m + 2))

    approx_normalizing_constant = np.empty(len(prior_hypotheses))
    approx_log_normalizing_constant = np.empty(len(prior_hypotheses))

    hypotheses_all_tracks_detected = np.empty(len(prior_hypotheses), dtype=bool)

    for k, (tracks, p) in enumerate(prior_hypotheses):
        R_sub = R_LC[tracks-1, :]
        lbp_probs, notTrackProb = lbp_marginal(R_sub)

        # We need to concatenate the JPDAprobs with all tracks and existence probs
        existing_tracks_idx = tracks - 1
        non_existing_tracks_idx = np.delete(all_tracks_idx, existing_tracks_idx)

        existing_probs = np.hstack((lbp_probs, np.zeros((lbp_probs.shape[0], 1))))
        nonexisting_probs = np.hstack((np.zeros((len(non_existing_tracks_idx), m + 1)), np.ones((len(non_existing_tracks_idx), 1))))

        conditioned_marginals[existing_tracks_idx] = existing_probs
        conditioned_marginals[non_existing_tracks_idx] = nonexisting_probs

        # detected_tracks = np.any(np.isfinite(R_sub[:, 1:]), axis=0)
        loglikelihoods = R_sub[:, 1:] #[:, detected_tracks]
        # hypotheses_all_tracks_detected[k] = detected_tracks.all()

        mu = (1 - np.exp(R_sub[:, 0])).sum()
        print(f"mu: {np.exp(-mu)}")
        normalizing_constant = np.exp(-mu)*np.exp(loglikelihoods).sum(axis=0).prod()

        approx_normalizing_constant[k] = normalizing_constant

        lbp_marginal_total += conditioned_marginals * normalizing_constant * p

    lbp_marginal_total = lbp_marginal_total / lbp_marginal_total.sum(axis=1, keepdims=True)


    lbp_probs_total_sub, notTrackProb = lbp_marginal(R_LC)

    existence_probs = tot_existence_prob(prior_hypotheses, n)

    lbp_probs_total = np.empty((lbp_probs_total_sub.shape[0], lbp_probs_total_sub.shape[1] + 1))

    lbp_probs_total[:, [-1, 0]] = lbp_probs_total_sub[:, [0]]*existence_probs
    lbp_probs_total[:, 1:-1] = lbp_probs_total_sub[:, 1:]

    # print(marginal_total)
    # print(lbp_marginal_total)

    # print(exact_normalizing_constant)
    # print(approx_normalizing_constant)
    # print(approx_log_normalizing_constant)

    exact_normalizing_constant = exact_normalizing_constant / exact_normalizing_constant.sum()
    approx_normalizing_constant = approx_normalizing_constant / approx_normalizing_constant.sum()

    figz, ax_norm_const = plt.subplots()
    ax_norm_const.plot(exact_normalizing_constant, approx_normalizing_constant, 'x', label="Normalization constant for hypotheses")
    xstart = exact_normalizing_constant.min()
    xstop = exact_normalizing_constant.max()
    x = np.linspace(xstart, xstop, exact_normalizing_constant.shape[0])
    ax_norm_const.plot(x, x, '--', label="Ideal mapping")
    ax_norm_const.set_title("Normalization constant")
    ax_norm_const.set_ylabel("Approximated normalizing constant")
    ax_norm_const.set_xlabel("Exact normalizing constant")
    ax_norm_const.legend()


    fig, axes = plt.subplots(nrows=2)
    marginal_error_means = (marginal_total - lbp_marginal_total).mean(axis=1)
    marginal_error_stds = (marginal_total - lbp_marginal_total).std(axis=1)

    axes[0].plot(marginal_error_means)
    axes[0].plot(marginal_error_means + marginal_error_stds, 'b--')
    axes[0].plot(marginal_error_means - marginal_error_stds, 'b--')
    axes[0].set_title(f"LBPs conditioned on prior hypotheses. RMSE: {np.sqrt((marginal_error_means**2).mean())}, median: {np.median(marginal_error_means)}")

    asso_prob, theta_probs, meas_probs = lbp_marginal_nonexistence(R_LC, prior_hypotheses, iter_per_check=300)

    marginal_error_means = (marginal_total - asso_prob).mean(axis=1)
    marginal_error_stds = (marginal_total - asso_prob).std(axis=1)

    axes[1].plot(marginal_error_means)
    axes[1].plot(marginal_error_means + marginal_error_stds, 'b--')
    axes[1].plot(marginal_error_means - marginal_error_stds, 'b--')
    axes[1].set_title(f"LBP complete. RMSE: {np.sqrt((marginal_error_means**2).mean())}, median: {np.median(marginal_error_means)}")


    plt.show()