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

        detected_tracks = np.any(np.isfinite(R_sub[:, 1:]), axis=0)
        loglikelihoods = R_sub[:, 1:][:, detected_tracks]
        hypotheses_all_tracks_detected[k] = detected_tracks.all()


        normalizing_constant = np.exp(loglikelihoods).sum(axis=0).prod()
        approx_log_normalizing_constant[k] = logsumexp(R_sub[: ,1:], axis=0).sum()

        approx_normalizing_constant[k] = normalizing_constant

        lbp_marginal_total += conditioned_marginals * normalizing_constant * p

    lbp_marginal_total = lbp_marginal_total / lbp_marginal_total.sum(axis=1).reshape(-1, 1)


    lbp_probs_total, notTrackProb = lbp_marginal_nonexistence(R_LC)


    # print(marginal_total)
    # print(lbp_marginal_total)

    # print(exact_normalizing_constant)
    # print(approx_normalizing_constant)
    # print(approx_log_normalizing_constant)

    exact_normalizing_constant = exact_normalizing_constant / exact_normalizing_constant.sum()
    approx_normalizing_constant = approx_normalizing_constant / approx_normalizing_constant.sum()

    plt.figure()
    plt.plot(exact_normalizing_constant[hypotheses_all_tracks_detected], approx_normalizing_constant[hypotheses_all_tracks_detected], 'go', label='All tracks detected')
    plt.plot(exact_normalizing_constant[~hypotheses_all_tracks_detected], approx_normalizing_constant[~hypotheses_all_tracks_detected], 'rx', label='Not all tracks detected')
    plt.legend()

    fig, ax = plt.subplots(nrows=2)
    marginal_error_means = (marginal_total - lbp_marginal_total).mean(axis=1)
    marginal_error_stds = (marginal_total - lbp_marginal_total).std(axis=1)

    marginal_error_means_2 = (marginal_total - lbp_probs_total).mean(axis=1)
    marginal_error_stds_2 = (marginal_total - lbp_probs_total).std(axis=1)

    ax[0].plot(marginal_error_means)
    ax[0].plot(marginal_error_means + marginal_error_stds, 'b--')
    ax[0].plot(marginal_error_means - marginal_error_stds, 'b--')
    ax[0].set_title("LBPs conditioned on prior hypotheses")

    ax[1].plot(marginal_error_means_2)
    ax[1].plot(marginal_error_means_2 + marginal_error_stds_2, 'b--')
    ax[1].plot(marginal_error_means_2 - marginal_error_stds_2, 'b--')
    ax[1].set_title("LBP on all tracks")

    plt.figure()
    plt.spy(np.exp(R))

    plt.show()