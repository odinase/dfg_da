from scipy.io import loadmat
from dfg_da.marginal_association_Odin import exact_marginal, lbp_marginal_nonexistence, logsumexp, lbp_marginal
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

DATA_PATH = "./data/at612"
MAT_FILE = "priorLikelihood612.mat"



def ws_to_prior_hypotheses(ws, num_clusters=None, use_largest_clusters=True):
    hypos = ws["hypos"].ravel().astype(int)
    hyposCard = ws["hyposCard"].ravel().astype(int)
    probLogHypos = ws["probLogHypos"].ravel()
    clustersCard = ws["clustersCard"].ravel().astype(int)
    clusters = ws["clusters"].ravel().astype(int) - 1 # We negate one here to make hypotheses 0-indexed, which is more convenient. Tracks we keep 1-indexed
    assert (clusters >= 0).all()

    if num_clusters is None:
        num_clusters = len(clustersCard)

    # Let's actually first figure out the clusters we are working with and find the probabilities

    hypo_probs_per_cluster = []
    hypotheses_per_cluster = []

    i = 0
    for clusterC in clustersCard:
        # These hypotheses are in the cluster
        stop = i + clusterC
        hypotheses_in_cluster = clusters[i:stop]

        # hyposCard is as long as probLogCard, so pick out the elements that correspond to the "indices" in hypothesis_in_clutter
        probLogHyposInCluster = probLogHypos[hypotheses_in_cluster]

        hypo_probs = np.exp(probLogHyposInCluster - logsumexp(probLogHyposInCluster))

        hypo_probs_per_cluster.append(hypo_probs)
        hypotheses_per_cluster.append(hypotheses_in_cluster)

        i += clusterC

    # Before we start looping over clusters, let's do this the simple way of making a list of lists, containing tracks contained in each hypothesis, then we sort it afterwards
    tracks_in_hypotheses = [None] * hyposCard.shape[0]
    for hypos_in_cluster in hypotheses_per_cluster:
        for h in hypos_in_cluster:
            start_idx = h
            stop_idx = h + hyposCard[h]
            tracks_in_hypothesis = hypos[start_idx:stop_idx]
            tracks_in_hypotheses[h] = tracks_in_hypothesis

    assert all(h is not None for h in tracks_in_hypotheses)

    # We now have all we need to return proper prior hypotheses

    # Compute the clusters we consider
    clusters_to_use = None
    if use_largest_clusters:
        clusters_to_use = np.argsort(clustersCard)[::-1][:num_clusters]
    else:
        clusters_to_use = np.arange(num_clusters)

    prior_hypotheses_per_cluster = []
    for c in clusters_to_use:
        hypotheses_in_cluster = hypotheses_per_cluster[c]
        hypo_probs_in_cluster = hypo_probs_per_cluster[c]
        prior_hypotheses_in_cluster = [(tracks_in_hypotheses[h], p) for h, p in zip(hypotheses_in_cluster, hypo_probs_in_cluster)]
        prior_hypotheses_per_cluster.append(prior_hypotheses_in_cluster)

    return prior_hypotheses_per_cluster, clusters_to_use


def valid_prob_dist_range(prob_dist):
    return ((0 <= prob_dist) & (prob_dist <= 1.0)).all()


def compute_exact_marginals_by_tot_prob(R_LC, prior_hypotheses):
    n, mp1 = R_LC.shape
    m = mp1 - 1
    all_tracks_idx = np.arange(n)

    normalizing_constants = np.empty(len(prior_hypotheses))
    marginal_total = np.zeros((n, m + 1 + 1))
    conditioned_marginals = np.empty((n, m + 2))

    for k, (tracks, hypo_prob) in enumerate(prior_hypotheses):
        R_sub = R_LC[tracks-1, :]
        JPDAprobs, _, loglikelihood = exact_marginal(R_sub, False)

        # We need to concatenate the JPDAprobs with all tracks and existence probs
        existing_tracks_idx = tracks - 1
        non_existing_tracks_idx = np.delete(all_tracks_idx, existing_tracks_idx)

        existing_probs = np.hstack((JPDAprobs, np.zeros((JPDAprobs.shape[0], 1))))
        nonexisting_probs = np.hstack((np.zeros((len(non_existing_tracks_idx), m + 1)), np.ones((len(non_existing_tracks_idx), 1))))

        conditioned_marginals[existing_tracks_idx] = existing_probs
        conditioned_marginals[non_existing_tracks_idx] =  nonexisting_probs

        normalizing_constant = np.exp(loglikelihood)

        normalizing_constants[k] = normalizing_constant

        marginal_total += conditioned_marginals * normalizing_constant * hypo_prob

    marginal_total = marginal_total / marginal_total.sum(axis=1).reshape(-1, 1)

    assert (np.abs(marginal_total.sum(axis=1) - 1.0) < 1e-6).all()
    assert ((0 <= marginal_total) & (marginal_total <= 1.0)).all()

    return marginal_total, normalizing_constants


def compute_lbp_marginals_by_tot_prob(R_LC, prior_hypotheses, self_normalizing_constants=None):
    n, mp1 = R_LC.shape
    m = mp1 - 1
    all_tracks_idx = np.arange(n)
    
    lbp_marginal_total = np.zeros((n, m + 1 + 1))

    conditioned_marginals = np.empty((n, m + 2))

    normalizing_constants = np.empty(len(prior_hypotheses))

    if self_normalizing_constants is not None:
        assert len(self_normalizing_constants) == len(prior_hypotheses)

    for k, (tracks, hypo_prob) in enumerate(prior_hypotheses):
        R_sub = R_LC[tracks-1, :]
        lbp_probs, bethe_const = lbp_marginal(R_sub)

        # We need to concatenate the JPDAprobs with all tracks and existence probs
        existing_tracks_idx = tracks - 1
        non_existing_tracks_idx = np.delete(all_tracks_idx, existing_tracks_idx)

        existing_probs = np.hstack((lbp_probs, np.zeros((lbp_probs.shape[0], 1))))
        nonexisting_probs = np.hstack((np.zeros((len(non_existing_tracks_idx), m + 1)), np.ones((len(non_existing_tracks_idx), 1))))

        conditioned_marginals[existing_tracks_idx] = existing_probs
        conditioned_marginals[non_existing_tracks_idx] = nonexisting_probs

        loglikelihoods = R_sub[:, 1:]
        gated_measurements = np.any(np.isfinite(loglikelihoods), axis=0)
        loglikelihoods = loglikelihoods[:, gated_measurements]
        
        mu = (1 - np.exp(R_sub[:, 0])).sum()
        if self_normalizing_constants is None:
            normalizing_constant = np.exp(-mu) * (np.exp(loglikelihoods).sum(0) + 1).prod()
        else:
            normalizing_constant = self_normalizing_constants[k]

        # normalizing_constant = bethe_const

        normalizing_constants[k] = normalizing_constant

        lbp_marginal_total += conditioned_marginals * normalizing_constant * hypo_prob

    lbp_marginal_total = lbp_marginal_total / lbp_marginal_total.sum(axis=1, keepdims=True)

    assert (np.abs(lbp_marginal_total.sum(axis=1) - 1.0) < 1e-6).all()
    assert ((0 <= lbp_marginal_total) & (lbp_marginal_total <= 1.0)).all()

    return lbp_marginal_total, normalizing_constants


def compute_marginals(prior_hypotheses, R_LC):
    marginals_exact, normalization_constants = compute_exact_marginals_by_tot_prob(R_LC, prior_hypotheses)
    marginals_lbp, approx_normalization_constants = compute_lbp_marginals_by_tot_prob(R_LC, prior_hypotheses)
    asso_prob, theta_probs, meas_probs = lbp_marginal_nonexistence(R_LC, prior_hypotheses, iter_per_check=250, max_iter=1000)

    return (marginals_exact, normalization_constants), (marginals_lbp, approx_normalization_constants), (asso_prob, theta_probs, meas_probs)


def marginal_statistics(exact_marginals, lbp_marginals):
    abs_error_marginals = np.abs(exact_marginals - lbp_marginals)
    assert ((0 <= abs_error_marginals) & (abs_error_marginals <= 1.0)).all()
    max_errors = abs_error_marginals.max(axis=1)
    abs_errors = abs_error_marginals.ravel()
    misdetection_errors = abs_error_marginals[:, 0]
    detection_errors = abs_error_marginals[:, 1:-1].ravel()
    nonexistence_errors  = abs_error_marginals[:, -1]

    return max_errors, abs_errors, misdetection_errors, detection_errors, nonexistence_errors


def plot_errors(axes, max_errors, abs_errors, misdetection_errors, detection_errors, nonexistence_errors, title_suffix=None):
    assert len(axes) == 5

    max_errors = np.hstack(max_errors)
    abs_errors = np.hstack(abs_errors)
    misdetection_errors = np.hstack(misdetection_errors)
    detection_errors = np.hstack(detection_errors)
    nonexistence_errors = np.hstack(nonexistence_errors)

    errors = [max_errors,
        abs_errors,
        misdetection_errors,
        detection_errors,
        nonexistence_errors]

    titles = [
        "max_errors",
        "abs_errors",
        "misdetection_errors",
        "detection_errors",
        "nonexistence_errors"
    ]
    if title_suffix is not None:
        titles = [title + title_suffix for title in titles]

    for ax, error, title in zip(axes, errors, titles):
        avg = error.mean()
        ax.set_title(f"{title}. avg: {avg:.5e}")
        ax.hist(error)
        ax.loglog()

def plot_survival_function(axes, max_errors, abs_errors, misdetection_errors, detection_errors, nonexistence_errors, label="_"):
    assert len(axes) == 5

    max_errors = np.sort(np.hstack(max_errors))
    abs_errors = np.sort(np.hstack(abs_errors))
    misdetection_errors = np.sort(np.hstack(misdetection_errors))
    detection_errors = np.sort(np.hstack(detection_errors))
    nonexistence_errors = np.sort(np.hstack(nonexistence_errors))

    errors = [max_errors,
        abs_errors,
        misdetection_errors,
        detection_errors,
        nonexistence_errors]

    titles = [
        "max_errors",
        "abs_errors",
        "misdetection_errors",
        "detection_errors",
        "nonexistence_errors"
    ]

    for ax, error, title in zip(axes, errors, titles):
        ax.set_title(title)
        steps = np.linspace(1.0, 0.0, len(error))
        ax.step(error, steps, label=label)
        # ax.set_yscale('symlog')
        ax.set_xscale('symlog', linthresh=1e-15)
        # ax.semilogx()
        ax.semilogy()
        # ax.loglog()
        if label != "_":
            ax.legend()


if __name__ == "__main__":
    ws = loadmat(DATA_PATH + "/" + MAT_FILE)
    R_wrapping = ws["gainMatPostC"] # This R has a strange shape...

    track_file = ws["trackFile"]
    measurements = ws["measurements"]
    
    n = track_file.shape[1]
    m = measurements.shape[1]

    R = R_wrapping[:n, :]

    R_LC = np.hstack((np.diag(R[:,m:])[:,None], R[:,:m]))
    prior_hypotheses_per_cluster, clusters = ws_to_prior_hypotheses(ws)

    max_errors_lbp_tot = []
    abs_errors_lbp_tot = []
    misdetection_errors_lbp_tot = []
    detection_errors_lbp_tot = []
    nonexistence_errors_lbp_tot = []

    max_errors_lbp_full = []
    abs_errors_lbp_full = []
    misdetection_errors_lbp_full = []
    detection_errors_lbp_full = []
    nonexistence_errors_lbp_full = []

    normalizing_constants_exacts = []
    approx_normalizing_constantss = []

    # Compute association marginals for each cluster
    for hypos_in_cluster, cluster in zip(prior_hypotheses_per_cluster, clusters):
        (marginals_exact, normalizing_constants_exact), \
        (lbp_marginals_tot, approx_normalizing_constants), \
        (lbp_marginals_full, _, _) = compute_marginals(hypos_in_cluster, R_LC)

        normalizing_constants_exacts.append(normalizing_constants_exact)
        approx_normalizing_constantss.append(approx_normalizing_constants)

        max_errors_lbp_tot_in_cluster, \
        abs_errors_lbp_tot_in_cluster, \
        misdetection_errors_lbp_tot_in_cluster, \
        detection_errors_lbp_tot_in_cluster, \
        nonexistence_errors_lbp_tot_in_cluster = marginal_statistics(marginals_exact, lbp_marginals_tot)

        max_errors_lbp_tot.append(max_errors_lbp_tot_in_cluster)
        abs_errors_lbp_tot.append(abs_errors_lbp_tot_in_cluster)
        misdetection_errors_lbp_tot.append(misdetection_errors_lbp_tot_in_cluster)
        detection_errors_lbp_tot.append(detection_errors_lbp_tot_in_cluster)
        nonexistence_errors_lbp_tot.append(nonexistence_errors_lbp_tot_in_cluster)

        max_errors_lbp_full_in_cluster, \
        abs_errors_lbp_full_in_cluster, \
        misdetection_errors_lbp_full_in_cluster, \
        detection_errors_lbp_full_in_cluster, \
        nonexistence_errors_lbp_full_in_cluster = marginal_statistics(marginals_exact, lbp_marginals_full)

        max_errors_lbp_full.append(max_errors_lbp_full_in_cluster)
        abs_errors_lbp_full.append(abs_errors_lbp_full_in_cluster)
        misdetection_errors_lbp_full.append(misdetection_errors_lbp_full_in_cluster)
        detection_errors_lbp_full.append(detection_errors_lbp_full_in_cluster)
        nonexistence_errors_lbp_full.append(nonexistence_errors_lbp_full_in_cluster)


    # fig_hist, axes_hist = plt.subplots(ncols=2, nrows=5, sharex=True, sharey=True)

    # fig_hist.suptitle("Histograms")

    # plot_errors(axes_hist[:,0], max_errors_lbp_tot, abs_errors_lbp_tot, misdetection_errors_lbp_tot, detection_errors_lbp_tot, nonexistence_errors_lbp_tot, "_tot")
    # plot_errors(axes_hist[:,1], max_errors_lbp_full, abs_errors_lbp_full, misdetection_errors_lbp_full, detection_errors_lbp_full, nonexistence_errors_lbp_full, "_full")

    fig_sf, axes_sf = plt.subplots(nrows=5, sharex=True)

    fig_sf.suptitle("Survival function")

    plot_survival_function(axes_sf, max_errors_lbp_tot, abs_errors_lbp_tot, misdetection_errors_lbp_tot, detection_errors_lbp_tot, nonexistence_errors_lbp_tot, "Williams LBP with estimated normalization constant")
    plot_survival_function(axes_sf, max_errors_lbp_full, abs_errors_lbp_full, misdetection_errors_lbp_full, detection_errors_lbp_full, nonexistence_errors_lbp_full, "LBP on full problem")

    fig_norm_const, ax_norm_const = plt.subplots(nrows=3)

    fig_norm_const.suptitle("Normalizing constant")

    normalizing_constants_exacts = np.hstack(normalizing_constants_exacts)
    approx_normalizing_constantss = np.hstack(approx_normalizing_constantss)

    ax_norm_const[0].plot(approx_normalizing_constantss, label="approx_normalizing_constant")
    ax_norm_const[0].plot(normalizing_constants_exacts, label="normalizing_constants_exact")
    ax_norm_const[0].set_xlabel("Datapoint")
    ax_norm_const[0].set_ylabel("Normalizing constant value")
    ax_norm_const[0].semilogy()
    ax_norm_const[0].legend()

    ax_norm_const[1].plot(approx_normalizing_constantss, normalizing_constants_exacts, 'o')
    ax_norm_const[1].set_ylabel("Exact normalizing constant")
    ax_norm_const[1].set_xlabel("Approximate normalizing constant")

    f = approx_normalizing_constantss
    y = normalizing_constants_exacts
    y_bar = y.mean()

    SS_res = np.sum((y - f)**2)
    SS_tot = np.sum((y - y_bar)**2)

    R2 = 1.0 - SS_res / SS_tot

    ax_norm_const[1].plot(y, y, '--')
    ax_norm_const[1].set_title(f"R^2: {R2}")

    f = np.log(approx_normalizing_constantss)
    y = np.log(normalizing_constants_exacts)
    y_bar = y.mean()

    SS_res = np.sum((y - f)**2)
    SS_tot = np.sum((y - y_bar)**2)

    R2 = 1.0 - SS_res / SS_tot

    ax_norm_const[2].plot(y, y, '--')
    ax_norm_const[2].set_title(f"R^2: {R2}: log scale")

    plt.show()