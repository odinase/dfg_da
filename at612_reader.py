from scipy.io import loadmat
from marginal_association_Odin import exact_marginal, lbp_marginal_nonexistence, logsumexp, lbp_marginal
import numpy as np

import matplotlib.pyplot as plt


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
        prior_hypotheses_in_cluster = []
        hypotheses_in_cluster = hypotheses_per_cluster[c]
        hypo_probs_in_cluster = hypo_probs_per_cluster[c]
        for h, p in zip(hypotheses_in_cluster, hypo_probs_in_cluster):
            prior_hypotheses_in_cluster.append((h, p))
        
        prior_hypotheses_per_cluster.append(prior_hypotheses_in_cluster)

    return prior_hypotheses_per_cluster        


def tot_existence_prob(prior_hypotheses, num_tracks):
    existence_probs = np.zeros((num_tracks, 2))
    for tracks, prob in prior_hypotheses:
        existence_probs[tracks - 1, 1] += prob

    existence_probs[:, 0] = 1.0 - existence_probs[:, 1]

    return existence_probs


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

    return marginal_total, normalizing_constants


def compute_lbp_marginals_by_tot_prob(R_LC, prior_hypotheses):
    n, mp1 = R_LC.shape
    m = mp1 - 1
    all_tracks_idx = np.arange(n)
    
    lbp_marginal_total = np.zeros((n, m + 1 + 1))

    conditioned_marginals = np.empty((n, m + 2))

    normalizing_constants = np.empty(len(prior_hypotheses))

    for k, (tracks, hypo_prob) in enumerate(prior_hypotheses):
        R_sub = R_LC[tracks-1, :]
        lbp_probs, _ = lbp_marginal(R_sub)

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
        normalizing_constant = np.exp(-mu)*np.exp(loglikelihoods).sum(axis=0).prod()

        normalizing_constants[k] = normalizing_constant

        lbp_marginal_total += conditioned_marginals * normalizing_constant * hypo_prob

    lbp_marginal_total = lbp_marginal_total / lbp_marginal_total.sum(axis=1, keepdims=True)

    return lbp_marginal_total, normalizing_constants


if __name__ == "__main__":
    ws = loadmat(DATA_PATH + "/" + MAT_FILE)
    R_wrapping = ws["gainMatPostC"] # This R has a strange shape...

    track_file = ws["trackFile"]
    measurements = ws["measurements"]
    
    n = track_file.shape[1]
    m = measurements.shape[1]

    R = R_wrapping[:n, :]

    R_LC = np.hstack((np.diag(R[:,m:])[:,None], R[:,:m]))

    num_clusters = 1

    prior_hypotheses = ws_to_prior_hypotheses(ws)



    # asso_prob, theta_probs, meas_probs = lbp_marginal_nonexistence(R_LC, prior_hypotheses, iter_per_check=300)
