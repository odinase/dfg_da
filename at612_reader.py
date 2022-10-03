from scipy.io import loadmat
from marginal_association_Odin import exact_marginal, logsumexp
import numpy as np


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

    for tracks, p in prior_hypotheses:
        R_sub = R_LC[tracks-1, :]
        JPDAprobs, notTrackProb, loglikelihood = exact_marginal(R_sub, False)

        # We need to concatenate the JPDAprobs with all tracks and existence probs
        existing_tracks_idx = tracks - 1
        non_existing_tracks_idx = np.delete(all_tracks_idx, existing_tracks_idx)

        existing_probs = np.hstack((JPDAprobs, np.zeros((JPDAprobs.shape[0], 1))))
        nonexisting_probs = np.hstack((np.zeros((len(non_existing_tracks_idx), m + 1)), np.ones((len(non_existing_tracks_idx), 1))))

        conditioned_marginals[existing_tracks_idx] = existing_probs
        conditioned_marginals[non_existing_tracks_idx] = nonexisting_probs

        marginal_total += conditioned_marginals*np.exp(loglikelihood)*p

    marginal_total = marginal_total / marginal_total.sum(axis=1).reshape(-1, 1)

    print(marginal_total.shape)

    # print(marginal_total)

    print(marginal_total[existing_tracks - 1])