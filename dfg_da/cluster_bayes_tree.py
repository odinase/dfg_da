import numpy as np
import py_dfg_da as pdd


def test_case():
    R = np.array([
        [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
        [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
        [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
        [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
        [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    ], order='F')

    n, mpn = R.shape
    m = mpn - n

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

    return R, prior_hypotheses_per_cluster


def find_interacting_tracks(R, tracks_per_cluster):
    n, mpn = R.shape
    m = mpn - n
    
    Rd = R[:, :m]

    # First we find the set of gated measurements for each cluster
    measurement_sets = []
    for tracks in tracks_per_cluster:
        # Get detections for cluster
        Rdc = Rd[tracks - 1]

        # For now we only concern ourselves with measurements that some track has gated, not what track
        # In this case the measurement set is just the indices of columns with at least one finite entry
        gated_measurements = set(np.where(np.isfinite(Rdc).any(axis=0))[0])

        measurement_sets.append(gated_measurements)

    # We now do the intersection over all the measurment sets to find the troubling measurements
    # 
    # We can find the cliques of interacting tracks by looking at the measurements gated by tracks in different clusters
    


# We need to know the tracks that have interacted with each other, and the cluster they belong to.
# To make this efficient we will need to know the cliques form
# First and foremost - The root is always the interacting tracks
# Not sure of the best way to find the set of tracks that 

if __name__ == "__main__":
    R, prior_hypotheses_per_cluster = test_case()

    trcaks_per_cluster = [np.sort(np.fromiter(ph.tracks(), dtype=int)) for ph in prior_hypotheses_per_cluster]

    find_interacting_tracks(R, trcaks_per_cluster)