import numpy as np
import py_dfg_da as pdd
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from cluster_data_asso import edmund_to_lc, lc_to_edmund
from collections import defaultdict
from typing import List, FrozenSet


def test_case():
    R = np.array([
        [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
        [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
        [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
        [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
        [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    ], order='F')

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

    assocLocal = np.array([
        [1, 1],
        [1, 0]
    ])

    return R, prior_hypotheses_per_cluster, assocLocal


class ClusterLinks:
    def __init__(self, R_LC: np.ndarray, prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList, assocLocal: np.ndarray):
        self.R_LC = R_LC
        self.merging_clusters = self.find_merging_clusters(assocLocal.copy())
        self.tracks_per_cluster = self.get_tracks_per_merging_cluster(self.merging_clusters, prior_hypotheses_per_cluster)

    def find_merging_clusters(self, assocLocal) -> List[FrozenSet]:
        assocLocal[0] -= 1
        master_clusters, counts = np.unique(assocLocal[0], return_counts=True)
        merging_masters = master_clusters[counts > 1]
        merging_clusters = []
        for merging_master in merging_masters:
            clusters_to_merge = np.where(assocLocal[0] == merging_master)[0]
            merging_clusters.append(frozenset(clusters_to_merge))

        return merging_clusters

    def get_tracks_per_cluster(self, merging_clusters, prior_hypotheses_per_cluster) -> List[np.ndarray]:
        relevant_clusters = set()
        for merge in merging_clusters:
            relevant_clusters |= merge

        tracks_per_cluster = []
        for cluster in relevant_clusters:
            tracks = np.sort(np.fromiter(prior_hypotheses_per_cluster[cluster]))
            tracks_per_cluster.append(tracks)

        return tracks_per_cluster


    def add_measurement_link(self, measurement_idx: int, cluster1: int, cluster2):
        link_tuple = (min(cluster1, cluster2), max(cluster1, cluster2))
        self.cluster_links[link_tuple].add(measurement_idx)


def find_interacting_tracks(R, prior_hypotheses_per_cluster):
    n, mpn = R.shape
    m = mpn - n

    Rd = R[:, :m]

    num_clusters = len(prior_hypotheses_per_cluster)
    trcaks_per_cluster = [np.sort(np.fromiter(ph.tracks(), dtype=int)) for ph in prior_hypotheses_per_cluster]

    cluster_links = ClusterLinks()
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
    R, prior_hypotheses_per_cluster, assocLocal = test_case()


    find_interacting_tracks(R, trcaks_per_cluster)