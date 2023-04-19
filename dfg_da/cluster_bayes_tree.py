import numpy as np
import py_dfg_da as pdd
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from cluster_data_asso import edmund_to_lc, lc_to_edmund
from collections import defaultdict
from typing import *


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
        self.tracks_per_merging_cluster = self.get_tracks_per_merging_cluster(self.merging_clusters, prior_hypotheses_per_cluster)
        
        # We now know what clusters should be merged and their tracks (which we need for finding linking measurements), collect linking measurements
        self.linking_measurement_to_cluster_map = self.find_linking_measurements(self.merging_clusters, self.tracks_per_merging_cluster)
        self.cluster_to_linking_measurement_map = self.invert_lm2c_map(self.merging_clusters, self.linking_measurement_to_cluster_map)

    def meas_to_clusters(self):
        return self.linking_measurement_to_cluster_map
    
    def cluster_to_linking_meas(self):
        return self.cluster_to_linking_measurement_map

    def find_merging_clusters(self, assocLocal) -> List[MutableSet]:
        assocLocal[0] -= 1
        master_clusters, counts = np.unique(assocLocal[0], return_counts=True)
        merging_masters = master_clusters[counts > 1]
        merging_clusters = []
        for merging_master in merging_masters:
            clusters_to_merge = np.where(assocLocal[0] == merging_master)[0]
            merging_clusters.append(set(clusters_to_merge))

        return merging_clusters

    def get_tracks_per_merging_cluster(self, merging_clusters, prior_hypotheses_per_cluster) -> Dict[int, np.ndarray]:
        relevant_clusters = set()
        for merge in merging_clusters:
            relevant_clusters |= merge

        tracks_per_merging_cluster = dict()
        for cluster in relevant_clusters:
            tracks = np.sort(np.fromiter(prior_hypotheses_per_cluster[cluster].tracks(), dtype=int))
            tracks_per_merging_cluster[cluster] = tracks

        return tracks_per_merging_cluster
    
    def gated_measurements(self, R_sub) -> FrozenSet[int]:
        Rd = R_sub[:, 1:]
        return frozenset(np.where(np.isfinite(Rd).any(axis=0))[0] + 1)  # Add 1 to use 1-indexed measurements

    def find_linking_measurements(self, merging_clusters, tracks_per_merging_cluster):
        # Initialize map from measurements to cluster idxs
        m2c_map = dict()

        num_measurements = self.R_LC.shape[1] - 1
        for cluster_idxs in merging_clusters:
            num_clusters = len(cluster_idxs)
            gated_measurements = np.empty((num_clusters, num_measurements), dtype=int)
            for c, cluster in enumerate(cluster_idxs):
                t_idx = tracks_per_merging_cluster[cluster] - 1
                R_sub = self.R_LC[t_idx]
                gated_measurements[c] = np.isfinite(R_sub[:, 1:]).any(axis=0)

            # We need to construct an index map back to the original clusters
            c_map = np.array(list(cluster_idxs))

            for j, clusters_gated_meas in enumerate(gated_measurements.T):
                if clusters_gated_meas.sum() > 1:
                    assert not (j + 1 in m2c_map), f"Measurement {j + 1} already accounted for, should not be possible(?)"
                    m2c_map[j + 1] = set(c_map[np.where(clusters_gated_meas)[0]])

        return m2c_map

    def invert_lm2c_map(self, merging_clusters, linking_measurement_to_cluster_map):
        # Should be a set of keys which are linking measurements with values set of clusters that gate that measurement
        # Start by forming set of all clusters that are merging
        all_clusters = set.union(*merging_clusters)
        c2m_map = defaultdict(set)

        # Now loop over the lm2c_map and check if cluster is in set of clusters that gate measurement. If it is, add measurement to value set of inverse mapping
        for cluster in all_clusters:
            for meas, cluster_set in linking_measurement_to_cluster_map.items():
                if cluster in cluster_set:
                    c2m_map[cluster].add(meas)
        
        return c2m_map            


if __name__ == "__main__":
    R, prior_hypotheses_per_cluster, assocLocal = test_case()


    find_interacting_tracks(R, trcaks_per_cluster)