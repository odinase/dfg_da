import numpy as np
import py_dfg_da as pdd
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from cluster_data_asso import edmund_to_lc, lc_to_edmund
from collections import defaultdict
from typing import *
from pyehm.core import EHM2
from dataclasses import dataclass


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

@dataclass
class LinkingMappings:
    cluster_to_linking_measurements: Dict[int, MutableSet[int]]
    linking_measurements_to_clusters: Dict[int, MutableSet[int]]


class ClusterLinks:
    def __init__(self, R_LC: np.ndarray, prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList, assocLocal: np.ndarray):
        self.R_LC = R_LC
        self.assocLocal = assocLocal
        self.merging_clusters = self.find_merging_clusters(assocLocal.copy())
        self.all_clusters = set(range(self.assocLocal.shape[1]))

        self.tracks_per_merging_cluster = self.get_tracks_per_merging_cluster(self.merging_clusters, prior_hypotheses_per_cluster)
        
        # We now know what clusters should be merged and their tracks (which we need for finding linking measurements), collect linking measurements
        self.linking_measurement_to_cluster_map = self.find_linking_measurements(self.merging_clusters, self.tracks_per_merging_cluster)
        self.cluster_to_linking_measurement_map = self.invert_lm2c_map(self.linking_measurement_to_cluster_map)

    def meas_to_clusters(self):
        return self.linking_measurement_to_cluster_map

    def cluster_to_linking_meas(self):
        return self.cluster_to_linking_measurement_map

    def clusters_that_merge(self) -> MutableSet[int]:
        return set.union(*self.merging_clusters)

    def unmerging_clusters(self) -> MutableSet[int]:
        return self.all_clusters - self.clusters_that_merge()

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

    def invert_lm2c_map(self, linking_measurement_to_cluster_map):
        # Should be a set of keys which are linking measurements with values set of clusters that gate that measurement
        c2m_map = defaultdict(set)

        # Now loop over the lm2c_map and check if cluster is in set of clusters that gate measurement. If it is, add measurement to value set of inverse mapping
        for cluster in self.all_clusters:
            for meas, cluster_set in linking_measurement_to_cluster_map.items():
                if cluster in cluster_set:
                    c2m_map[cluster].add(meas)
        
        return c2m_map
    
    def invert_c2lm_map(self, cluster_to_linking_measurement_map):
        measurements = set.union(*list(m for m in cluster_to_linking_measurement_map.values()))
        lm2c_map = defaultdict(set)
        # Now loop over the lm2c_map and check if cluster is in set of clusters that gate measurement. If it is, add measurement to value set of inverse mapping
        for m in measurements:
            for cluster, measurement_set in cluster_to_linking_measurement_map.items():
                if m in measurement_set:
                    lm2c_map[m].add(cluster)
        
        return lm2c_map
    
    def linking_mappings_per_merging_clusters(self) -> List[LinkingMappings]:
        map_list = []
        for clusters in self.merging_clusters:
            c2lm = { c: self.cluster_to_linking_measurement_map[c] for c in clusters }
            lm2c = self.invert_c2lm_map(c2lm)
            map_list.append(LinkingMappings(cluster_to_linking_measurements=c2lm, linking_measurements_to_clusters=lm2c))

        return map_list


def multihypothesis_ehm2(R_cluster, prior_hypotheses, reindex_tracks: bool = True):
    if reindex_tracks:
        prior_hypotheses.reindex_tracks()  # reindex tracks to {1, 2, ..., n} for later convenience
    all_tracks_idx = np.arange(R_cluster.shape[0])
    n, mp1 = R_cluster.shape
    marginals = np.zeros((n, mp1))
    conditioned_marginals = np.empty((n, mp1))
    _0 = np.zeros((n, mp1))
    _1 = np.ones((n, 1))
    hypo_cond_normalizing_constants = np.empty(len(prior_hypotheses))

    normalizing_constant_cluster = 0.0

    def R_LC_to_validation_likelihood_matrix(R_LC: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        likelihood_matrix = np.asfortranarray(np.exp(R_LC))
        validation_matrix = np.asfortranarray((likelihood_matrix > 0.0).astype(np.int32))

        return validation_matrix, likelihood_matrix


    for k, hypothesis in enumerate(prior_hypotheses):
        prob = hypothesis.probability()
        existing_tracks_idx = (np.array(hypothesis.tracks()) - 1).astype(int)
        if existing_tracks_idx.shape[0] > 0:
            R_sub = R_cluster[existing_tracks_idx, :]
            validation_matrix, likelihood_matrix = R_LC_to_validation_likelihood_matrix(R_sub)
            JPDAprobs, likelihood = EHM2.run_and_likelihood(validation_matrix, likelihood_matrix)
        else:
            JPDAprobs = np.empty((0, mp1))
            likelihood = 1.0

        # We need to concatenate the JPDAprobs with all tracks and existence probs
        non_existing_tracks_idx = np.setdiff1d(all_tracks_idx, existing_tracks_idx, assume_unique=True)

        existing_probs = np.hstack((JPDAprobs, _0[:len(existing_tracks_idx), 0, None]))
        nonexisting_probs = np.hstack((_0[:len(non_existing_tracks_idx)], _1[:len(non_existing_tracks_idx), 0, None]))

        conditioned_marginals[existing_tracks_idx] = existing_probs
        conditioned_marginals[non_existing_tracks_idx] =  nonexisting_probs

        hypo_cond_normalization_constant = likelihood

        hypo_cond_normalizing_constants[k] = hypo_cond_normalization_constant

        marginals += conditioned_marginals * likelihood * prob

        normalizing_constant_cluster += likelihood * prob

    return marginals, normalizing_constant_cluster


# We can now construct the components of the "tree" (with depth 1 lol).
# We use ClusterLinks above to find the measurements that are linked to other clusters
# It should be initialized with a multicluster case and separate the clusters that are merged with the unaffected clusters
# It is probably the best to do this in two stages - One that separates clusters and delegates the computation, and one that the performs that supercluster marginal computation
class MulticlusterEfficientMarginals:
    def __init__(self, R_LC, prior_hypotheses_per_cluster, assocLocal):
        # Store input for convenience
        self.R_LC = R_LC
        self.prior_hypotheses_per_cluster = prior_hypotheses_per_cluster
        self.assocLocal = assocLocal
        
        # Compute linking measurements and superclusters
        self.cluster_links = ClusterLinks(R_LC=R_LC, prior_hypotheses_per_cluster=prior_hypotheses_per_cluster, assocLocal=assocLocal)

        # Make superclusters
        self.superclusters = [
            ConditionalSuperclusterMarginals(
                R_LC=R_LC,
                prior_hypotheses_per_cluster=prior_hypotheses_per_cluster,
                linking_mappings=linking_mappings
            )
            for linking_mappings in self.cluster_links.linking_mappings_per_merging_clusters()
        ]

    def compute_marginals_likelihood(self) -> np.ndarray:
        # Should in principle be straight forward at this level: simply query the marginals from each cluster/supercluster and concatenate
        n, mp1 = self.R_LC.shape

        marginals = np.empty((n, mp1))
        self._0 = np.zeros((n, mp1))
        self._1 = np.ones((n, 1))

        likelihood = 1.0
        # Collect supercluster marginals
        for supercluster in self.superclusters:
            marginals_supercluster, likelihood_supercluster = supercluster.compute_marginals_likelihood()
            t_idxs = supercluster.supercluster_t_idxs()
            marginals[t_idxs] = marginals_supercluster
            likelihood *= likelihood_supercluster

        # Unmerging clusters just do total marginals over hypotheses
        for cluster in self.cluster_links.unmerging_clusters():
            prior_hypotheses = self.prior_hypotheses_per_cluster[cluster]
            t_idxs = np.sort(np.fromiter(prior_hypotheses.tracks(), dtype=int)) - 1
            R_cluster = self.R_LC[t_idxs]
            marginals_unmerged, likelihood_unmerged = multihypothesis_ehm2(R_cluster, prior_hypotheses)
            marginals[t_idxs] = marginals_unmerged
            likelihood *= likelihood_unmerged

        return marginals, likelihood
            

class ConditionalSuperclusterMarginals:
    def __init__(self, R_LC: np.ndarray, prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList, linking_mappings: LinkingMappings):
        self.R_LC = R_LC
        self.prior_hypotheses_per_cluster = prior_hypotheses_per_cluster
        self.linking_mappings = linking_mappings

        # Construct conditioned clusters
        self.conditioned_clusters: Dict[int, ConditionedCluster] = dict()
        for cluster, linking_measurements in linking_mappings.cluster_to_linking_measurements.items():
            prior_hypotheses = prior_hypotheses_per_cluster[cluster]
            t_idxs = prior_hypotheses.t_idxs()
            R_cluster = self.R_LC[t_idxs]
            linking_measurements = np.fromiter(linking_measurements, dtype=int)
            self.conditioned_clusters[cluster] = ConditionedCluster(
                R_cluster=R_cluster,
                prior_hypotheses=prior_hypotheses,
                linking_measurements=linking_measurements
            )

    def supercluster_t_idxs(self):
        idxs = [ph.tracks() for ph in self.prior_hypotheses_per_cluster]
        return np.sort(np.fromiter(set.union(*idxs), dtype=int)) - 1

    def enumerate_meas_exist(linking_measurements) -> np.ndarray:
        # This function needs to take all linking measurements of the supercluster and make a list of all valid assignments the measurements can do
        # We have access to the map for each cluster what measurements it 

        n = linking_measurements.shape[0]
        return np.array([np.fromiter(np.binary_repr(b, width=n), dtype=int) for b in range(2**n)])


    def compute_marginals_likelihood(self) -> np.ndarray:
        t_idxs = self.supercluster_t_idxs()
        num_tracks = t_idxs.shape[0]
        num_measurements = self.R_LC.shape[1] - 1
        marginals = np.zeros((num_tracks, num_measurements + 1))

        # It is at this point we need to loop over all ways to assign the linking measurements, multiply the clusters conditioned marginals together
        # (since they're independent) and sum up

        

        # for cluster_idx, conditioned_cluster in self.conditioned_clusters.items():
        #     cluster_linking_measurements = conditioned_cluster.linking_measurements
        #     enumerated_meas_exist = self.enumerate_meas_exist(cluster_linking_measurements)
        #     cluster_t_idxs = conditioned_cluster.t_idxs
        #     for meas_exist in enumerated_meas_exist:
        #         conditioned_marginals, conditioned_likelihood = conditioned_cluster.meas_conditionend_marginals(meas_exist)
        #         marginals[cluster_t_idxs] += conditioned_marginals

        return marginals, 1.0


class ConditionedCluster:
    def __init__(self, R_cluster: np.ndarray, prior_hypotheses: pdd.hypothesis.Hypotheses, linking_measurements: np.ndarray):
        self.t_idxs = self.prior_hypotheses.t_idxs()

        self.R_cluster: np.ndarray = R_cluster
        self.prior_hypotheses: pdd.hypothesis.Hypotheses = prior_hypotheses
        self.prior_hypotheses.reindex_tracks()
        self.linking_measurements: np.ndarray = linking_measurements

        # We should definitively cache results, but not sure right now the best way. Will probably be more "obvious" later
        self.cache: Dict[Tuple[int], Tuple[np.ndarray, float]] = dict()

    def conditioned_reward_matrix(self, meas_exist) -> np.ndarray:
        R_conditioned = self.R_cluster.copy()
 
        nonexisting_linking_meas = self.linking_measurements[~meas_exist]
        R_conditioned[:, nonexisting_linking_meas] = -np.inf

        return R_conditioned

    def meas_conditionend_marginals(self, meas_exist: np.ndarray) -> np.ndarray:
        # meas_exist should be m long for all linking measurements in the cluster, where each element is 0 or 1 at index of measurement (0-indexed) indicating whether it's conditionend or not
        meas_exist_tuple = tuple(meas_exist)
        if meas_exist_tuple in self.cache:
            return self.cache[meas_exist_tuple]

        R_conditioned = self.conditioned_reward_matrix(meas_exist)

        # At this point we simply do normal computation??
        marginals_conditioned, likelihood_conditioned = multihypothesis_ehm2(R_conditioned, self.prior_hypotheses, reindex_tracks=False)

        output = (marginals_conditioned, likelihood_conditioned)
        # Cache for later
        self.cache[meas_exist_tuple] = output

        return output


if __name__ == "__main__":
    R, prior_hypotheses_per_cluster, assocLocal = test_case()
