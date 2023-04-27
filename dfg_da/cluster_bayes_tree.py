import numpy as np
import py_dfg_da as pdd
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from cluster_data_asso import edmund_to_lc, lc_to_edmund
from collections import defaultdict
from typing import *
from pyehm.core import EHM2
from dataclasses import dataclass
from dfg_da.marginal_association_Odin import exact_marginal
import dfg_da.marginals_computers as mc
from scipy.special import binom


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

    def all_linking_measurements_idxs(self) -> MutableSet[int]:
        return set(m for m in self.linking_measurements_to_clusters)

    def all_cluster_idxs(self) -> MutableSet[int]:
        return set(c for c in self.cluster_to_linking_measurements)


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


def cartesian_product(*arrays):
    import numpy

    la = len(arrays)
    dtype = numpy.result_type(*arrays)
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def multihypothesis_ehm2(R_cluster, prior_hypotheses, reindex_tracks: bool = True):
    if reindex_tracks:
        prior_hypotheses.reindex_tracks()  # reindex tracks to {1, 2, ..., n} for later convenience
    all_tracks_idx = np.arange(R_cluster.shape[0])
    n, mp1 = R_cluster.shape
    marginals = np.zeros((n, mp1 + 1))
    conditioned_marginals = np.empty((n, mp1 + 1))  # Needs to add nonexistence
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
            # JPDAprobs, _, loglikelihood = exact_marginal(R_sub, False)
            # likelihood = np.exp(loglikelihood)
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

    marginals = marginals / marginals.sum(axis=1, keepdims=True)

    return marginals, normalizing_constant_cluster


def conditioned_reward_matrix_bversion(R_cluster, meas_existence_mapping) -> np.ndarray:
    R = R_cluster.copy()
    n, mp1 = R.shape
    m = mp1 - 1

    meas_idxs, mask = meas_existence_mapping.T
    existing_meas = meas_idxs[mask == 1]
    nonexisting_meas = meas_idxs[mask == 0]

    R[:, nonexisting_meas] = -np.inf

    # First, normalize by misdetections. Can probably be done only once when initializing, but we do it here for now
    R[:, 1:] -= R[:, [0]]

    # Make b-version of matrix - Number of measurements that can be associated to a track or new track
    R_conditioned_bversion = np.empty((m, n + 1))
    R_conditioned_bversion[:, 1:] = R[:, 1:].T

    R_conditioned_bversion[:, 0] = 0.0  # log(1) = 0

    R_conditioned_bversion[existing_meas - 1, 0] = -np.inf  # log(0) = -inf, 0 probability of new track

    return R_conditioned_bversion


def R_LC_to_validation_likelihood_matrix(R_LC: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    likelihood_matrix = np.asfortranarray(np.exp(R_LC))
    validation_matrix = np.asfortranarray((likelihood_matrix > 0.0).astype(np.int32))

    return validation_matrix, likelihood_matrix


def b_probs_likelihood_to_a_probs_likelihood(b_probs, b_likelihood, R):
    tot_misdetection_prob =  np.exp(np.sum(R[:, 0]))
    # The b likelihood is computed based on a likelihood scaled by the total product of misdetection probs, so rescale here
    a_likelihood = b_likelihood * tot_misdetection_prob
    
    a_probs = np.empty_like(R)
    a_probs[:, 1:] = b_probs[:, 1:].T
    a_probs[:, 0] = 1.0 - a_probs[:, 1:].sum(1)

    return a_probs, a_likelihood


def multihypothesis_ehm2_meas_conditioned(R_cluster, prior_hypotheses, meas_existence_mapping, reindex_tracks: bool = True):
    if reindex_tracks:
        prior_hypotheses.reindex_tracks()  # reindex tracks to {1, 2, ..., n} for later convenience
    all_tracks_idx = np.arange(R_cluster.shape[0])
    n, mp1 = R_cluster.shape
    marginals = np.zeros((n, mp1 + 1))
    count_added = 0
    conditioned_marginals = np.empty((n, mp1 + 1))  # Needs to add nonexistence
    _0 = np.zeros((n, mp1))
    _1 = np.ones((n, 1))
    hypo_cond_normalizing_constants = np.empty(len(prior_hypotheses))

    normalizing_constant_cluster = 0.0

    for k, hypothesis in enumerate(prior_hypotheses):
        prob = hypothesis.probability()
        existing_tracks_idx = (np.array(hypothesis.tracks()) - 1).astype(int)
        if existing_tracks_idx.shape[0] > 0:
            R_sub = R_cluster[existing_tracks_idx, :]
            R_conditioned_bversion = conditioned_reward_matrix_bversion(R_sub, meas_existence_mapping)
            validation_matrix, likelihood_matrix = R_LC_to_validation_likelihood_matrix(R_conditioned_bversion)
            meas_is_gated = validation_matrix.any(axis=1)
            if not meas_is_gated.all():
                JPDAprobs = np.zeros((len(existing_tracks_idx), mp1))
                likelihood = 0.0
            else:
                b_probs, b_likelihood = EHM2.run_and_likelihood(validation_matrix, likelihood_matrix)
                JPDAprobs, likelihood = b_probs_likelihood_to_a_probs_likelihood(b_probs, b_likelihood, R_sub)
        else:
            JPDAprobs = np.zeros((0, mp1))
            # We need to check if some measurement has to be associated. If not, we use likelihood 1. Otherwise, 0
            if meas_existence_mapping[:, 1].any():
                likelihood = 0.0
            else:
                likelihood = 1.0

        # We need to concatenate the JPDAprobs with all tracks and existence probs
        non_existing_tracks_idx = np.setdiff1d(all_tracks_idx, existing_tracks_idx, assume_unique=True)

        existing_probs = np.hstack((JPDAprobs, _0[:len(existing_tracks_idx), 0, None]))
        nonexisting_probs = np.hstack((_0[:len(non_existing_tracks_idx)], _1[:len(non_existing_tracks_idx), 0, None]))

        conditioned_marginals[existing_tracks_idx] = existing_probs
        conditioned_marginals[non_existing_tracks_idx] = nonexisting_probs

        hypo_cond_normalizing_constants[k] = likelihood

        if likelihood > 0.0:
            count_added += 1
            assert np.isfinite(conditioned_marginals).all() and np.isfinite(likelihood)
            marginals += conditioned_marginals * likelihood * prob
            normalizing_constant_cluster += likelihood * prob

    if not (marginals.sum(axis=1) > 0.0).all():
        pass
        # print(marginals)
    marginals = marginals / marginals.sum(axis=1, keepdims=True)

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
                prior_hypotheses_per_cluster=self.prior_hypotheses_per_cluster,
                linking_mappings=linking_mappings
            )
            for linking_mappings in self.cluster_links.linking_mappings_per_merging_clusters()
        ]

    def compute_marginals_likelihood(self) -> np.ndarray:
        # Should in principle be straight forward at this level: simply query the marginals from each cluster/supercluster and concatenate
        n, mp1 = self.R_LC.shape

        marginals = np.empty((n, mp1 + 1))
        self._0 = np.zeros((n, mp1))
        self._1 = np.ones((n, 1))

        likelihood = 1.0
        # Collect supercluster marginals
        for supercluster in self.superclusters:
            marginals_supercluster, likelihood_supercluster = supercluster.compute_marginals_likelihood()
            t_idxs = supercluster.supercluster_t_idxs()
            if not np.isfinite(marginals_supercluster).all() or not np.isfinite(likelihood_supercluster).all():
                marginals_supercluster, likelihood_supercluster = supercluster.compute_marginals_likelihood()
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

        marginals = marginals / marginals.sum(axis=1, keepdims=True)

        return marginals, likelihood

def print_numbers_to_chars_assignment(assignment):
    assert len(assignment) == 2
    chr1 = chr(assignment[0] + ord('A')) if assignment[0] != -1 else 'N'
    chr2 = chr(assignment[1] + ord('X') - 1) if assignment[1] != -1 else 'N'
    print(f"({chr1}, {chr2})")


class ConditionalSuperclusterMarginals:
    def __init__(self, R_LC: np.ndarray, prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList, linking_mappings: LinkingMappings):
        self.R_LC = R_LC
        self.prior_hypotheses_per_cluster = prior_hypotheses_per_cluster
        self.linking_mappings = linking_mappings
        self.t_idxs = self.supercluster_t_idxs()

        # Construct conditioned clusters
        # The supercluster knows all linking measurements between the clusters, and so enumerates all and passes this enumeration to the cluster
        # Each Conditional Cluster then looks up whether its cluster received the measurement or not? We know the cluster idx, so simply name them by that

        # Before we construct the conditional clusters we need to remap the measurement idxs that exist to a more useful "array indexing index"
        self.lm2arr_idx = self.remap_linking_measurements(self.linking_mappings)

        self.conditioned_clusters: List[ConditionedCluster] = []
        for cluster, linking_measurements in linking_mappings.cluster_to_linking_measurements.items():
            prior_hypotheses = prior_hypotheses_per_cluster[cluster]
            t_idxs = prior_hypotheses.t_idxs()
            R_cluster = self.R_LC[t_idxs]

            # We actually need to know both the actual measurement idx and it's reindexed index for conditioning
            # The actual measurement idxs are used for conditioning the reward matrix, while the reindex index is used to look up the assignment mask
            # Each row has first element the actual idx while the second is the reindexed index
            linking_mappings_mapping_mat = np.array(tuple((lm, self.lm2arr_idx[lm]) for lm in linking_measurements))
            self.conditioned_clusters.append(ConditionedCluster(
                cluster_idx=cluster,
                R_cluster=R_cluster,
                prior_hypotheses=prior_hypotheses,
                linking_mappings_mapping_mat=linking_mappings_mapping_mat
            ))

    def remap_linking_measurements(self, linking_mappings: LinkingMappings) -> Dict[int, int]:
        # The simplest is probably to just make a dictionary from one idx to another?
        return {
            lm: idx for idx, lm in enumerate(linking_mappings.all_linking_measurements_idxs())
        }

    def supercluster_t_idxs(self):
        if hasattr(self, 't_idxs'):
            return self.t_idxs

        idxs = [ph.t_idxs() for c, ph in enumerate(self.prior_hypotheses_per_cluster) if c in self.linking_mappings.all_cluster_idxs()]
        return np.sort(np.hstack(idxs))

    def enumerate_meas_exist(self) -> np.ndarray:
        # Should be simply make a list for each measurement that should 
        # Needs to be careful that the measurements are enumerated in the correct column to make conditioning work

        # Premake list over assignments
        meas_assignements = [None]*len(self.lm2arr_idx)
        # Populate the list with the assignments in the correct place
        for lm, idx in self.lm2arr_idx.items():
            meas_assignements[idx] = np.array([-1, *np.fromiter(self.linking_mappings.linking_measurements_to_clusters[lm], dtype=int)])

        return cartesian_product(*meas_assignements)


    def compute_marginals_likelihood(self) -> np.ndarray:
        t_idxs = self.t_idxs
        # num_tracks = t_idxs.shape[0]
        # num_measurements = self.R_LC.shape[1] - 1

        # Let's use a lazy solution for now to avoid more index mapping hell than necessary
        n, mp1 = self.R_LC.shape
        marginals = np.empty((n , mp1 + 1))
        marginals[t_idxs] = 0.0

        # It is at this point we need to loop over all ways to assign the linking measurements, multiply the clusters conditioned marginals together
        # (since they're independent) and sum up

        # Need to figure out assignments to loop over, construct matrix first
        measurement_assignments = self.enumerate_meas_exist()

        # Preallocate marginal_term variable. We will write to all rows for each iteration, so safe to do here
        marginal_term = np.empty_like(marginals)
        likelihood = 0.0
        # Sum, loop over assignments
        for measurement_assignment in measurement_assignments:
            # Construct each term
            # Since the clusters now are independent, we simply compute the conditional marginals for each cluster and appropriately insert them into the supercluster marginal, and sum
            assignment_likelihood = 1.0
            for cluster in self.conditioned_clusters:
                conditioned_cluster_marginal, conditioned_cluster_likelihood = cluster.meas_conditioned_marginals(measurement_assignment)
                if not np.isfinite(conditioned_cluster_marginal).all() or not np.isfinite(conditioned_cluster_likelihood):
                    conditioned_cluster_marginal, conditioned_cluster_likelihood = cluster.meas_conditioned_marginals(measurement_assignment)                    
                cluster_t_idxs = cluster.t_idxs
                if conditioned_cluster_likelihood > 0.0:
                    marginal_term[cluster_t_idxs] = conditioned_cluster_marginal
                    assignment_likelihood *= conditioned_cluster_likelihood
                else:
                    assignment_likelihood = 0.0
                    break

            if assignment_likelihood > 0.0:
                marginals += marginal_term*assignment_likelihood
                likelihood += assignment_likelihood

        marginals: np.ndarray = marginals[t_idxs]
        marginals = marginals / marginals.sum(axis=1, keepdims=True)

        return marginals, likelihood


class ConditionedCluster:
    def __init__(self, cluster_idx: int, R_cluster: np.ndarray, prior_hypotheses: pdd.hypothesis.Hypotheses, linking_mappings_mapping_mat: np.ndarray):
        self.t_idxs = prior_hypotheses.t_idxs()

        self.cluster_idx = cluster_idx
        self.R_cluster: np.ndarray = R_cluster
        self.prior_hypotheses: pdd.hypothesis.Hypotheses = prior_hypotheses
        self.prior_hypotheses.reindex_tracks()
        self.linking_mappings_mapping_mat: np.ndarray = linking_mappings_mapping_mat
        self.actual_meas_idxs: np.ndarray = linking_mappings_mapping_mat[:, 0]
        self.reindex_meas: np.ndarray = linking_mappings_mapping_mat[:, 1]

        # We should definitively cache results, but not sure right now the best way. Will probably be more "obvious" later
        self.cache: Dict[Tuple[int], Tuple[np.ndarray, float]] = dict()

    def conditioned_reward_matrix(self, assigned_to_this_cluster_mask) -> np.ndarray:
        R_conditioned = self.R_cluster.copy()
 
        nonexisting_linking_meas = self.actual_meas_idxs[~assigned_to_this_cluster_mask]
        # Since column 0 is misdetection and meas idx >= 1, we can access the conditioned matrix directly with nonexisting_linking_meas
        R_conditioned[:, nonexisting_linking_meas] = -np.inf

        return R_conditioned
    

    def conditioned_reward_matrix_bversion(self, assigned_to_this_cluster_mask) -> np.ndarray:
        R = self.R_cluster.copy()
        n, mp1 = R.shape
        m = mp1 - 1

        # Get global index of measurements that need to be fixed to associating to a track
        existing_linking_meas = self.actual_meas_idxs[assigned_to_this_cluster_mask]

        # First, normalize by misdetections. Can probably be done only once when initializing, but we do it here for now
        R[:, 1:] -= R[:, [0]]

        # Make b-version of matrix - Number of measurements that can be associated to a track or new track
        R_conditioned_bversion = np.empty((m, n + 1))
        R_conditioned_bversion[:, 0] = 0.0  # log(1) = 0

        R_conditioned_bversion[:, existing_linking_meas - 1] = -np.inf  # log(0) = -inf, 0 probability of new track

        return R_conditioned_bversion

    def parse_meas_assign_to_cluster_meas_and_assign_mask(self, measurement_assignments: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Should parse the assignment, which is over the assignment of all linking measurements in the supercluster,
        # to a subarray with just the assignments of the measurements in the cluster and a boolean mask for whether each 
        # measurement is assign to this cluster or not
        this_cluster_meas_assignments = measurement_assignments[self.reindex_meas]
        assigned_to_this_cluster_mask = this_cluster_meas_assignments == self.cluster_idx
        return this_cluster_meas_assignments, assigned_to_this_cluster_mask


    def meas_conditioned_marginals(self, measurement_assignments: np.ndarray) -> np.ndarray:
        # measurement_assignments is an array num linking measurements in supercluster long
        # It might be beneficial to reindex the measurement idxs to an array indexing index for quick look-up.
        (
            this_cluster_meas_assignments,
            assigned_to_this_cluster_mask
        ) = self.parse_meas_assign_to_cluster_meas_and_assign_mask(measurement_assignments)

        meas_exist_tuple = tuple(assigned_to_this_cluster_mask)
        if meas_exist_tuple in self.cache:
            return self.cache[meas_exist_tuple]

        meas_existence_mapping = np.vstack((self.actual_meas_idxs, assigned_to_this_cluster_mask)).T

        # At this point we simply do normal computation??
        marginals_conditioned, likelihood_conditioned = multihypothesis_ehm2_meas_conditioned(self.R_cluster, self.prior_hypotheses, meas_existence_mapping=meas_existence_mapping, reindex_tracks=False)
        if not np.isfinite(marginals_conditioned).all() or not np.isfinite(likelihood_conditioned):
            marginals_conditioned, likelihood_conditioned = multihypothesis_ehm2_meas_conditioned(self.R_cluster, self.prior_hypotheses, meas_existence_mapping=meas_existence_mapping, reindex_tracks=False)

        output = (marginals_conditioned, likelihood_conditioned)
        # Cache for later
        self.cache[meas_exist_tuple] = output

        return output


if __name__ == "__main__":
    R, prior_hypotheses_per_cluster, assocLocal = test_case()
