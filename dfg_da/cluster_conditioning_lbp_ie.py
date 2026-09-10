import numpy as np
from .cluster_bayes_tree import ClusterLinks, LinkingMappings, cartesian_product
from .marginals_computers import LBPMarginalsByTotalProbBethe, LBPMarginalsByTotalProbPHD
from .stats_logger import MulticlusterConditionendLBPOutput
import py_dfg_da as pdd
from typing import *
from collections import defaultdict

import time
import warnings

# We can now construct the components of the "tree" (with depth 1 lol).
# We use ClusterLinks above to find the measurements that are linked to other clusters
# It should be initialized with a multicluster case and separate the clusters that are merged with the unaffected clusters
# It is probably the best to do this in two stages - One that separates clusters and delegates the computation, and one that the performs that supercluster marginal computation
class MulticlusterEfficientMarginalsLBPInclusionExclusion:
    def __init__(self, R_LC, prior_hypotheses_per_cluster, assocLocal, lbp_solver, cluster_links: Optional[ClusterLinks] = None):
        # Store input for convenience
        self.R_LC = R_LC
        self.prior_hypotheses_per_cluster = prior_hypotheses_per_cluster
        self.assocLocal = assocLocal

        # Compute linking measurements and superclusters
        if cluster_links is not None:
            self.cluster_links = cluster_links
        else:
            self.cluster_links = ClusterLinks(R_LC=R_LC, prior_hypotheses_per_cluster=prior_hypotheses_per_cluster, assocLocal=assocLocal)

        # Make superclusters
        self.superclusters = [
            ConditionalSuperclusterMarginals(
                R_LC=R_LC,
                prior_hypotheses_per_cluster=self.prior_hypotheses_per_cluster,
                linking_mappings=linking_mappings,
                lbp_solver=lbp_solver
            )
            for linking_mappings in self.cluster_links.linking_mappings_per_merging_clusters()
        ]

        self.lbp = lbp_solver

    def compute_marginals_likelihood(self) -> MulticlusterConditionendLBPOutput:
        # Should in principle be straight forward at this level: simply query the marginals from each cluster/supercluster and concatenate
        # Solve-only timing. The driver stamps the fair total (construction + solve) onto
        # .runtime; the difference between the two is this method's setup cost.
        _t0 = time.perf_counter()
        n, mp1 = self.R_LC.shape

        marginals = np.empty((n, mp1 + 1))
        theta_posteriors = dict()
        self._0 = np.zeros((n, mp1))
        self._1 = np.ones((n, 1))

        likelihood = 1.0
        runtime_warning = False
        # Collect supercluster marginals
        with warnings.catch_warnings(record=True) as caught_warnings:
            for supercluster in self.superclusters:
                marginals_supercluster, theta_posteriors_supercluster, likelihood_supercluster = supercluster.compute_marginals_likelihood()
                t_idxs = supercluster.supercluster_t_idxs()
                marginals[t_idxs] = marginals_supercluster
                likelihood *= likelihood_supercluster
                theta_posteriors.update(theta_posteriors_supercluster)

            # Unmerging clusters just do total marginals over hypotheses
            for cluster in self.cluster_links.unmerging_clusters():
                prior_hypotheses = self.prior_hypotheses_per_cluster[cluster]
                t_idxs = np.sort(np.fromiter(prior_hypotheses.tracks(), dtype=int)) - 1
                R_cluster = self.R_LC[t_idxs]
                prior_hypotheses.reindex_tracks()
                marginals_unmerged, theta_posterior_cluster, likelihood_unmerged = self.lbp(R_cluster, prior_hypotheses)
                marginals[t_idxs] = marginals_unmerged
                likelihood *= likelihood_unmerged
                theta_posteriors[cluster] = theta_posterior_cluster

            marginals = marginals / marginals.sum(axis=1, keepdims=True)

            theta_posteriors_list = [None]*len(theta_posteriors)
            for prior_c in theta_posteriors:
                theta_posteriors_list[prior_c] = theta_posteriors[prior_c] / theta_posteriors[prior_c].sum()

            if caught_warnings:
                for warning in caught_warnings:
                    if issubclass(warning.category, RuntimeWarning):
                        runtime_warning = True

        return MulticlusterConditionendLBPOutput(
            marginals=marginals,
            likelihood=likelihood,
            theta_posteriors=theta_posteriors_list,
            raised_warning=runtime_warning,
            solve_runtime=time.perf_counter() - _t0
        )

def print_numbers_to_chars_assignment(assignment):
    assert len(assignment) == 2
    chr1 = chr(assignment[0] + ord('A')) if assignment[0] != -1 else 'N'
    chr2 = chr(assignment[1] + ord('X') - 1) if assignment[1] != -1 else 'N'
    print(f"({chr1}, {chr2})")


class ConditionalSuperclusterMarginals:
    def __init__(self, R_LC: np.ndarray, prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList, linking_mappings: LinkingMappings, lbp_solver):
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
                linking_mappings_mapping_mat=linking_mappings_mapping_mat,
                lbp_solver=lbp_solver
            ))

        # Compute number of competing clusters for each measurment
        self.num_competing_clusters_per_lmk_arr_idx = np.empty(len(self.lm2arr_idx))
        for lnk_meas_idx, clusters in self.linking_mappings.linking_measurements_to_clusters.items():
            num_competing_lnk_meas_idx = len(clusters)
            self.num_competing_clusters_per_lmk_arr_idx[self.lm2arr_idx[lnk_meas_idx]] = num_competing_lnk_meas_idx

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

        theta_posteriors = defaultdict(lambda: 0)
        theta_posteriors_term = defaultdict(lambda: 0)
        # It is at this point we need to loop over all ways to assign the linking measurements, multiply the clusters conditioned marginals together
        # (since they're independent) and sum up

        # Need to figure out assignments to loop over, construct matrix first
        measurement_assignments = self.enumerate_meas_exist()

        # Preallocate marginal_term variable. We will write to all rows for each iteration, so safe to do here
        marginal_term = np.empty_like(marginals)
        likelihood = 0.0
        # Sum, loop over assignments
        for measurement_assignment in measurement_assignments:
            weight = np.prod(1 - self.num_competing_clusters_per_lmk_arr_idx[measurement_assignment == -1])
            # Construct each term
            # Since the clusters now are independent, we simply compute the conditional marginals for each cluster and appropriately insert them into the supercluster marginal, and sum
            assignment_likelihood = 1.0
            valid = True
            for cluster in self.conditioned_clusters:
                conditioned_cluster_marginal, conditioned_theta_posterior, conditioned_cluster_likelihood = cluster.meas_conditioned_marginals(measurement_assignment)
                cluster_t_idxs = cluster.t_idxs
                if conditioned_cluster_likelihood > 0.0:
                    marginal_term[cluster_t_idxs] = conditioned_cluster_marginal
                    assignment_likelihood *= conditioned_cluster_likelihood
                    theta_posteriors_term[cluster.cluster_idx] = conditioned_theta_posterior
                else:
                    valid = False
                    break

            if not valid:
                continue

            assignment_likelihood *= weight

            marginals += marginal_term*assignment_likelihood
            likelihood += assignment_likelihood
            for c, p in theta_posteriors_term.items():
                theta_posteriors[c] += p * assignment_likelihood

        marginals: np.ndarray = marginals[t_idxs]
        marginals = marginals / marginals.sum(axis=1, keepdims=True)
        for c, p in theta_posteriors.items():
            theta_posteriors[c] = p / p.sum()

        return marginals, theta_posteriors, likelihood


class ConditionedCluster:
    def __init__(self, cluster_idx: int, R_cluster: np.ndarray, prior_hypotheses: pdd.hypothesis.Hypotheses, linking_mappings_mapping_mat: np.ndarray, lbp_solver = LBPMarginalsByTotalProbBethe()):
        self.t_idxs = prior_hypotheses.t_idxs()

        self.cluster_idx = cluster_idx
        self.R_cluster: np.ndarray = R_cluster
        self.prior_hypotheses: pdd.hypothesis.Hypotheses = prior_hypotheses
        self.prior_hypotheses.reindex_tracks()
        self.linking_mappings_mapping_mat: np.ndarray = linking_mappings_mapping_mat
        self.actual_meas_idxs: np.ndarray = linking_mappings_mapping_mat[:, 0]
        self.reindex_meas: np.ndarray = linking_mappings_mapping_mat[:, 1]

        self.lbp = lbp_solver

        # We should definitively cache results, but not sure right now the best way. Will probably be more "obvious" later
        self.cache: Dict[Tuple[int], Tuple[np.ndarray, float]] = dict()

    def conditioned_reward_matrix(self, assigned_to_this_cluster_mask) -> np.ndarray:
        R_conditioned = self.R_cluster.copy()
 
        nonexisting_linking_meas = self.actual_meas_idxs[~assigned_to_this_cluster_mask]
        # Since column 0 is misdetection and meas idx >= 1, we can access the conditioned matrix directly with nonexisting_linking_meas
        R_conditioned[:, nonexisting_linking_meas] = -np.inf

        return R_conditioned
    

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

        R_conditioned = self.conditioned_reward_matrix(assigned_to_this_cluster_mask)

        # At this point we simply do normal computation??
        lbp_marginal_total, conditioned_theta_posterior, likelihood_conditioned = self.lbp(R_conditioned, self.prior_hypotheses)

        output = (lbp_marginal_total, conditioned_theta_posterior, likelihood_conditioned)
        # Cache for later
        self.cache[meas_exist_tuple] = output

        return output
