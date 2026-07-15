use itertools::Itertools;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;

use crate::cluster_links as cl;
use crate::hypothesis as hyp;
use crate::marginal_solver as ms;
use ndarray as nd;
use ndarray::prelude::*;
use std::iter::once;

pub struct ConditionalSuperclusterMarginals {
    num_tracks: usize,
    num_measurements: usize,
    t_idxs: BTreeSet<usize>,
    lm2arr_idx: HashMap<usize, usize>,
    conditioned_clusters: Vec<ConditionedCluster>,
    num_competing_clusters_per_lm_array_idx: BTreeMap<usize, usize>,
    linking_mappings: cl::LinkingMappings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementDelegation {
    Delegated(usize),
    NotDelegated,
}

use MeasurementDelegation as MD;

impl MeasurementDelegation {
    pub fn is_not_delegated(&self) -> bool {
        self == &Self::NotDelegated
    }
}

fn compute_supercluster_track_indices(
    prior_hypotheses_per_cluster: &[hyp::Hypotheses],
    linking_mappings: &cl::LinkingMappings,
) -> BTreeSet<usize> {
    linking_mappings
        .all_cluster_idxs()
        .iter()
        .flat_map(|&c| prior_hypotheses_per_cluster[c].all_tracks())
        .map(|t| t - 1)
        .collect()
}

fn remap_linking_measurements(linking_mappings: &cl::LinkingMappings) -> HashMap<usize, usize> {
    linking_mappings
        .all_linking_measurement_idxs()
        .iter()
        .enumerate()
        .map(|(k, &lm)| (lm, k))
        .collect()
}

impl ConditionalSuperclusterMarginals {
    pub fn new(
        llr: ArrayView2<f64>,
        prior_hypotheses_per_cluster: &[hyp::Hypotheses],
        linking_mappings: &cl::LinkingMappings,
        marginal_solver: Rc<dyn ms::MhAssociationSolver>,
    ) -> Self {
        let (n, mp1) = llr.dim();
        let num_tracks = n;
        let num_measurements = mp1 - 1;

        let t_idxs =
            compute_supercluster_track_indices(prior_hypotheses_per_cluster, linking_mappings);

        let lm2arr_idx = remap_linking_measurements(linking_mappings);

        // Global track idx -> compact row in the supercluster marginals array.
        // t_idxs is sorted (BTreeSet), so the rank in the set is the row.
        let t_idx_to_row: HashMap<usize, usize> = t_idxs
            .iter()
            .enumerate()
            .map(|(row, &t)| (t, row))
            .collect();

        let conditioned_clusters = linking_mappings
            .cluster_to_linking_measurements()
            .iter()
            .map(|(&cluster, linking_measurements)| {
                let prior_hypotheses = &prior_hypotheses_per_cluster[cluster];
                let t_idxs: Vec<_> = prior_hypotheses.track_as_indices();
                let llr_cluster = llr.select(Axis(0), t_idxs.as_slice());

                // Row i of this cluster's marginal output goes into row
                // supercluster_rows[i] of the compact supercluster array.
                let supercluster_rows = t_idxs.iter().map(|t| t_idx_to_row[t]).collect();

                let (actual_meas_idxs, reindex_meas) = linking_measurements
                    .iter()
                    .copied()
                    .map(|lm| (lm, lm2arr_idx[&lm]))
                    .unzip();

                ConditionedCluster::new(
                    cluster,
                    llr_cluster,
                    prior_hypotheses.clone(),
                    Rc::clone(&marginal_solver),
                    actual_meas_idxs,
                    reindex_meas,
                    supercluster_rows,
                )
            })
            .collect();

        let num_competing_clusters_per_lm_array_idx = linking_mappings
            .linking_measurement_to_clusters()
            .iter()
            .map(|(lnk_meas_idx, clusters)| {
                let num_competing_meas_idx = clusters.len();
                (lm2arr_idx[lnk_meas_idx], num_competing_meas_idx)
            })
            .collect();

        ConditionalSuperclusterMarginals {
            num_tracks,
            num_measurements,
            t_idxs,
            lm2arr_idx,
            conditioned_clusters,
            num_competing_clusters_per_lm_array_idx,
            linking_mappings: linking_mappings.clone(),
        }
    }

    // Lazily built domain of each linking measurement: unassigned or delegated to one
    // of its competing clusters. The BTreeMap yields clusters in ascending lm order,
    // which matches the lm array idx order assigned by remap_linking_measurements, so
    // the domain at position k belongs to the measurement with array idx k.
    fn assignment_domains(
        &self,
    ) -> impl Iterator<Item = impl Iterator<Item = MeasurementDelegation> + Clone + '_> + '_ {
        self.linking_mappings
            .linking_measurement_to_clusters()
            .values()
            .map(|clusters| {
                once(MD::NotDelegated).chain(clusters.iter().copied().map(MD::Delegated))
            })
    }

    /// Sorted global track indices covered by this supercluster; row k of the marginals
    /// returned by [`Self::compute_marginals`] corresponds to the k-th index here.
    pub fn supercluster_t_idxs(&self) -> &BTreeSet<usize> {
        &self.t_idxs
    }

    pub fn compute_marginals(&self) -> (Array2<f64>, BTreeMap<usize, Vec<f64>>, f64) {
        // Compact allocation: only the supercluster's tracks, in sorted t_idxs order.
        let num_supercluster_tracks = self.t_idxs.len();
        let cols = self.num_measurements + 2; // misdetection + measurements + nonexistence

        let mut marginals = Array2::<f64>::zeros((num_supercluster_tracks, cols));
        // All rows are written each accepted iteration, so reuse across iterations.
        let mut marginal_term = Array2::<f64>::zeros((num_supercluster_tracks, cols));
        let mut theta_posteriors: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        let mut likelihood = 0.0;

        // Sum over all ways to delegate the linking measurements. Conditioned on a
        // delegation the clusters are independent, so each term is a product of
        // per-cluster conditional marginals. Undelegated measurements would be double
        // counted across delegations, so each term carries an inclusion-exclusion
        // weight of (1 - #competing clusters) per undelegated measurement.
        'measurement_assignment: for measurement_assignment in
            self.assignment_domains().multi_cartesian_product()
        {
            let weight: f64 = measurement_assignment
                .iter()
                .enumerate()
                .filter(|(_, delegation)| delegation.is_not_delegated())
                .map(|(arr_idx, _)| {
                    1.0 - self.num_competing_clusters_per_lm_array_idx[&arr_idx] as f64
                })
                .product();

            let mut assignment_likelihood = 1.0;
            // let mut valid = true;
            let mut cluster_outputs = Vec::with_capacity(self.conditioned_clusters.len());
            for cluster in &self.conditioned_clusters {
                let output = cluster.meas_conditioned_marginals(&measurement_assignment);
                if output.likelihood() > 0.0 {
                    assignment_likelihood *= output.likelihood();
                    cluster_outputs.push((cluster, output));
                } else {
                    // valid = false;
                    // break;
                    break 'measurement_assignment;
                }
            }

            // if !valid {
            //     continue;
            // }

            // The weight can be negative (inclusion-exclusion), so accumulate
            // unconditionally once the assignment is valid.
            assignment_likelihood *= weight;

            for (cluster, output) in &cluster_outputs {
                let cluster_marginals = output.marginals();
                for (i, &row) in cluster.supercluster_rows.iter().enumerate() {
                    marginal_term.row_mut(row).assign(&cluster_marginals.row(i));
                }

                let term = output.theta_posteriors();
                let acc = theta_posteriors
                    .entry(cluster.cluster_idx)
                    .or_insert_with(|| vec![0.0; term.len()]);
                for (a, &p) in acc.iter_mut().zip(term.iter()) {
                    *a += p * assignment_likelihood;
                }
            }

            marginals.scaled_add(assignment_likelihood, &marginal_term);
            likelihood += assignment_likelihood;
        }

        // Normalize marginals row-wise and theta posteriors to sum to one.
        for mut row in marginals.rows_mut() {
            let row_sum = row.sum();
            row /= row_sum;
        }
        for posterior in theta_posteriors.values_mut() {
            let sum: f64 = posterior.iter().sum();
            for p in posterior.iter_mut() {
                *p /= sum;
            }
        }

        (marginals, theta_posteriors, likelihood)
    }
}

struct ConditionedCluster {
    t_idxs: Vec<usize>,
    cluster_idx: usize,
    llr_cluster: Array2<f64>,
    prior_hypotheses: hyp::Hypotheses,
    marginal_solver: Rc<dyn ms::MhAssociationSolver>,
    actual_meas_idxs: Vec<usize>,
    reindex_meas: Vec<usize>,
    // Row i of this cluster's marginal output belongs in row supercluster_rows[i]
    // of the compact supercluster marginals array.
    supercluster_rows: Vec<usize>,
    // TODO: Figure out a good type to use here
    // Interior mutability is valid here since it doesnt change the logical behavior of the struct
    cache: RefCell<HashMap<Vec<bool>, ms::MhAssociationMarginalOutput>>,
}

impl ConditionedCluster {
    fn new(
        cluster_idx: usize,
        llr_cluster: Array2<f64>,
        prior_hypotheses: hyp::Hypotheses,
        marginal_solver: Rc<dyn ms::MhAssociationSolver>,
        actual_meas_idxs: Vec<usize>,
        reindex_meas: Vec<usize>,
        supercluster_rows: Vec<usize>,
    ) -> Self {
        let t_idxs = prior_hypotheses.track_as_indices();
        let prior_hypotheses = prior_hypotheses.into_reindexed();
        Self {
            t_idxs,
            cluster_idx,
            llr_cluster,
            prior_hypotheses,
            marginal_solver,
            actual_meas_idxs,
            reindex_meas,
            supercluster_rows,
            cache: RefCell::new(HashMap::new()),
        }
    }

    fn parse_meas_assign_to_assign_mask(
        &self,
        measurement_assignments: &[MeasurementDelegation],
    ) -> Vec<bool> {
        self.reindex_meas
            .iter()
            .map(|&idx| measurement_assignments[idx] == MD::Delegated(self.cluster_idx))
            .collect()
    }

    fn conditioned_reward_matrix(&self, assigned_to_this_cluster_mask: &[bool]) -> Array2<f64> {
        let mut llr_conditioned = self.llr_cluster.clone();

        let removed_measurements = assigned_to_this_cluster_mask
            .iter()
            .zip(self.actual_meas_idxs.iter())
            .filter_map(|(assigned, meas_idx)| if !*assigned { Some(*meas_idx) } else { None });

        for col in removed_measurements {
            llr_conditioned.column_mut(col).fill(f64::NEG_INFINITY);
        }

        llr_conditioned
    }

    fn meas_conditioned_marginals(
        &self,
        measurement_assignments: &[MeasurementDelegation],
    ) -> ms::MhAssociationMarginalOutput {
        let assign_mask = self.parse_meas_assign_to_assign_mask(measurement_assignments);
        if let Some(cached_mh_asso_output) = self.cache.borrow().get(&assign_mask) {
            return cached_mh_asso_output.clone();
        }

        let llr_conditioned = self.conditioned_reward_matrix(assign_mask.as_slice());
        let mh_asso_output = self
            .marginal_solver
            .compute_marginals(llr_conditioned.view(), &self.prior_hypotheses);

        self.cache
            .borrow_mut()
            .insert(assign_mask, mh_asso_output.clone());

        mh_asso_output
    }
}
