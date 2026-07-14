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

        let conditioned_clusters = linking_mappings
            .cluster_to_linking_measurements()
            .iter()
            .map(|(&cluster, linking_measurements)| {
                let prior_hypotheses = &prior_hypotheses_per_cluster[cluster];
                let t_idxs: Vec<_> = prior_hypotheses.track_as_indices();
                let llr_cluster = llr.select(Axis(0), t_idxs.as_slice());

                let (actual_meas_idxs, reindex_meas) = linking_measurements
                    .iter()
                    .copied()
                    .map(|lm| (lm, lm2arr_idx[&lm]))
                    .unzip();

                ConditionedCluster::new(
                    cluster,
                    llr_cluster,
                    prior_hypotheses.clone().into_reindexed(),
                    Rc::clone(&marginal_solver),
                    actual_meas_idxs,
                    reindex_meas,
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
        }
    }

fn assignment_domains<'a>(
    &'a self,
    linking_mappings: &'a cl::LinkingMappings,
) -> impl Iterator<Item = impl Iterator<Item = MeasurementDelegation> + Clone + 'a> + 'a {
    linking_mappings
        .linking_measurement_to_clusters()
        .iter()   // BTreeMap: yields (&lm, &clusters) in ascending lm order
        .map(|(_lm, clusters)| {
            once(MD::NotDelegated).chain(clusters.iter().copied().map(MD::Delegated))
        })
}

    fn compute_marginals(&self) -> (Array2<f64>, HashMap<usize, Vec<f64>, f64>) {
        
    }
}

struct ConditionedCluster {
    cluster_idx: usize,
    llr_cluster: Array2<f64>,
    prior_hypotheses: hyp::Hypotheses,
    marginal_solver: Rc<dyn ms::MhAssociationSolver>,
    actual_meas_idxs: Vec<usize>,
    reindex_meas: Vec<usize>,
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
    ) -> Self {
        Self {
            cluster_idx,
            llr_cluster,
            prior_hypotheses,
            marginal_solver,
            actual_meas_idxs,
            reindex_meas,
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

struct ConditionedClusterMarginalOutput {}
