use itertools::Itertools;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;

use crate::cluster_links as cl;
use crate::hypothesis as hyp;
use crate::marginal_solver as ms;
use ndarray as nd;
use ndarray::prelude::*;
use std::cell::Cell;

pub struct ConditionalSuperclusterMarginals {
    num_tracks: usize,
    num_measurements: usize,
}

fn compute_supercluster_track_indices(
    prior_hypotheses_per_cluster: &[hyp::Hypotheses],
    linking_mappings: &cl::LinkingMappings,
) -> BTreeSet<usize> {
    let clusters_in_supercluster = linking_mappings.all_cluster_idxs();
    prior_hypotheses_per_cluster
        .iter()
        .map(|h| h.all_tracks())
        .flatten()
        .filter_map(|t| {
            if clusters_in_supercluster.contains(&t) {
                Some(t - 1)
            } else {
                None
            }
        })
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
            .zip(prior_hypotheses_per_cluster.iter())
            .map(|((cluster, linking_measurements), prior_hypotheses)| {
                let t_idxs: Vec<_> = prior_hypotheses.track_as_indices();
                let llr_cluster = llr.select(Axis(0), t_idxs.as_slice());

                let (actual_meas_idxs, reindex_meas) = linking_measurements
                    .iter()
                    .copied()
                    .map(|lm| (lm, lm2arr_idx[&lm]))
                    .unzip();
            });
    }
}

struct ConditionedCluster {
    cluster_idx: usize,
    llr_cluster: Array2<f64>,
    prior_hypotheses: hyp::Hypotheses,
    marginal_solver: Rc<dyn ms::MhAssociationSolver>,
    actual_meas_idxs: Vec<usize>,
    reindex_meas: Vec<usize>,
    cache: RefCell<HashMap<Vec<bool>, ms::MhAssociationMarginalOutput>>, // TODO: Figure out a good type to use here
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

    fn parse_meas_assign_to_assign_mask(&self, measurement_assigments: &[usize]) -> Vec<bool> {
        let assigned_to_this_cluster_mask = self
            .reindex_meas
            .iter()
            .map(|idx| measurement_assigments[*idx] == self.cluster_idx)
            .collect();

        assigned_to_this_cluster_mask
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
        measurement_assigments: &[usize],
    ) -> ms::MhAssociationMarginalOutput {
        let assign_mask = self.parse_meas_assign_to_assign_mask(measurement_assigments);
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
