use itertools::Itertools;
use std::collections::BTreeSet;
use std::collections::HashMap;

use crate::cluster_links as cl;
use crate::hypothesis as hyp;
use crate::marginal_solver as ms;
use ndarray as nd;
use ndarray::prelude::*;

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

            });
    }
}
