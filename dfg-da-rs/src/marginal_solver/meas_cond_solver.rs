use crate::{cluster::Cluster, hypothesis::Hypotheses};

use super::{McMhAssociationSolver, MhAssociationSolver};
use ndarray::{Array2, ArrayView2, ShapeArg};

use std::collections::{HashMap, HashSet};

// Uses delegating measurements to avoid merging clusters. For each cluster, run multihypothesis solver
pub struct MeasCondSolver {
    mh_solver: Box<dyn MhAssociationSolver>,
    asso_info: AssociationInfo,
}

impl McMhAssociationSolver for MeasCondSolver {
    fn compute_marginals(&self) -> super::McMhAssociationMarginalOutput {}
}

pub struct AssociationInfo {
    llr: Array2<f64>,
    prior_hypotheseses_per_cluster: Vec<Hypotheses>,
    clusters_links: ClusterLinks,
}

impl MeasCondSolver {
    pub fn new(asso_info: AssociationInfo, mh_solver: Box<dyn MhAssociationSolver>) -> Self {
        Self {
            asso_info,
            mh_solver,
        }
    }
}

pub struct ClusterLinks {}

fn unique_with_counts(data: &[u8]) -> HashMap<u8, usize> {
    let mut counts: HashMap<u8, usize> = HashMap::new();
    for &x in data {
        *counts.entry(x).or_insert(0) += 1;
    }
    counts
}

fn find_merging_clusters(assoc_local: &ArrayView2<u8>) -> Vec<HashSet<usize>> {
    let cluster_idxs_might_merge = &assoc_local.row(0) - 1;
    let master_clusters_with_counts =
        unique_with_counts(cluster_idxs_might_merge.as_slice().unwrap());
    let merging_masters: Vec<_> = master_clusters_with_counts
        .iter()
        .filter_map(|(&c_idx, &count)| if count > 1 { Some(c_idx) } else { None })
        .collect();
    let merging_clusters: Vec<HashSet<_>> = merging_masters
        .iter()
        .map(|&merging_master| {
            cluster_idxs_might_merge
                .iter()
                .enumerate()
                .filter_map(|(idx, &cidx)| {
                    if cidx == merging_master {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect();

    merging_clusters
}

fn get_tracks_per_merging_cluster(
    merging_clusters: &[HashSet<usize>],
    prior_hypothses_per_cluster: &[Hypotheses],
) -> Vec<Vec<usize>> {
    let relevant_clusters: HashSet<_> = merging_clusters
        .iter().flatten().collect();
    
           
}

impl ClusterLinks {
    pub fn from_parsed_mat_file(
        llr: &ArrayView2<'_, f64>,
        prior_hypotheses_per_cluster: &[Hypotheses],
        assoc_local: &Array2<u8>,
    ) -> Self {
        let all_clusters: HashSet<_> = (0..assoc_local.dim().1).collect();
        let merging_clusters = find_merging_clusters(&assoc_local.view());
        let tracks_per_merging_cluster = get_tracks_per_merging_cluster(
            merging_clusters.as_slice(),
            prior_hypotheses_per_cluster,
        );
    }
}
