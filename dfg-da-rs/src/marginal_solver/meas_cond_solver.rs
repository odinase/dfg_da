use crate::{
    cluster_links::ClusterLinks,
    conditional_supercluster_marginals::ConditionalSuperclusterMarginals, hypothesis::Hypotheses,
    marginal_solver::McMhAssociationMarginalOutput, utils,
};

use super::{McMhAssociationSolver, MhAssociationSolver};
use ndarray::{Array1, Array2, Axis};

use std::{collections::BTreeMap, sync::Arc};

// Uses delegating measurements to avoid merging clusters. For each cluster, run multihypothesis solver
pub struct MeasCondSolver {
    mh_solver: Arc<dyn MhAssociationSolver>,
    asso_info: AssociationInfo,
    superclusters: Vec<ConditionalSuperclusterMarginals>,
    cluster_links: ClusterLinks,
}

impl McMhAssociationSolver for MeasCondSolver {
    fn compute_marginals(&self) -> McMhAssociationMarginalOutput {
        let (n, mp1) = self.asso_info.llr.dim();

        // One extra column vs. the llr for the nonexistence hypothesis (mp1 + 1 = m + 2).
        let mut marginals = Array2::zeros((n, mp1 + 1));
        // theta posteriors keyed by cluster id; superclusters and unmerging clusters
        // are disjoint, so no key is written twice.
        let mut theta_posterior: BTreeMap<usize, Array1<f64>> = BTreeMap::new();

        let mut likelihood = 1.0;

        // Superclusters: query the already-conditioned marginals and scatter each
        // supercluster's rows into their global track indices.
        for supercluster in &self.superclusters {
            let (marginals_supercluster, theta_posteriors_supercluster, likelihood_supercluster) =
                supercluster.compute_marginals();
            for (t_cluster_idx, &t_idx) in supercluster.supercluster_t_idxs().iter().enumerate() {
                marginals
                    .row_mut(t_idx)
                    .assign(&marginals_supercluster.row(t_cluster_idx));
            }
            likelihood *= likelihood_supercluster;
            for (cluster, posterior) in theta_posteriors_supercluster {
                theta_posterior.insert(cluster, posterior);
            }
        }

        // Unmerging clusters just do total marginals over their prior hypotheses.
        for &cluster in self.cluster_links.unmerging_clusters() {
            let prior_hypotheses = self.asso_info.prior_hypotheses_per_cluster[cluster].clone();
            let t_idxs = prior_hypotheses.track_as_indices();
            let llr_cluster = self.asso_info.llr.select(Axis(0), t_idxs.as_slice());
            let mh_association_marginal_output = self
                .mh_solver
                .compute_marginals(llr_cluster.view(), &prior_hypotheses.into_reindexed());
            for (t_cluster_idx, &t_idx) in t_idxs.iter().enumerate() {
                marginals.row_mut(t_idx).assign(
                    &mh_association_marginal_output
                        .marginals()
                        .row(t_cluster_idx),
                );
            }
            likelihood *= mh_association_marginal_output.likelihood();
            theta_posterior.insert(cluster, mh_association_marginal_output.theta_posteriors);
        }

        utils::normalize_rows_inplace(&mut marginals);

        // Flatten into a per-cluster list indexed by cluster id, each posterior
        // renormalized to sum to one. Every cluster is either merging or unmerging,
        // so every slot must be present.
        let num_clusters = self.asso_info.prior_hypotheses_per_cluster.len();
        let mut theta_posteriors: Vec<Array1<f64>> = Vec::with_capacity(num_clusters);
        for cluster in 0..num_clusters {
            let mut posterior = theta_posterior
                .remove(&cluster)
                .unwrap_or_else(|| panic!("cluster {cluster} produced no theta posterior"));
            utils::normalize_rows_inplace(&mut posterior);
            theta_posteriors.push(posterior);
        }

        McMhAssociationMarginalOutput {
            marginals,
            likelihood,
            theta_posteriors,
        }
    }
}

pub struct AssociationInfo {
    llr: Array2<f64>,
    prior_hypotheses_per_cluster: Vec<Hypotheses>,
}

impl AssociationInfo {
    pub fn new(llr: Array2<f64>, prior_hypotheses_per_cluster: Vec<Hypotheses>) -> Self {
        Self {
            llr,
            prior_hypotheses_per_cluster,
        }
    }
}

impl MeasCondSolver {
    pub fn new(
        asso_info: AssociationInfo,
        mh_solver: Arc<dyn MhAssociationSolver>,
        cluster_links: ClusterLinks,
    ) -> Self {
        let superclusters = cluster_links
            .linking_mappings_per_merging_clusters()
            .iter()
            .map(|linking_mappings| {
                ConditionalSuperclusterMarginals::new(
                    asso_info.llr.view(),
                    asso_info.prior_hypotheses_per_cluster.as_slice(),
                    linking_mappings,
                    Arc::clone(&mh_solver),
                )
            })
            .collect();

        Self {
            mh_solver,
            asso_info,
            superclusters,
            cluster_links,
        }
    }
}
