use crate::{
    cluster::Cluster, cluster_links::ClusterLinks, hypothesis::Hypotheses,
    marginal_solver::McMhAssociationMarginalOutput,
};

use super::{McMhAssociationSolver, MhAssociationSolver};
use ndarray::{Array2, ArrayView2, ShapeArg};

use std::{collections::{BTreeSet, HashMap, HashSet}, rc::Rc};

// Uses delegating measurements to avoid merging clusters. For each cluster, run multihypothesis solver
pub struct MeasCondSolver {
    mh_solver: Rc<dyn MhAssociationSolver>,
    asso_info: AssociationInfo,
}

impl McMhAssociationSolver for MeasCondSolver {
    fn compute_marginals(&self) -> McMhAssociationMarginalOutput {
        todo!()
    }
}

pub struct AssociationInfo {
    llr: Array2<f64>,
    prior_hypotheseses_per_cluster: Vec<Hypotheses>,
    clusters_links: ClusterLinks,
}

impl MeasCondSolver {
    pub fn new(asso_info: AssociationInfo, mh_solver: Rc<dyn MhAssociationSolver>, cluster_links: ClusterLinks) -> Self {
        Self {
            asso_info,
            mh_solver,
        }
    }
}
