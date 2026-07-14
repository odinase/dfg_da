pub mod lbp;
pub mod mcmh_lbp;
pub mod meas_cond_solver;
pub mod mh_lbp;

use crate::hypothesis as hyp;
use ndarray::{Array2, ArrayView2};

#[derive(Debug, Clone)]
pub struct McMhAssociationMarginalOutput {
    cluster_marginals: Vec<MhAssociationMarginalOutput>,
}

// Association solver that works on multiple clusters and multiple hypotheseses
pub trait McMhAssociationSolver {
    fn compute_marginals(&self) -> McMhAssociationMarginalOutput;
}

#[derive(Debug, Clone)]
pub struct MhAssociationMarginalOutput {
    marginals: AssociationMarginalOutput,
    theta_posteriors: Vec<f64>,
}

// Association solver that works on a single cluster and multiple hypotheseses
pub trait MhAssociationSolver {
    fn compute_marginals(&self, llr: ArrayView2<f64>, prior_hypotheses: &hyp::Hypotheses) -> MhAssociationMarginalOutput;
}

#[derive(Debug, Clone)]
pub struct AssociationMarginalOutput {
    marginals: Array2<f64>,
    likelihood: f64,
}

// Association solver that works on a single cluster and a single hypothesis
pub trait AssociationSolver {
    fn compute_marginals(&self) -> AssociationMarginalOutput;
}
