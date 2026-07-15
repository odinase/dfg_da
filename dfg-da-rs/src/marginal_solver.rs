pub mod lbp;
pub mod mcmh_lbp;
pub mod meas_cond_solver;
pub mod mh_lbp;

use crate::hypothesis as hyp;
use ndarray::{Array2, ArrayView2};

#[derive(Debug, Clone)]
pub struct McMhAssociationMarginalOutput {
    pub(crate) cluster_marginals: Vec<MhAssociationMarginalOutput>,
}

// Association solver that works on multiple clusters and multiple hypotheseses
pub trait McMhAssociationSolver {
    fn compute_marginals(&self) -> McMhAssociationMarginalOutput;
}

#[derive(Debug, Clone)]
pub struct MhAssociationMarginalOutput {
    pub(crate) marginals: AssociationMarginalOutput,
    pub(crate) theta_posteriors: Vec<f64>,
}

impl MhAssociationMarginalOutput {
    pub fn marginals(&self) -> &Array2<f64> {
        &self.marginals.marginals
    }

    pub fn likelihood(&self) -> f64 {
        self.marginals.likelihood
    }

    pub fn theta_posteriors(&self) -> &[f64] {
        &self.theta_posteriors
    }
}

// Association solver that works on a single cluster and multiple hypotheseses
pub trait MhAssociationSolver {
    fn compute_marginals(&self, llr: ArrayView2<f64>, prior_hypotheses: &hyp::Hypotheses) -> MhAssociationMarginalOutput;
}

#[derive(Debug, Clone)]
pub struct AssociationMarginalOutput {
    pub(crate) marginals: Array2<f64>,
    pub(crate) likelihood: f64,
}

// Association solver that works on a single cluster and a single hypothesis
pub trait AssociationSolver {
    fn compute_marginals(&self) -> AssociationMarginalOutput;
}
