pub mod lbp;
pub mod mcmh_lbp;
pub mod meas_cond_solver;
pub mod mh_lbp;

use ndarray::Array2;

pub struct McMhAssociationMarginalOutput {
    cluster_marginals: Vec<MhAssociationMarginalOutput>,
}

// Association solver that works on multiple clusters and multiple hypotheseses
pub trait McMhAssociationSolver {
    fn compute_marginals(&self) -> McMhAssociationMarginalOutput;
}

pub struct MhAssociationMarginalOutput {
    marginals: AssociationMarginalOutput,
    theta_posteriors: Vec<f64>,
}

// Association solver that works on a single cluster and multiple hypotheseses
pub trait MhAssociationSolver {
    fn compute_marginals(&self) -> MhAssociationMarginalOutput;
}

pub struct AssociationMarginalOutput {
    marginals: Array2<f64>,
    likelihood: f64,
}

// Association solver that works on a single cluster and a single hypothesis
pub trait AssociationSolver {
    fn compute_marginals(&self) -> AssociationMarginalOutput;
}
