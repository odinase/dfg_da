pub mod hyp_cond_solver;
pub mod lbp;
pub mod mcmh_lbp;
pub mod meas_cond_solver;
pub mod mh_lbp;

use crate::hypothesis as hyp;
use ndarray::prelude::*;
use ndarray::{Array2, ArrayView2};

// Multicluster, multihypothesis marginals over the whole association problem.
// Mirrors the Python `MulticlusterConditionendLBPOutput`: a single concatenated
// marginals matrix, a scalar likelihood, and per-cluster theta posteriors indexed
// by cluster id. Every cluster is either merging or unmerging, so every slot is
// always filled (unlike the Python list, which defaults entries to None).
#[derive(Debug, Clone)]
pub struct McMhAssociationMarginalOutput {
    pub(crate) marginals: Array2<f64>,
    pub(crate) likelihood: f64,
    pub(crate) theta_posteriors: Vec<Array1<f64>>,
}

impl McMhAssociationMarginalOutput {
    pub fn marginals(&self) -> ArrayView2<'_, f64> {
        self.marginals.view()
    }

    pub fn likelihood(&self) -> f64 {
        self.likelihood
    }

    pub fn theta_posteriors(&self) -> &[Array1<f64>] {
        &self.theta_posteriors
    }
}

// Association solver that works on multiple clusters and multiple hypotheseses
pub trait McMhAssociationSolver: Send + Sync {
    fn compute_marginals(&self) -> McMhAssociationMarginalOutput;
}

#[derive(Debug, Clone)]
pub struct MhAssociationMarginalOutput {
    pub(crate) marginals: AssociationMarginalOutput,
    pub(crate) theta_posteriors: Array1<f64>,
}

impl MhAssociationMarginalOutput {
    pub fn marginals(&self) -> ArrayView2<'_, f64> {
        self.marginals.marginals.view()
    }

    pub fn likelihood(&self) -> f64 {
        self.marginals.likelihood
    }

    pub fn theta_posteriors(&self) -> ArrayView1<'_, f64> {
        self.theta_posteriors.view()
    }
}

// Association solver that works on a single cluster and multiple hypotheseses
pub trait MhAssociationSolver: Send + Sync {
    fn compute_marginals(
        &self,
        llr: ArrayView2<f64>,
        prior_hypotheses: &hyp::Hypotheses,
    ) -> MhAssociationMarginalOutput;
}

#[derive(Debug, Clone)]
pub struct AssociationMarginalOutput {
    pub(crate) marginals: Array2<f64>,
    pub(crate) likelihood: f64,
}

// Association solver that works on a single cluster and a single hypothesis
pub trait AssociationSolver: Send + Sync {
    fn compute_marginals(&self, llr: ArrayView2<f64>) -> AssociationMarginalOutput;
}
