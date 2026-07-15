use crate::{
    marginal_solver as ms,
    hypothesis as hyp
};
use ndarray::{prelude::*, self as nd};
use std::rc;

// Single cluster solver that takes in a single hypothesis solver and conditiones over prior hypotheses and computes the marginals with total probability
pub struct HypCondSolver {
    solver: rc::Rc<dyn ms::AssociationSolver>,
}

impl ms::MhAssociationSolver for HypCondSolver {
    fn compute_marginals(&self, llr: ArrayView2<f64>, prior_hypotheses: &hyp::Hypotheses) -> ms::MhAssociationMarginalOutput {
        // llr is reward matrix for cluster, so n is number of tracks in cluster. We assume that the prior hypotheses are reindexed (TODO: )
        let (n, mp1) = llr.dim();
        let m = mp1 -1 ;

        let mut lbp_marginal_total = Array2::zeros((n, m + 1 + 1));
let mut conditioned_marginals = Array2::zeros((n, m + 1 + 1));

let mut theta_posterior = Vec::with_capacity(prior_hypotheses.num_hypotheses());

let mut likelihood = 0.0;

for (k, ph) in prior_hypotheses.hypotheses().iter().enumerate() {
    let hypo_track_indices: Vec<_> = ph.tracks().iter().map(|t| t - 1).collect();
    let hypo_prob = ph.probability();

    let llr_sub = llr.select(Axis(0), hypo_track_indices.as_slice());
    
    
    }

        todo!()
    }
}
