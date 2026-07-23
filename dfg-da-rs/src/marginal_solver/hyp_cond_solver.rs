use crate::{hypothesis as hyp, marginal_solver as ms, utils};
use ndarray::{self as nd, prelude::*};
use std::rc;

// Single cluster solver that takes in a single hypothesis solver and conditiones over prior hypotheses and computes the marginals with total probability
pub struct HypCondSolver {
    solver: rc::Rc<dyn ms::AssociationSolver>,
}

impl ms::MhAssociationSolver for HypCondSolver {
    fn compute_marginals(
        &self,
        llr: ArrayView2<f64>,
        prior_hypotheses: &hyp::Hypotheses,
    ) -> ms::MhAssociationMarginalOutput {
        let (n, mp1) = llr.dim();
        let m = mp1 - 1;

        let mut lbp_marginal_total = Array2::<f64>::zeros((n, m + 2));
        let mut conditioned_marginals = Array2::<f64>::zeros((n, m + 2));
        let mut theta_posterior = Array1::<f64>::zeros(prior_hypotheses.num_hypotheses());
        let mut likelihood = 0.0;

        for (k, ph) in prior_hypotheses.hypotheses().iter().enumerate() {
            // 1-indexed track IDs -> 0-indexed. (Python sorts these; order is
            // irrelevant as long as select and scatter stay consistent.)
            let track_ids: Vec<usize> = ph.tracks().iter().map(|t| t - 1).collect();
            let hypo_prob = ph.probability();

            let llr_sub = llr.select(Axis(0), track_ids.as_slice());

            // Non-empty -> run LBP. Empty -> normalizing constant exp(0) = 1, no marginals.
            let (hypo_marginals, hypo_likelihood) = if !track_ids.is_empty() {
                let out = self.solver.compute_marginals(llr_sub.view());
                (out.marginals, out.likelihood)
            } else {
                (Array2::<f64>::zeros((0, m + 1)), 1.0)
            };

            // Existing tracks: scatter the (m+1) LBP marginals into their cluster
            // rows; existence column (last) = 0.
            for (sub_row, &track_id) in track_ids.iter().enumerate() {
                let mut row = conditioned_marginals.row_mut(track_id);
                row.slice_mut(s![..-1]).assign(&hypo_marginals.row(sub_row));
                *row.last_mut().unwrap() = 0.0;
            }

            // Non-existing tracks: P(a_t = non-existent) = 1, everything else 0.
            for track_id in (0..n).filter(|t| !track_ids.contains(t)) {
                let mut row = conditioned_marginals.row_mut(track_id);
                row.slice_mut(s![..-1]).fill(0.0);
                *row.last_mut().unwrap() = 1.0;
            }

            let hypo_weight = hypo_likelihood * hypo_prob;
            likelihood += hypo_weight;
            theta_posterior[k] = hypo_weight;
            lbp_marginal_total.scaled_add(hypo_weight, &conditioned_marginals);
        }

        assert!(
            likelihood > 0.0 && likelihood.is_finite(),
            "total hypothesis likelihood is {likelihood}; all hypothesis weights \
     may have underflowed (consider accumulating in log-space)"
        );

        utils::normalize_rows_inplace(&mut theta_posterior);
        utils::normalize_rows_inplace(&mut lbp_marginal_total);

        assert!(
            lbp_marginal_total.iter().all(|&x| (0.0..=1.0).contains(&x)),
            "values are not all within [0, 1]"
        );

        ms::MhAssociationMarginalOutput {
            marginals: ms::AssociationMarginalOutput {
                marginals: lbp_marginal_total,
                likelihood,
            },
            theta_posteriors: theta_posterior,
        }
    }
}
