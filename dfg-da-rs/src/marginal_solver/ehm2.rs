use crate::marginal_solver as ms;
use std::mem::MaybeUninit;

use ndarray::Zip;
use ndarray::{Array2, ArrayBase, Data, Ix2, prelude::*, s};

// Add parameters here later?
pub struct Ehm2;

fn llr_to_validation_likelihood_matrix(llr: ArrayView2<f64>) -> (Array2<i32>, Array2<f64>) {
    let likelihood_matrix = llr.mapv(f64::exp);
    let validation_matrix = likelihood_matrix.mapv(|e| i32::from(e > 0.0));

    (validation_matrix, likelihood_matrix)
}

impl ms::AssociationSolver for Ehm2 {
    fn compute_marginals(&self, llr: ArrayView2<f64>) -> ms::AssociationMarginalOutput {
        let (validation, likelihood) = llr_to_validation_likelihood_matrix(llr);
        let (marginals, likelihood) = crate::ehm2_run_and_likelihood(validation.view(), likelihood.view()).unwrap();

        ms::AssociationMarginalOutput{
            likelihood,
            marginals
        }
    }
}