use crate::marginal_solver as ms;

use std::ffi::CStr;

use ndarray::prelude::*;

use dfg_da_sys as sys;

/// Error carrying the C API's thread-local `dfg_last_error()` message.
#[derive(Debug, Clone)]
struct DfgError(String);

/// EHM2 C API error message (thread-local `dfg_ehm2_last_error()`).
fn ehm2_last_error() -> String {
    // SAFETY: dfg_ehm2_last_error() returns a valid NUL-terminated C string.
    unsafe { CStr::from_ptr(sys::dfg_ehm2_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// Exact EHM2 marginal association probabilities + loglikelihood (pyehm fork).
///
/// `validation` (0/1 gating mask) and `likelihood` must share the same
/// `n x (m+1)` shape, with column 0 the missed-detection hypothesis. Returns the
/// association-probability matrix (same shape) and the cluster loglikelihood.
fn ehm2_run_and_likelihood(
    validation: ArrayView2<i32>,
    likelihood: ArrayView2<f64>,
) -> Result<(Array2<f64>, f64), DfgError> {
    let (rows, cols) = validation.dim();
    assert_eq!(
        likelihood.dim(),
        (rows, cols),
        "validation {:?} and likelihood {:?} shapes differ",
        validation.dim(),
        likelihood.dim()
    );
    // The C API wants row-major contiguous input.
    let validation = validation.as_standard_layout();
    let likelihood = likelihood.as_standard_layout();
    let val = validation
        .as_slice()
        .expect("standard layout is contiguous");
    let lik = likelihood
        .as_slice()
        .expect("standard layout is contiguous");

    let mut assoc = Array2::<f64>::zeros((rows, cols));
    let mut loglik = 0.0_f64;
    // SAFETY: pointers + dims describe valid rows*cols matrices; `assoc` is a fresh
    // contiguous rows*cols buffer the C API fills row-major.
    let rc = unsafe {
        sys::dfg_ehm2_run_and_likelihood(
            val.as_ptr(),
            lik.as_ptr(),
            rows,
            cols,
            assoc.as_mut_ptr(),
            &mut loglik,
        )
    };
    if rc != 0 {
        return Err(DfgError(ehm2_last_error()));
    }
    Ok((assoc, loglik))
}


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
        let (marginals, likelihood) = ehm2_run_and_likelihood(validation.view(), likelihood.view()).unwrap();

        ms::AssociationMarginalOutput{
            likelihood,
            marginals
        }
    }
}