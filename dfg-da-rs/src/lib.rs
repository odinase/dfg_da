//! Safe, hand-written Rust wrappers over the `dfg-da-sys` raw FFI.
//!
//! Mirrors the `kalib` <- `kalib-sys` relationship in ~/kalib-rs: the raw
//! bindings are bindgen-generated, everything here is written by hand — RAII
//! `Drop` for the opaque handles and slice-based, `Result`-returning calls.

use std::ffi::CStr;
use std::fmt;

use dfg_da_sys as sys;

pub mod clustering;
pub mod lbp;

pub use clustering::{Clustering, ClusterError, Jagged};
pub mod hypothesis;

use ndarray::{Array2, ArrayView2, ArrayBase, Data, prelude::*};


// WIP: signature completed minimally so the crate compiles — finish as needed.
pub fn find_linking_measurements<D: Data<Elem = f64>>(_llr: &ArrayBase<D, Ix2>, _cluster_idxs: &[usize]) {
    todo!("find_linking_measurements")
}





#[inline(never)]
pub fn lbp_marginal_f64(llr: ArrayView2<f64>) -> (Array2<f64>, f64) {
    lbp::lbp_marginal(&llr)
}

#[inline(never)]
pub fn lbp_marginal_f64_zip(llr: ArrayView2<f64>) -> (Array2<f64>, f64) {
    lbp::lbp_marginal_zip(&llr)
}

/// Error carrying the C API's thread-local `dfg_last_error()` message.
#[derive(Debug, Clone)]
pub struct DfgError(pub String);

impl fmt::Display for DfgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dfg_da error: {}", self.0)
    }
}

impl std::error::Error for DfgError {}

fn last_error() -> String {
    // SAFETY: dfg_last_error() returns a valid NUL-terminated C string.
    unsafe { CStr::from_ptr(sys::dfg_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// A builder of prior hypotheses for one cluster (owns the C++ object).
pub struct Hypotheses {
    handle: sys::DfgHypotheses,
}

impl Hypotheses {
    pub fn new() -> Result<Self, DfgError> {
        // SAFETY: create returns an owned handle or NULL on failure.
        let handle = unsafe { sys::dfg_hypotheses_create() };
        if handle.is_null() {
            return Err(DfgError(last_error()));
        }
        Ok(Self { handle })
    }

    /// Append one hypothesis: the global track ids it asserts exist + its log prob.
    pub fn add(&mut self, tracks: &[usize], log_prob: f64) {
        // SAFETY: `tracks`/`len` describe a valid slice; handle is non-null.
        unsafe {
            sys::dfg_hypotheses_add(self.handle, tracks.as_ptr(), tracks.len(), log_prob);
        }
    }

    pub fn len(&self) -> usize {
        // SAFETY: handle is a valid, owned pointer.
        unsafe { sys::dfg_hypotheses_len(self.handle) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Drop for Hypotheses {
    fn drop(&mut self) {
        // SAFETY: handle was created by dfg_hypotheses_create and not freed yet.
        unsafe { sys::dfg_hypotheses_destroy(self.handle) };
    }
}

/// Result of a single-cluster LBP run (owns the C++ output object).
pub struct LbpOutput {
    handle: sys::DfgLbpOutput,
}

impl LbpOutput {
    /// The Bethe pseudodual normalization constant (the cluster likelihood).
    pub fn bethe_norm_const(&self) -> f64 {
        // SAFETY: handle is a valid, owned pointer.
        unsafe { sys::dfg_lbp_output_bethe_norm_const(self.handle) }
    }
}

impl Drop for LbpOutput {
    fn drop(&mut self) {
        // SAFETY: handle was created by dfg_lbp_single_cluster and not freed yet.
        unsafe { sys::dfg_lbp_output_destroy(self.handle) };
    }
}

/// Run single-cluster LBP + Bethe on a `rows x cols` row-major reward matrix
/// (the "edmund" layout) under the given prior hypotheses.
pub fn lbp_single_cluster(
    reward_row_major: &[f64],
    rows: usize,
    cols: usize,
    hyps: &Hypotheses,
    max_iters: usize,
) -> Result<LbpOutput, DfgError> {
    assert_eq!(
        reward_row_major.len(),
        rows * cols,
        "reward matrix length {} != rows*cols {}",
        reward_row_major.len(),
        rows * cols
    );
    // SAFETY: pointer + dims describe a valid matrix; handle is non-null.
    let handle = unsafe {
        sys::dfg_lbp_single_cluster(
            reward_row_major.as_ptr(),
            rows,
            cols,
            hyps.handle,
            max_iters,
        )
    };
    if handle.is_null() {
        return Err(DfgError(last_error()));
    }
    Ok(LbpOutput { handle })
}
