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

pub use clustering::{ClusterError, Clustering, Jagged};
pub mod cluster;
pub mod cluster_links;
pub mod conditional_supercluster_marginals;
pub mod hypothesis;
pub mod marginal_solver;
pub mod marginal_solvers;
pub mod utils;

use ndarray::prelude::*;

#[inline(never)]
pub fn lbp_marginal_f64(llr: ArrayView2<f64>) -> (Array2<f64>, f64) {
    lbp::lbp_marginal(&llr)
}

#[inline(never)]
pub fn lbp_marginal_f64_zip(llr: ArrayView2<f64>) -> (Array2<f64>, f64) {
    lbp::lbp_marginal_zip(&llr)
}
