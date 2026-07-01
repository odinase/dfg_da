//! Python bindings to the Rust `dfg-da` crate (which wraps this repo's C++
//! single-cluster LBP through a C API). Built as a Python extension module with
//! maturin: `cd dfg-da-py && maturin develop`.

use dfg_da_rs::{lbp_single_cluster, Hypotheses};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Run single-cluster LBP+Bethe and return the normalization constant.
///
/// Args:
///   reward: row-major flat reward matrix ("edmund" layout), length rows*cols.
///   rows, cols: its dimensions.
///   hyps: list of (tracks, log_prob) prior hypotheses; `tracks` are 1-indexed
///         local track ids as the C++ kernel expects.
///   max_iters: LBP iteration cap.
#[pyfunction]
#[pyo3(signature = (reward, rows, cols, hyps, max_iters = 300))]
fn lbp_single_cluster_bethe(
    reward: Vec<f64>,
    rows: usize,
    cols: usize,
    hyps: Vec<(Vec<usize>, f64)>,
    max_iters: usize,
) -> PyResult<f64> {
    let mut h = Hypotheses::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    for (tracks, log_prob) in &hyps {
        h.add(tracks, *log_prob);
    }
    let out = lbp_single_cluster(&reward, rows, cols, &h, max_iters)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.bethe_norm_const())
}

#[pymodule]
fn dfg_da_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lbp_single_cluster_bethe, m)?)?;
    Ok(())
}
