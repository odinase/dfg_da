//! Python bindings to the Rust `dfg-da` crate (which wraps this repo's C++
//! single-cluster LBP through a C API). Built as a Python extension module with
//! maturin: `cd dfg-da-py && maturin develop`.

use dfg_da_rs::{lbp_single_cluster, Hypotheses, lbp};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};

/// Run single-cluster LBP+Bethe and return the normalization constant.
///
/// Args:
///   reward: row-major flat reward matrix ("edmund" layout), length rows*cols.
///   rows, cols: its dimensions.
///   hyps: list of (tracks, log_prob) prior hypotheses; `tracks` are 1-indexed
///         local track ids as the C++ kernel expects.
///   max_iters: LBP iteration cap.
#[gen_stub_pyfunction]
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

#[gen_stub_pyfunction]
#[pyfunction]
fn lbp_marginal<'py>(
    py: Python<'py>,
    llr: PyReadonlyArray2<'py, f64>,
) -> (Bound<'py, PyArray2<f64>>, f64) {
    let view = llr.as_array(); // zero-copy ArrayView2<f64>
    let (marginals, value) = py.detach(|| lbp::lbp_marginal(&view));
    (marginals.into_pyarray(py), value)
}

#[pymodule]
fn dfg_da_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lbp_single_cluster_bethe, m)?)?;

    m.add_function(wrap_pyfunction!(lbp_marginal, m)?)?;

    Ok(())
}

// Gathers the `#[gen_stub_*]`-annotated items so the `stub_gen` binary can emit
// `dfg_da_py.pyi`. Generates a `pub fn stub_info() -> Result<StubInfo>`.
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
