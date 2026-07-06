//! Python bindings to the Rust `dfg-da` crate (which wraps this repo's C++
//! single-cluster LBP through a C API). Built as a Python extension module with
//! maturin: `cd dfg-da-py && maturin develop`.

use dfg_da_rs::{lbp_single_cluster, Clustering, Hypotheses, lbp};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

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

/// Decode cluster membership from the four PMBM cloud arrays.
///
/// Pass the raw arrays from ``priorLikelihood*.mat`` (1-based ids) as
/// contiguous ``np.uint64`` arrays; they are borrowed zero-copy and cast to
/// ``usize`` internally. Returns ``cluster -> sorted track numbers`` (cluster
/// index 0-based).
#[gen_stub_pyfunction]
#[pyfunction]
fn cluster_tracks(
    clusters: PyReadonlyArray1<'_, u64>,
    clusters_card: PyReadonlyArray1<'_, u64>,
    hypos: PyReadonlyArray1<'_, u64>,
    hypos_card: PyReadonlyArray1<'_, u64>,
) -> PyResult<Vec<Vec<usize>>> {
    let to_usize = |a: &PyReadonlyArray1<'_, u64>| -> PyResult<Vec<usize>> {
        Ok(a.as_slice()?.iter().map(|&x| x as usize).collect())
    };
    let clusters = to_usize(&clusters)?;
    let clusters_card = to_usize(&clusters_card)?;
    let hypos = to_usize(&hypos)?;
    let hypos_card = to_usize(&hypos_card)?;
    let clustering = Clustering::decode(&clusters, &clusters_card, &hypos, &hypos_card)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(clustering.to_vecs())
}

/// Exact EHM2 marginal association probabilities + loglikelihood (pyehm fork).
///
/// Args:
///   validation_matrix: (n, m+1) int32 gating mask (column 0 = missed detection).
///   likelihood_matrix: (n, m+1) float64 association likelihoods, same shape.
///
/// Returns:
///   (association_matrix (n, m+1) float64, loglikelihood).
#[gen_stub_pyfunction]
#[pyfunction]
fn ehm2_run_and_likelihood<'py>(
    py: Python<'py>,
    validation_matrix: PyReadonlyArray2<'py, i32>,
    likelihood_matrix: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, f64)> {
    let validation = validation_matrix.as_array();
    let likelihood = likelihood_matrix.as_array();
    let (assoc, loglik) = py
        .detach(|| dfg_da_rs::ehm2_run_and_likelihood(validation, likelihood))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((assoc.into_pyarray(py), loglik))
}

#[pymodule]
fn dfg_da_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lbp_single_cluster_bethe, m)?)?;

    m.add_function(wrap_pyfunction!(lbp_marginal, m)?)?;

    m.add_function(wrap_pyfunction!(cluster_tracks, m)?)?;

    m.add_function(wrap_pyfunction!(ehm2_run_and_likelihood, m)?)?;

    Ok(())
}

// Gathers the `#[gen_stub_*]`-annotated items so the `stub_gen` binary can emit
// `dfg_da_py.pyi`. Generates a `pub fn stub_info() -> Result<StubInfo>`.
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
