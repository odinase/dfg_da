//! Python bindings to the Rust `dfg-da` crate (which wraps this repo's C++
//! single-cluster LBP through a C API). Built as a Python extension module with
//! maturin: `cd dfg-da-py && maturin develop`.

use dfg_da_rs::{lbp, lbp_single_cluster, Clustering, Hypotheses};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

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

/// Register declarative PyO3 submodules in `sys.modules`.
///
/// Nested `#[pymodule]` submodules are reachable by attribute access
/// (`parent.sub`) out of the box, but `import parent.sub` and
/// `from parent.sub import x` only work if the submodule is also present in
/// `sys.modules`. This macro generates that wiring for every listed submodule
/// under its full dotted path.
///
/// It expands to a block expression of type `PyResult<()>`, so drop it in as the
/// body of the parent module's `#[pymodule_init]` hook. (It can't emit the
/// `#[pymodule_init] fn` itself: `#[pymodule]` scans the module body for that
/// attribute *before* function-like macros expand, so a generated one is never
/// seen — you write the tiny shell, the macro fills the tedious part.) The first
/// argument is the init hook's `&Bound<PyModule>`; then list each submodule by
/// its dotted path relative to the parent (nested submodules included):
///
/// ```ignore
/// #[pymodule]
/// mod dfg_da_py {
///     #[pymodule] mod linalg { #[pymodule] mod decomp {} }
///     #[pymodule] mod clustering {}
///
///     #[pymodule_init]
///     fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
///         register_pysubmodules!(m, linalg, linalg.decomp, clustering)
///     }
/// }
/// ```
///
/// The `sys.modules` key is built as `<user-facing package>.<dotted ident path>`.
/// The package base is the module's `__package__` when maturin wraps the
/// extension (e.g. `dfg_da_py`, even though the extension itself is installed as
/// `dfg_da_py.dfg_da_py`), falling back to the module's own `__name__` when it is
/// top-level. This works whether a submodule was declared inline (PyO3 gives it a
/// dotted `__name__`) or imported from a file via `#[pymodule_export]` (which
/// keeps its bare ident name). Each submodule's `__name__` is also updated to its
/// importable path so introspection is consistent.
#[allow(unused_macros)]
macro_rules! register_pysubmodules {
    ( $m:expr, $( $head:ident $(.$rest:ident)* ),+ $(,)? ) => {{
        use ::pyo3::prelude::*;
        let __m: &::pyo3::Bound<'_, ::pyo3::types::PyModule> = $m;
        let __sys = __m.py().import("sys")?.getattr("modules")?;
        let __pkg: ::std::string::String = {
            let p: ::std::option::Option<::std::string::String> =
                __m.getattr("__package__")?.extract()?;
            match p {
                ::std::option::Option::Some(p) if !p.is_empty() => p,
                _ => __m.name()?.extract()?,
            }
        };
        $(
            {
                let __obj: ::pyo3::Bound<'_, ::pyo3::PyAny> = __m
                    .getattr(::core::stringify!($head))
                    $( .and_then(|__o| __o.getattr(::core::stringify!($rest))) )*?;
                let mut __name = __pkg.clone();
                __name.push('.');
                __name.push_str(::core::stringify!($head));
                $(
                    __name.push('.');
                    __name.push_str(::core::stringify!($rest));
                )*
                __obj.setattr("__name__", &__name)?;
                __sys.set_item(&__name, &__obj)?;
            }
        )+
        ::pyo3::PyResult::Ok(())
    }};
}

pub mod marginal_solvers;

#[pymodule]
mod dfg_da_py {
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::gen_stub_pyfunction;

    #[pymodule_export]
    use super::lbp_single_cluster_bethe;

    #[pymodule_export]
    use super::lbp_marginal;

    #[pymodule_export]
    use super::cluster_tracks;

    #[pymodule_export]
    use super::ehm2_run_and_likelihood;

    // Export the `#[pymodule]` item from the file module. Its Rust ident is
    // `r#impl`, but its `#[pyo3(name = "marginal_solvers")]` makes it appear in
    // Python as `dfg_da_py.marginal_solvers`.
    #[pymodule_export]
    use super::marginal_solvers::r#impl;

    #[gen_stub_pyfunction]
    #[pyfunction]
    fn testtt(x: u32) {
        println!("Received {x}");
    }

    // Register submodules in sys.modules so `import dfg_da_py.marginal_solvers`
    // (and `from dfg_da_py.marginal_solvers import ...`) work, not just attribute
    // access. Add more comma-separated paths here as submodules are added.
    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        register_pysubmodules!(m, marginal_solvers)
    }
}

// Gathers the `#[gen_stub_*]`-annotated items so the `stub_gen` binary can emit
// `dfg_da_py.pyi`. Generates a `pub fn stub_info() -> Result<StubInfo>`.
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
