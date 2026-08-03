//! PyO3 wrapper submodule exposing `dfg_da_rs::marginal_solvers` as
//! `dfg_da_py.marginal_solvers`.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

/// `dfg_da_py.marginal_solvers` — Python view over the Rust `marginal_solvers`.
///
/// The Rust ident is `r#impl`, but the Python-visible module name is forced to
/// `marginal_solvers` via `#[pyo3(name = ...)]` (that name is baked into the
/// module's `_PYO3_DEF`, which is what `add_to_module` uses — a `use ... as`
/// rename on the export side would *not* change it). Must be `pub` so the parent
/// module can reference it as `super::marginal_solvers::r#impl`.
#[pymodule]
#[pyo3(name = "marginal_solvers")]
pub mod r#impl {
    use super::*;
    use dfg_da_rs::marginal_solver as ms;
    use std::sync::Arc;

    use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
    use dfg_da_rs::cluster_links::ClusterLinks;
    use dfg_da_rs::hypothesis::{Hypotheses, Hypothesis};
    use ms::McMhAssociationSolver; // brings compute_marginals() into scope

    /// Placeholder wrapper — replace with the real solver entry points once
    /// `dfg_da_rs::marginal_solvers` grows them.
    #[pyfunction]
    #[gen_stub_pyfunction]
    fn solve_placeholder(x: f64) -> f64 {
        dfg_da_rs::marginal_solvers::solve_placeholder(x)
    }

    #[pyclass(name = "MeasCondSolver")]
    struct PyMeasCondSolver {
        inner: ms::meas_cond_solver::MeasCondSolver,
    }

    #[pyclass(name  = "Lbp")]
    struct PyLbp {
        inner: Arc<ms::lbp::Lbp>,
    }

    #[pymethods]
    impl PyLbp {
        #[new]
        fn new() -> Self {
            Self {
                inner: Arc::new(ms::lbp::Lbp),
            }
        }
    }

    #[pyclass(name = "HypCondSolver")]
    struct PyHypCondSolver {
        inner: Arc<dyn ms::MhAssociationSolver>,
    }

    #[pymethods]
    impl PyHypCondSolver {
        #[new]
        fn new(lbp: &PyLbp) -> Self {
            // Arc<Lbp> coerces to Arc<dyn AssociationSolver> at the call boundary.
            let inner: Arc<dyn ms::MhAssociationSolver> =
                Arc::new(ms::hyp_cond_solver::HypCondSolver::new(lbp.inner.clone()));
            Self { inner }
        }
    }

    #[pymethods]
    impl PyMeasCondSolver {
        /// TEMPORARY array-based binding mirroring the ravens-script init style
        /// (`MulticlusterEfficientMarginalsLBP(R_LC=..., ...)`); replace with a stable
        /// hypothesis-object constructor later. Call as `MeasCondSolver.temp_new(...)`.
        ///
        /// Args:
        ///   R_LC: (n, m+1) float64 local-cluster LLR, column 0 = misdetection.
        ///   prior_hypotheses_per_cluster: per-cluster list of (tracks, log_weight);
        ///     tracks are 1-indexed global ids.
        ///   assocLocal: (2, num_clusters) uint8 association array (row 0 = 1-indexed master ids).
        ///   lbp_solver: an MhAssociationSolver (e.g. HypCondSolver).
        #[staticmethod]
        #[pyo3(signature = (R_LC, prior_hypotheses_per_cluster, assocLocal, lbp_solver))]
        #[allow(non_snake_case)]
        fn temp_new(
            R_LC: PyReadonlyArray2<'_, f64>,
            prior_hypotheses_per_cluster: Vec<Vec<(Vec<usize>, f64)>>,
            assocLocal: PyReadonlyArray2<'_, u8>,
            lbp_solver: &PyHypCondSolver,
        ) -> Self {
            let llr_view = R_LC.as_array();
            let assoc_view = assocLocal.as_array();

            let prior: Vec<Hypotheses> = prior_hypotheses_per_cluster
                .into_iter()
                .map(|cluster| {
                    Hypotheses::with_hypotheses(
                        cluster
                            .into_iter()
                            .map(|(t, w)| Hypothesis::new(t, w))
                            .collect(),
                    )
                })
                .collect();

            // Borrows released once ClusterLinks (owned) is built; then move prior into AssociationInfo.
            let cluster_links = ClusterLinks::from_parsed_mat_file(llr_view, &prior, assoc_view);
            let asso_info =
                ms::meas_cond_solver::AssociationInfo::new(llr_view.to_owned(), prior);

            let inner = ms::meas_cond_solver::MeasCondSolver::new(
                asso_info,
                lbp_solver.inner.clone(),
                cluster_links,
            );
            Self { inner }
        }

        /// Run the solver → (marginals (n, m+2) float64, likelihood, per-cluster theta posteriors).
        fn compute_marginals_likelihood<'py>(
            &self,
            py: Python<'py>,
        ) -> (Bound<'py, PyArray2<f64>>, f64, Vec<Vec<f64>>) {
            let out = self.inner.compute_marginals();
            let marginals = out.marginals().to_owned().into_pyarray(py);
            let thetas = out.theta_posteriors().iter().map(|a| a.to_vec()).collect();
            (marginals, out.likelihood(), thetas)
        }
    }
}

// Gathers the `#[gen_stub_*]`-annotated items so the `stub_gen` binary can emit
// `dfg_da_py.pyi`. Generates a `pub fn stub_info() -> Result<StubInfo>`.
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
