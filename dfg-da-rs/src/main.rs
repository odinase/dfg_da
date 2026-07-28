//! Scratch binary entry point for `dfg-da-rs`.
//!
//! Adding this file gives the package a binary target (named `dfg-da-rs`)
//! alongside the `dfg_da_rs` library, so you can `cargo run` to exercise code.
//! Everything below just calls into the public library API — edit `main` freely.

use core::f64;
use std::rc::Rc;

use dfg_da_rs::utils;
use ndarray::{self as na, array, prelude::*};

use anyhow::Result;
use dfg_da_rs::cluster_links as cl;
use dfg_da_rs::hypothesis::{Hypotheses, Hypothesis};
use dfg_da_rs::marginal_solver::hyp_cond_solver::HypCondSolver;
use dfg_da_rs::marginal_solver::lbp::Lbp;
use dfg_da_rs::marginal_solver::meas_cond_solver as mcs;
use dfg_da_rs::marginal_solver::{AssociationSolver, McMhAssociationSolver, MhAssociationSolver};
use ndarray::ShapeBuilder;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path;
use thiserror::Error;

#[derive(Debug, Error)]
#[error("Failed to lookup name {0}")]
struct FailedLookupError(&'static str);

/// Looks up `$field` in the workspace and copies it into an `Array2<$elem>`.
///
/// This bridges through `matfile`'s raw `NumericData` (`$variant`) rather than
/// its `TryInto` impls: `matfile` links `ndarray` 0.16 while this crate uses
/// 0.17, so its `TryInto` targets a different `Array2` type. `matfile` stores
/// column-major, hence `.f()` on the shape.
macro_rules! find_array_in_ws {
    ($ws:expr, $field:expr, $variant:ident, $elem:ty) => {{
        let field: &'static str = $field;
        (|| -> Result<Array2<$elem>, FailedLookupError> {
            let arr = $ws.find_by_name(field).ok_or(FailedLookupError(field))?;
            let size = arr.size();
            let &[rows, cols] = size.as_slice() else {
                return Err(FailedLookupError(field));
            };
            match arr.data() {
                matfile::NumericData::$variant { real, imag: _ } => {
                    Array2::from_shape_vec((rows, cols).f(), real.clone())
                        .map_err(|_| FailedLookupError(field))
                }
                _ => Err(FailedLookupError(field)),
            }
        })()
    }};
}

fn load_matfile_into_asso_parts(
    matfile_path: impl AsRef<path::Path>,
) -> Result<(Array2<f64>, Vec<Hypotheses>, Array2<u8>)> {
    let ws = matfile::MatFile::parse(BufReader::new(File::open(matfile_path)?))?;

    let llr_edmund: Array2<f64> = find_array_in_ws!(ws, "gainMatPostC", Double, f64)?;
    let _track_file: Array2<u8> = find_array_in_ws!(ws, "trackFile", UInt8, u8)?;
    let _measurements: Array2<u8> = find_array_in_ws!(ws, "measurements", UInt8, u8)?;

    let llr = utils::edmund_to_lc(&llr_edmund);
    let (_n, _mp1) = llr.dim();

    // TODO: derive prior_hypotheses_per_cluster + assoc_local from trackFile/measurements
    todo!("build Vec<Hypotheses> and assocLocal from the workspace")
}

fn run_hypo_solver() {
    // Log-likelihood-ratio matrix: one row per track, column 0 is misdetection
    // and columns 1.. are the measurements.
    let llr = array![[0.0, 1.5, 0.3], [0.0, 0.2, 1.1],];

    // Prior hypotheses over the two tracks (1-indexed track ids).
    let prior_hypotheses = Hypotheses::with_hypotheses(vec![
        Hypothesis::new(vec![1, 2], 0.7f64.ln()),
        Hypothesis::new(vec![1], 0.3f64.ln()),
    ]);

    // Single-cluster solver: LBP per hypothesis, conditioned over the prior.
    let solver: Rc<dyn AssociationSolver> = Rc::new(Lbp);
    let hyp_cond_solver = HypCondSolver::new(solver);

    let output = hyp_cond_solver.compute_marginals(llr.view(), &prior_hypotheses);

    println!("marginals =\n{}", output.marginals());
    println!("likelihood = {}", output.likelihood());
    println!("theta_posteriors = {}", output.theta_posteriors());
}

// return R, prior_hypotheses_per_cluster, assocLocal
fn test_case3() -> (Array2<f64>, Vec<Hypotheses>, Array2<u8>) {
    //     R = np.array([
    //     [    3.0, -inf, -inf, -inf,   -0.60, -inf, -inf, -inf, -inf, -inf, -inf],
    //     [    3.2, -inf, -inf, -inf, -inf,   -0.56, -inf, -inf, -inf, -inf, -inf],
    //     [   -3.0,     2.0,     1.2, -inf, -inf, -inf,   -0.46, -inf, -inf, -inf, -inf],
    //     [-inf, -inf,     3.0, -inf, -inf, -inf, -inf,   -0.62, -inf, -inf, -inf],
    //     [-inf,    -0.4,    -1.8, -inf, -inf, -inf, -inf, -inf,   -0.55, -inf, -inf],
    //     [-inf,     0.5, -inf,    -0.1, -inf, -inf, -inf, -inf, -inf,   -0.62, -inf],
    //     [-inf, -inf, -inf,     0.8, -inf, -inf, -inf, -inf, -inf, -inf,   -0.55],
    // ], order='F')

    let inf = f64::INFINITY;

    let llr = array![
        [
            3.0, -inf, -inf, -inf, -0.60, -inf, -inf, -inf, -inf, -inf, -inf
        ],
        [
            3.2, -inf, -inf, -inf, -inf, -0.56, -inf, -inf, -inf, -inf, -inf
        ],
        [
            -3.0, 2.0, 1.2, -inf, -inf, -inf, -0.46, -inf, -inf, -inf, -inf
        ],
        [
            -inf, -inf, 3.0, -inf, -inf, -inf, -inf, -0.62, -inf, -inf, -inf
        ],
        [
            -inf, -0.4, -1.8, -inf, -inf, -inf, -inf, -inf, -0.55, -inf, -inf
        ],
        [
            -inf, 0.5, -inf, -0.1, -inf, -inf, -inf, -inf, -inf, -0.62, -inf
        ],
        [
            -inf, -inf, -inf, 0.8, -inf, -inf, -inf, -inf, -inf, -inf, -0.55
        ],
    ];

    let llr = utils::edmund_to_lc(&llr);

    let prior_hypotheses_per_cluster = vec![
        Hypotheses::with_hypotheses(vec![
            Hypothesis::new(vec![1, 2], 0.5f64.ln()),
            Hypothesis::new(vec![1, 3], 0.5f64.ln()),
        ]),
        Hypotheses::with_hypotheses(vec![
            Hypothesis::new(vec![4], 0.5f64.ln()),
            Hypothesis::new(vec![5], 0.5f64.ln()),
        ]),
        Hypotheses::with_hypotheses(vec![
            Hypothesis::new(vec![6, 7], 0.2f64.ln()),
            Hypothesis::new(vec![6], 0.8f64.ln()),
        ]),
    ];

    let assoc_local = array![
        [1, 1, 1], //
        [1, 0, 0],
    ];

    // prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
    //     pdd.hypothesis.Hypotheses([
    //         pdd.hypothesis.Hypothesis([1, 2], np.log(0.5)),
    //         pdd.hypothesis.Hypothesis([1, 3], np.log(0.5))
    //     ]),
    //     pdd.hypothesis.Hypotheses([
    //         pdd.hypothesis.Hypothesis([4], np.log(0.5)),
    //         pdd.hypothesis.Hypothesis([5], np.log(0.5))
    //     ]),
    //     pdd.hypothesis.Hypotheses([
    //         pdd.hypothesis.Hypothesis([6, 7], np.log(0.2)),
    //         pdd.hypothesis.Hypothesis([6],    np.log(0.8))
    //     ])
    // ])

    // assocLocal = np.array([
    //     [1, 1, 1],
    //     [1, 0, 0]
    // ])

    // return R, prior_hypotheses_per_cluster, assocLocal
    (llr, prior_hypotheses_per_cluster, assoc_local)
}

fn run_multicluster_hypo_solver() {
    let (llr, prior_hypotheses_per_cluster, assoc_local) = test_case3();
    let cluster_links = cl::ClusterLinks::from_parsed_mat_file(
        llr.view(),
        prior_hypotheses_per_cluster.as_slice(),
        assoc_local.view(),
    );

    let asso_info = mcs::AssociationInfo::new(llr, prior_hypotheses_per_cluster);

    let solver: Rc<dyn AssociationSolver> = Rc::new(Lbp);
    let mh_solver: Rc<dyn MhAssociationSolver> = Rc::new(HypCondSolver::new(solver));

    let mcmh_solver = mcs::MeasCondSolver::new(asso_info, mh_solver, cluster_links);

    let mcmh_marginals = mcmh_solver.compute_marginals();
    println!(
        "marginals=\n{}\nlikelihood: {}\ntheta posterior: {:?}",
        mcmh_marginals.marginals(),
        mcmh_marginals.likelihood(),
        mcmh_marginals.theta_posteriors()
    );
}

fn main() {
    run_multicluster_hypo_solver();
}
