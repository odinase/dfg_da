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

/// Reads `$field` from the workspace as `f64`, regardless of the numeric class
/// MATLAB stored it in, and returns `(size, column_major_data)`.
///
/// MATLAB picks the smallest integer type that fits, so the same field can be
/// `uint8` in one file and `uint16`/`double` in another (e.g. `hypos`,
/// `clusters`). Matching every `NumericData` variant and widening to `f64`
/// keeps the caller agnostic to that. The returned data is in `matfile`'s
/// native column-major order.
fn find_numeric_as_f64(
    ws: &matfile::MatFile,
    field: &'static str,
) -> Result<(Vec<usize>, Vec<f64>), FailedLookupError> {
    let arr = ws.find_by_name(field).ok_or(FailedLookupError(field))?;
    let size = arr.size().clone();
    // Widen whichever integer/float class MATLAB used to `f64`.
    macro_rules! widen {
        ($real:expr) => {
            $real.iter().map(|&x| x as f64).collect()
        };
    }
    let data: Vec<f64> = match arr.data() {
        matfile::NumericData::Int8 { real, .. } => widen!(real),
        matfile::NumericData::UInt8 { real, .. } => widen!(real),
        matfile::NumericData::Int16 { real, .. } => widen!(real),
        matfile::NumericData::UInt16 { real, .. } => widen!(real),
        matfile::NumericData::Int32 { real, .. } => widen!(real),
        matfile::NumericData::UInt32 { real, .. } => widen!(real),
        matfile::NumericData::Int64 { real, .. } => widen!(real),
        matfile::NumericData::UInt64 { real, .. } => widen!(real),
        matfile::NumericData::Single { real, .. } => widen!(real),
        matfile::NumericData::Double { real, .. } => real.clone(),
    };
    Ok((size, data))
}

/// Rebuilds `prior_hypotheses_per_cluster` from the flat PMBM workspace arrays.
///
/// Port of `MatFileParser.ws_to_prior_hypotheses_cpp` in
/// `ravens_parser_parallell_multicluster2.py`'s `stats_logger`:
/// - `hypos`        — track ids of every hypothesis, concatenated (1-indexed).
/// - `hyposCard`    — number of tracks in each hypothesis (indexes into `hypos`).
/// - `probLogHypos` — log-weight of each hypothesis.
/// - `clustersCard` — number of hypotheses in each cluster.
/// - `clusters`     — hypothesis indices per cluster, concatenated (1-indexed
///   in MATLAB; we subtract 1 to index `hypos`/`probLogHypos`/`hyposCard`).
///
/// Track ids stay 1-indexed to match `Hypothesis::new` usage elsewhere.
fn ws_to_prior_hypotheses(ws: &matfile::MatFile) -> Result<Vec<Hypotheses>> {
    let (_, hypos_f) = find_numeric_as_f64(ws, "hypos")?;
    let (_, hypos_card_f) = find_numeric_as_f64(ws, "hyposCard")?;
    let (_, prob_log_hypos) = find_numeric_as_f64(ws, "probLogHypos")?;
    let (_, clusters_card_f) = find_numeric_as_f64(ws, "clustersCard")?;
    let (_, clusters_f) = find_numeric_as_f64(ws, "clusters")?;

    let hypos: Vec<usize> = hypos_f.iter().map(|&x| x as usize).collect();
    let hypos_card: Vec<usize> = hypos_card_f.iter().map(|&x| x as usize).collect();
    // MATLAB 1-indexes hypothesis ids; shift to 0-indexed to slice our arrays.
    let clusters: Vec<usize> = clusters_f.iter().map(|&x| x as usize - 1).collect();
    let clusters_card: Vec<usize> = clusters_card_f.iter().map(|&x| x as usize).collect();

    // `end_ind[h]` is the exclusive end of hypothesis `h`'s tracks inside
    // `hypos`; its start is `end_ind[h] - hypos_card[h]`.
    let end_ind: Vec<usize> = hypos_card
        .iter()
        .scan(0usize, |acc, &card| {
            *acc += card;
            Some(*acc)
        })
        .collect();

    // Walk `clusters` in `clustersCard`-sized chunks; each chunk is one cluster.
    let mut prior_hypotheses_per_cluster = Vec::with_capacity(clusters_card.len());
    let mut offset = 0usize;
    for &cluster_card in &clusters_card {
        let hyp_idxs = &clusters[offset..offset + cluster_card];
        offset += cluster_card;

        let hypotheses = hyp_idxs
            .iter()
            .map(|&h| {
                let tracks = hypos[end_ind[h] - hypos_card[h]..end_ind[h]].to_vec();
                Hypothesis::new(tracks, prob_log_hypos[h])
            })
            .collect();
        prior_hypotheses_per_cluster.push(Hypotheses::with_hypotheses(hypotheses));
    }

    Ok(prior_hypotheses_per_cluster)
}

/// Parses a PMBM `.mat` file into the three inputs `run_multicluster_hypo_solver`
/// consumes: the log-likelihood-ratio matrix in LC form, the prior hypotheses
/// per cluster, and `assocLocal`.
///
/// Mirrors `MatFileParser.__init__` + `ws_to_prior_hypotheses_cpp`.
fn load_matfile_into_asso_parts(
    matfile_path: impl AsRef<path::Path>,
) -> Result<(Array2<f64>, Vec<Hypotheses>, Array2<u8>)> {
    let ws = matfile::MatFile::parse(BufReader::new(File::open(matfile_path)?))?;

    // `n` = #tracks (trackFile columns), `m` = #measurements (measurements
    // columns). `gainMatPostC` is a padded matrix; Python slices it down to
    // `gainMatPostC[:n, :n+m]` (the "Edmund" reward matrix) before converting
    // to LC form (misdetection column ++ per-measurement columns).
    let (track_file_size, _) = find_numeric_as_f64(&ws, "trackFile")?;
    let (measurements_size, _) = find_numeric_as_f64(&ws, "measurements")?;
    let n = track_file_size[1];
    let m = measurements_size[1];

    let gain_mat: Array2<f64> = find_array_in_ws!(ws, "gainMatPostC", Double, f64)?;
    let reward_edmund = gain_mat.slice(s![..n, ..(n + m)]).to_owned();
    let llr = utils::edmund_to_lc(&reward_edmund);

    let prior_hypotheses_per_cluster = ws_to_prior_hypotheses(&ws)?;

    // assocLocal: row 0 = master cluster index (1-indexed), row 1 = is-master
    // flag; one column per cluster. `matfile` hands us column-major data, but
    // `ClusterLinks::from_parsed_mat_file` reads `assoc_local.row(0).as_slice()`,
    // which requires a C-contiguous array — so materialize standard layout.
    let (assoc_size, assoc_data) = find_numeric_as_f64(&ws, "assocLocal")?;
    let &[a_rows, a_cols] = assoc_size.as_slice() else {
        return Err(FailedLookupError("assocLocal").into());
    };
    let assoc_u8: Vec<u8> = assoc_data.iter().map(|&x| x as u8).collect();
    let assoc_local = Array2::from_shape_vec((a_rows, a_cols).f(), assoc_u8)
        .map_err(|_| FailedLookupError("assocLocal"))?
        .as_standard_layout()
        .into_owned();

    Ok((llr, prior_hypotheses_per_cluster, assoc_local))
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

fn run_multicluster_hypo_solver(
    llr: Array2<f64>,
    prior_hypotheses_per_cluster: Vec<Hypotheses>,
    assoc_local: Array2<u8>,
) {
    println!(
        "parsed: llr {:?}, {} clusters, assocLocal {:?}",
        llr.dim(),
        prior_hypotheses_per_cluster.len(),
        assoc_local.dim()
    );

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

    // Full marginals are huge for real files; only dump them for small problems.
    if mcmh_marginals.marginals().len() <= 200 {
        println!("marginals=\n{}", mcmh_marginals.marginals());
    } else {
        println!("marginals dim={:?}", mcmh_marginals.marginals().dim());
    }
    println!("likelihood: {}", mcmh_marginals.likelihood());
    println!("theta posterior: {:?}", mcmh_marginals.theta_posteriors());
}

fn main() -> Result<()> {
    // With a `.mat` path argument, parse a real PMBM file; otherwise fall back
    // to the built-in `test_case3()` fixture.
    let (llr, prior_hypotheses_per_cluster, assoc_local) = match std::env::args().nth(1) {
        Some(path) => {
            println!("Loading association problem from {path}");
            load_matfile_into_asso_parts(&path)?
        }
        None => {
            println!("No .mat path given; using built-in test_case3()");
            test_case3()
        }
    };

    run_multicluster_hypo_solver(llr, prior_hypotheses_per_cluster, assoc_local);
    Ok(())
}
