//! Compute the multicluster inclusion-exclusion marginals for one fixture and
//! print the result as JSON on stdout.
//!
//! Usage: `cargo run --example ie_fixture -- <path-to-fixture.json>`
//!
//! Reads the fixture's inputs (`R_LC` with -inf encoded as JSON null,
//! `prior_hypotheses_per_cluster`, `assoc_local`), runs the Rust `MeasCondSolver`,
//! and emits `{"marginals": [[..]], "likelihood": f, "theta_posteriors": [[..]]}`.
//! The Python parity test drives this and compares against the Python reference.

use std::rc::Rc;

use ndarray::Array2;
use serde_json::{json, Value};

use dfg_da_rs::cluster_links::ClusterLinks;
use dfg_da_rs::hypothesis::{Hypotheses, Hypothesis};
use dfg_da_rs::marginal_solver::hyp_cond_solver::HypCondSolver;
use dfg_da_rs::marginal_solver::lbp::Lbp;
use dfg_da_rs::marginal_solver::meas_cond_solver::{AssociationInfo, MeasCondSolver};
use dfg_da_rs::marginal_solver::{AssociationSolver, McMhAssociationSolver, MhAssociationSolver};

/// A JSON number, or `null` meaning -inf (a measurement a track does not gate).
fn cell(v: &Value) -> f64 {
    if v.is_null() {
        f64::NEG_INFINITY
    } else {
        v.as_f64().expect("R_LC cell must be a number or null")
    }
}

fn to_matrix(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = v
        .as_array()
        .expect("expected a 2-D array")
        .iter()
        .map(|row| row.as_array().unwrap().iter().map(cell).collect())
        .collect();
    let n = rows.len();
    let m = if n == 0 { 0 } else { rows[0].len() };
    Array2::from_shape_vec((n, m), rows.into_iter().flatten().collect()).unwrap()
}

fn parse_priors(v: &Value) -> Vec<Hypotheses> {
    v.as_array()
        .expect("prior_hypotheses_per_cluster must be an array")
        .iter()
        .map(|cluster| {
            let hypotheses = cluster
                .as_array()
                .unwrap()
                .iter()
                .map(|h| {
                    let tracks = h["tracks"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|t| t.as_u64().unwrap() as usize)
                        .collect();
                    Hypothesis::new(tracks, h["log_weight"].as_f64().unwrap())
                })
                .collect();
            Hypotheses::with_hypotheses(hypotheses)
        })
        .collect()
}

fn parse_assoc_local(v: &Value) -> Array2<u8> {
    let rows: Vec<Vec<u8>> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u8)
                .collect()
        })
        .collect();
    let n = rows.len();
    let m = rows[0].len();
    Array2::from_shape_vec((n, m), rows.into_iter().flatten().collect()).unwrap()
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: ie_fixture <path-to-fixture.json>");
    let text = std::fs::read_to_string(&path).unwrap();
    let fixture: Value = serde_json::from_str(&text).unwrap();

    let llr = to_matrix(&fixture["R_LC"]);
    let priors = parse_priors(&fixture["prior_hypotheses_per_cluster"]);
    let assoc_local = parse_assoc_local(&fixture["assoc_local"]);

    let cluster_links =
        ClusterLinks::from_parsed_mat_file(llr.view(), priors.as_slice(), assoc_local.view());

    let lbp: Rc<dyn AssociationSolver> = Rc::new(Lbp);
    let mh_solver: Rc<dyn MhAssociationSolver> = Rc::new(HypCondSolver::new(lbp));
    let asso_info = AssociationInfo::new(llr, priors);
    let solver = MeasCondSolver::new(asso_info, mh_solver, cluster_links);

    let output = solver.compute_marginals();

    let marginals: Vec<Vec<f64>> = output
        .marginals()
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();
    let theta_posteriors: Vec<Vec<f64>> = output
        .theta_posteriors()
        .iter()
        .map(|t| t.to_vec())
        .collect();

    let result = json!({
        "marginals": marginals,
        "likelihood": output.likelihood(),
        "theta_posteriors": theta_posteriors,
    });

    println!("{}", serde_json::to_string(&result).unwrap());
}
