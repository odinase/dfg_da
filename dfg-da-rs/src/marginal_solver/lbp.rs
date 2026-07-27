use crate::marginal_solver as ms;
use std::mem::MaybeUninit;

use ndarray::Zip;
use ndarray::{Array2, ArrayBase, Data, Ix2, prelude::*, s};

// Add parameters here later?
pub struct Lbp;

impl ms::AssociationSolver for Lbp {
    fn compute_marginals(&self, llr: ArrayView2<f64>) -> ms::AssociationMarginalOutput {
        let (marginals, loglikelihood) = lbp_marginal(&llr);
        ms::AssociationMarginalOutput {
            marginals,
            likelihood: loglikelihood.exp(),
        }
    }
}

pub fn lbp_marginal<S: Data<Elem = f64>>(llr: &ArrayBase<S, Ix2>) -> (Array2<f64>, f64) {
    let (n, mp1) = llr.dim();
    let m = mp1 - 1;
    if n == 0 || m == 0 {
        return (Array2::zeros((n, mp1)), 1.0);
    }

    let psi = (&llr.slice(s![.., 1..])
        - &llr
            .column(0)
            .insert_axis(Axis(1))
            .broadcast((n, m))
            .unwrap())
        .exp();

    let mut mu = &psi
        / (1.0
            + (&psi
                .sum_axis(Axis(1))
                .insert_axis(Axis(1))
                .broadcast(psi.dim())
                .unwrap()
                - &psi));

    let mut nu = 1.0
        / (1.0
            + (&mu
                .sum_axis(Axis(0))
                .insert_axis(Axis(0))
                .broadcast(psi.dim())
                .unwrap()
                - &mu));

    let thresh = 1e-5;
    let mut term_val = f64::INFINITY;
    let max_iter = 300;
    let mut iter = 0;

    while term_val >= thresh && iter < max_iter {
        let psi_times_msg = &psi * &nu;
        mu = &psi
            / (1.0
                + (&psi_times_msg
                    .sum_axis(Axis(1))
                    .insert_axis(Axis(1))
                    .broadcast(psi_times_msg.dim())
                    .unwrap()
                    - &psi_times_msg));

        let new_nu = 1.0
            / (1.0
                + (&mu
                    .sum_axis(Axis(0))
                    .insert_axis(Axis(0))
                    .broadcast(psi.dim())
                    .unwrap()
                    - &mu));

        let bmesg_ratio = &new_nu / &nu;
        let max_ratio = bmesg_ratio.fold(f64::NEG_INFINITY, |m, &x| m.max(x));
        let min_ratio = bmesg_ratio.fold(f64::INFINITY, |m, &x| m.min(x));
        let max_abs = max_ratio.max(1.0 / min_ratio);
        term_val = max_abs.ln();

        nu = new_nu;
        iter += 1;
    }

    let mut unnormed_probs = Array2::uninit((n, mp1));
    let psi_times_msg = &psi * &nu;

    unnormed_probs.column_mut(0).fill(MaybeUninit::new(1.0));
    psi_times_msg.assign_to(unnormed_probs.slice_mut(s![.., 1..]));

    let probs = {
        let unnormed_probs = unsafe { unnormed_probs.assume_init() };
        &unnormed_probs
            / &unnormed_probs
                .sum_axis(Axis(1))
                .insert_axis(Axis(1))
                .broadcast(unnormed_probs.dim())
                .unwrap()
    };

    // Since psi is normalized by psi(0) (misdetection) we need to add it back here
    let log_misdetection_weight = llr.column(0).sum();
    let ln_z_bethe = bethe_loglikelihood(&psi, &mu, &nu) + log_misdetection_weight;

    (probs, ln_z_bethe)
}

fn bethe_loglikelihood<D: Data<Elem = f64>>(
    psi: &ArrayBase<D, Ix2>,
    mu: &ArrayBase<D, Ix2>,
    nu: &ArrayBase<D, Ix2>,
) -> f64 {
    let zt = compute_zt(psi, nu);
    let zj = compute_zj(mu);
    let ztj = compute_ztj(psi, mu, nu);
    let m = zj.len();
    let n = zt.len();
    // Bethe free energy F, then return the loglikelihood −F (matches the Python reference
    // `bethe_loglikelihood_single_cluster`).

    let f = (m - 1) as f64 * zt.mapv(f64::ln).sum() + (n - 1) as f64 * zj.mapv(f64::ln).sum()
        - ztj.mapv(f64::ln).sum();
    -f
}

fn compute_zt<D: Data<Elem = f64>>(psi: &ArrayBase<D, Ix2>, nu: &ArrayBase<D, Ix2>) -> Array1<f64> {
    let psi_times_msg = psi * nu;
    1.0 + &psi_times_msg.sum_axis(Axis(1))
}

fn compute_zj<D: Data<Elem = f64>>(mu: &ArrayBase<D, Ix2>) -> Array1<f64> {
    1.0 + mu.sum_axis(Axis(0))
}

fn compute_ztj<D: Data<Elem = f64>>(
    psi: &ArrayBase<D, Ix2>,
    mu: &ArrayBase<D, Ix2>,
    nu: &ArrayBase<D, Ix2>,
) -> Array2<f64> {
    let psi_times_msg = psi * nu;
    // z_t = w_0 + (Σ_j wtm[i,j] − wtm[i,j]),  with the misdetection weight w_0 = 1.
    let z_t = 1.0
        + (&psi_times_msg
            .sum_axis(Axis(1))
            .insert_axis(Axis(1))
            .broadcast(psi_times_msg.dim())
            .unwrap()
            - &psi_times_msg);
    // z_j = 1 + (Σ_i mu[i,j] − mu[i,j])
    let z_j = 1.0
        + (&mu
            .sum_axis(Axis(0))
            .insert_axis(Axis(0))
            .broadcast(mu.dim())
            .unwrap()
            - mu);
    &z_t * &z_j + psi
}

/// mu[i,j] = psi[i,j] / (1 + Σ_k w[i,k] − w[i,j])   — row reduction (Axis 1)
/// `w` is `psi` for the initial pass and `psi * nu` inside the loop.
fn fill_mu(mu: &mut Array2<f64>, psi: &Array2<f64>, w: &Array2<f64>) {
    Zip::from(mu.rows_mut())
        .and(psi.rows())
        .and(w.rows())
        .for_each(|mut mu_row, psi_row, w_row| {
            let s = w_row.sum();
            Zip::from(&mut mu_row)
                .and(&psi_row)
                .and(&w_row)
                .for_each(|mu, &p, &w| *mu = p / (1.0 + s - w));
        });
}

/// col_sum[j] = Σ_i mu[i,j]   — column reduction (Axis 0), in place.
fn column_sums(mu: &Array2<f64>, col_sum: &mut Array1<f64>) {
    col_sum.fill(0.0);
    for row in mu.rows() {
        *col_sum += &row;
    }
}

/// nu[i,j] = 1 / (1 + col_sum[j] − mu[i,j])
fn fill_nu(nu: &mut Array2<f64>, mu: &Array2<f64>, col_sum: &Array1<f64>) {
    Zip::from(nu.rows_mut())
        .and(mu.rows())
        .for_each(|mut nu_row, mu_row| {
            Zip::from(&mut nu_row)
                .and(&mu_row)
                .and(col_sum)
                .for_each(|nu, &mu, &cs| *nu = 1.0 / (1.0 + cs - mu));
        });
}

pub fn lbp_marginal_zip<S: Data<Elem = f64>>(llr: &ArrayBase<S, Ix2>) -> (Array2<f64>, f64) {
    let (n, mp1) = llr.dim();
    let m = mp1 - 1;
    if n == 0 || m == 0 {
        return (Array2::zeros((n, mp1)), 1.0);
    }

    // psi[i,j] = exp(llr[i, j+1] - llr[i, 0])   (computed once, loop-invariant)
    let psi = (&llr.slice(s![.., 1..]) - &llr.slice(s![.., 0..1])).mapv(f64::exp);

    // Every iteration buffer allocated exactly once.
    let mut mu = Array2::<f64>::zeros((n, m));
    let mut nu = Array2::<f64>::zeros((n, m));
    let mut new_nu = Array2::<f64>::zeros((n, m));
    let mut wtm = Array2::<f64>::zeros((n, m)); // psi * nu
    let mut col_sum = Array1::<f64>::zeros(m);

    // Initial messages.
    fill_mu(&mut mu, &psi, &psi); // w = psi
    column_sums(&mu, &mut col_sum);
    fill_nu(&mut nu, &mu, &col_sum);

    let thresh = 1e-5;
    let max_iter = 300;
    let mut term_val = f64::INFINITY;
    let mut iter = 0; // NB: original never incremented this; fixed so max_iter binds

    while term_val >= thresh && iter < max_iter {
        Zip::from(&mut wtm)
            .and(&psi)
            .and(&nu)
            .for_each(|w, &p, &nu| *w = p * nu);

        fill_mu(&mut mu, &psi, &wtm);
        column_sums(&mu, &mut col_sum);
        fill_nu(&mut new_nu, &mu, &col_sum);

        // Convergence on new_nu / nu, fused into one pass (no temporary).
        let (mut max_r, mut min_r) = (f64::NEG_INFINITY, f64::INFINITY);
        Zip::from(&new_nu).and(&nu).for_each(|&a, &b| {
            let r = a / b;
            max_r = max_r.max(r);
            min_r = min_r.min(r);
        });
        term_val = max_r.max(1.0 / min_r).ln();

        std::mem::swap(&mut nu, &mut new_nu); // adopt new_nu, no copy
        iter += 1;
    }

    // Marginals: col 0 = 1, cols 1.. = psi*nu, then row-normalise in place.
    Zip::from(&mut wtm)
        .and(&psi)
        .and(&nu)
        .for_each(|w, &p, &nu| *w = p * nu);

    let mut probs = Array2::<f64>::zeros((n, mp1));
    probs.slice_mut(s![.., 0]).fill(1.0);
    probs.slice_mut(s![.., 1..]).assign(&wtm);
    Zip::from(probs.rows_mut()).for_each(|mut row| {
        let s = row.sum();
        row.map_inplace(|x| *x /= s);
    });

    (probs, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*; // brings the parent module's items into scope

    #[test]
    fn test_lbp_marginal() {
        let llr = arr2(&[
            [-0.46, 5.78, 3.78],
            [-0.52, 6.37, 6.57],
            [-0.60, 7.58, 2.58],
        ]);

        let (probs, norm) = lbp_marginal(&llr);
        println!("probs:\n{:?}", probs);
    }
}
