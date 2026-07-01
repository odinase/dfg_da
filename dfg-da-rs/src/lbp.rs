use std::mem::MaybeUninit;

use ndarray::Zip;
use ndarray::{prelude::*, s, Array2, ArrayBase, Data, Ix2};

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
    let iter = 0;

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
    }

    let mut unnormed_probs = Array2::uninit((n, mp1));
    let psi_times_msg = &psi * &nu;

    unnormed_probs.column_mut(0).fill(MaybeUninit::new(1.0));
    psi_times_msg.assign_to(unnormed_probs.slice_mut(s![.., 1..]));

    let probs = {
        let mut unnormed_probs = unsafe { unnormed_probs.assume_init() };
        unnormed_probs = &unnormed_probs
            / &unnormed_probs
                .sum_axis(Axis(1))
                .insert_axis(Axis(1))
                .broadcast(unnormed_probs.dim())
                .unwrap();
        unnormed_probs
    };

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
