use ndarray::{Array, ArrayBase, Axis, Data, DataMut, Dimension, prelude::*};

/// Normalizes each lane along the last axis in place so it sums to 1.
///
/// - 1-D: the whole array is normalized to sum to 1.
/// - 2-D: each row is normalized to sum to 1.
/// - N-D: every 1-D lane along the last axis is normalized independently.
///
/// # Panics
/// Panics if any lane sum is zero, NaN, or infinite.
pub fn normalize_rows_inplace<S, D>(mat: &mut ArrayBase<S, D>)
where
    S: DataMut<Elem = f64>,
    D: Dimension,
{
    let last = Axis(mat.ndim() - 1);

    // Validate every lane first, so a panic leaves the array unmodified.
    for (i, lane) in mat.lanes(last).into_iter().enumerate() {
        let sum = lane.sum();
        assert!(
            sum != 0.0 && sum.is_finite(),
            "cannot normalize lane {i}: sum is {sum}"
        );
    }

    for mut lane in mat.lanes_mut(last) {
        let sum = lane.sum();
        lane /= sum;
    }
}

/// Returns a copy of `mat` with each lane along the last axis normalized to sum to 1.
///
/// See [`normalize_rows_inplace`] for the exact semantics.
///
/// # Panics
/// Panics if any lane sum is zero, NaN, or infinite.
pub fn normalized_rows<S, D>(mat: &ArrayBase<S, D>) -> Array<f64, D>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    let mut out = mat.to_owned();
    normalize_rows_inplace(&mut out);
    out
}

pub fn edmund_to_lc<D: Data<Elem = f64>>(llr_edmund: &ArrayBase<D, Ix2>) -> Array2<f64> {
    let (n, mpn) = llr_edmund.dim();
    let m = mpn - n;
    let mp1 = m + 1;
    let misdetection_block = llr_edmund.slice(s![.., m..]);
    let right_diag = misdetection_block.diag();
    let mut llr_lc = Array2::zeros((n, mp1));
    llr_lc.column_mut(0).assign(&right_diag);
    llr_lc
        .slice_mut(s![.., 1..])
        .assign(&llr_edmund.slice(s![.., ..m]));

    llr_lc
}
