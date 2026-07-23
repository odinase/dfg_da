use ndarray::{Array, ArrayBase, Axis, Data, DataMut, Dimension};

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