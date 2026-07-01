// Basic end-to-end check that the C++ -> C API -> Rust path works.
// The exact numeric cross-check against py_dfg_da is done at the Python level
// (dfg-da-py) where both the compiled extension and this Rust path are callable.

use dfg_da_rs::{lbp_single_cluster, Hypotheses};

#[test]
fn single_cluster_runs() {
    // 2 tracks, 2 measurements, "edmund" layout: [meas(2) | 2x2 existence block],
    // misdetection log-reward on the existence diagonal, -inf off it.
    let ninf = f64::NEG_INFINITY;
    let rows = 2usize;
    let cols = 4usize; // m (2) + n (2)
    #[rustfmt::skip]
    let reward = vec![
        // m0    m1    exist0  exist1
        0.30,  0.10,  0.00,   ninf,
        0.05,  0.40,  ninf,   0.00,
    ];

    let mut hyps = Hypotheses::new().unwrap();
    hyps.add(&[1, 2], 0.0); // one prior hypothesis over tracks 1 and 2
    assert_eq!(hyps.len(), 1);

    let out = lbp_single_cluster(&reward, rows, cols, &hyps, 300).unwrap();
    let z = out.bethe_norm_const();
    assert!(z.is_finite() && z > 0.0, "expected finite positive Z, got {z}");
}
