"""Tests for the multicluster Murty marginals port (``dfg_da.murty_marginals``).

These are self-consistency and hand-checkable tests. The parity check against the
original MATLAB lives in ``test_murty_octave_parity.py``.
"""

from pathlib import Path

import numpy as np
import pytest

import dfg_da.murty_marginals as mm

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN = REPO_ROOT / "data" / "pmbm_output_files" / "priorLikelihood10.mat"


def _two_cluster_contention_ws():
    """Two single-track clusters contending for one measurement, merged into one
    supercluster.

    Reward layout (``gainMatPostC``): column 0 is the measurement, columns 1-2 the
    misdetection/existence block, row 2 the newborn row for the measurement.

        track 1 -> meas : 3.0        track 1 misdetect : 0.0
        track 2 -> meas : 2.0        track 2 misdetect : 0.0
        newborn on meas : 0.0

    so the three feasible global hypotheses score 3, 2 and 0.
    """
    inf = np.inf
    return {
        "hypos": np.array([[1, 2]], dtype=float),
        "hyposCard": np.array([[1, 1]], dtype=float),
        "clusters": np.array([[1, 2]], dtype=float),
        "clustersCard": np.array([[1, 1]], dtype=float),
        "probLogHypos": np.array([[0.0, 0.0]]),
        "assocLocal": np.array([[1.0, 1.0], [1.0, 0.0]]),
        "gainMatPostC": np.array([[3.0, 0.0, -inf],
                                  [2.0, -inf, 0.0],
                                  [0.0, 0.0, 0.0]]),
        "indicesOfNewbornTracks": np.array([[5]], dtype=float),
        # posterior track id per feasible (row, col) cell of gainMatPostC
        "trackNumberLookup": np.array([[1.0, 2.0, np.nan],
                                       [3.0, np.nan, 4.0],
                                       [5.0, np.nan, np.nan]]),
        # last row: which measurement each posterior track claims (0 = none)
        "meaHistColNew": np.array([[1.0, 0.0, 1.0, 0.0, 1.0]]),
        "k": np.array([[3.0]]),
    }


def test_two_cluster_contention_marginals_and_z():
    out = mm.murty_marginals_likelihood(_two_cluster_contention_ws(),
                                        num_tracks=2, num_measurements=1,
                                        n_hypo_total_max=30)

    # All three hypotheses fit under K=30, so this is the exact answer.
    w = np.exp([3.0, 2.0, 0.0])
    p1, p2, p3 = w / w.sum()

    # columns: [misdetection, measurement 1, nonexistence]
    expected = np.array([[p2 + p3, p1, 0.0],
                         [p1 + p3, p2, 0.0]])
    assert out.marginals.shape == (2, 3)
    np.testing.assert_allclose(out.marginals, expected, atol=1e-12)
    assert np.isclose(out.likelihood, w.sum())


def test_marginals_are_distributions_on_real_scan():
    md = _mat(SCAN)
    out = mm.murty_marginals_likelihood(md.ws, md.num_tracks, md.num_measurements, 150)

    assert out.marginals.shape == (md.num_tracks, md.num_measurements + 2)
    np.testing.assert_allclose(out.marginals.sum(axis=1), 1.0, atol=1e-9)
    assert (out.marginals >= -1e-12).all() and (out.marginals <= 1.0 + 1e-12).all()
    assert out.likelihood > 0.0


def test_z_and_marginals_converge_with_k():
    md = _mat(SCAN)
    sweep = mm.murty_sweep(md.ws, md.num_tracks, md.num_measurements)

    assert sorted(sweep) == sorted(mm.K_SWEEP)

    ks = sorted(sweep)
    zs = np.array([sweep[k].likelihood for k in ks])
    # Z is a truncated sum over the same hypothesis set, so it can only grow with K.
    assert (np.diff(zs) >= -1e-12).all(), zs

    # ... and the marginals must settle down as the truncation is relaxed.
    deltas = [np.abs(sweep[a].marginals - sweep[b].marginals).max()
              for a, b in zip(ks, ks[1:])]
    assert deltas[-1] <= deltas[0] + 1e-12, deltas


def test_murty_z_does_not_exceed_exact():
    md = _mat(SCAN)
    import dfg_da.marginals_computers as mc

    exact = mc.MulticlusterExactEHM2()(np.asfortranarray(md.reward_matrix_lc),
                                       md.prior_hypotheses_per_cluster,
                                       assocLocal=md.ws["assocLocal"].copy())
    out = mm.murty_marginals_likelihood(md.ws, md.num_tracks, md.num_measurements, 150)

    # Murty enumerates a subset of the hypotheses the exact solver sums over.
    assert out.likelihood <= exact.exact_normalization_constant * (1 + 1e-9)
    # Shape parity matters: plot_convergence_stats silently drops a method whose
    # marginals do not line up with the exact ones.
    assert out.marginals.shape == exact.exact_marginals.shape


def _mat(path):
    import dfg_da.stats_logger as sl
    if not path.exists():
        pytest.skip(f"{path} not available")
    return sl.MatFileParser(str(path), use_cpp=True)
