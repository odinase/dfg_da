"""Cross-language parity tests for the multicluster inclusion-exclusion solver.

Each shared fixture in `dfg-da-rs/tests/fixtures/*.json` is run through both the
Python reference (`MulticlusterEfficientMarginalsLBPInclusionExclusion` with the
Bethe LBP solver) and the Rust `MeasCondSolver` (invoked via the compiled
`ie_fixture` example). The two outputs — marginals, likelihood, and per-cluster
theta posteriors — are asserted equal. The stored golden `expected` values (also
produced by the Python reference) give a Python-side regression check.

Fixtures cover: an all-merging supercluster (`test_case3`), a mixed
merging + unmerging case (`two_merge_one_separate`), fully independent clusters
(`all_separate`), and edge cases (single hypothesis, empty/nonexistence
hypothesis). Regenerate with `.venv/bin/python dfg-da-rs/tests/fixtures/generate.py`.
"""
import json
import math
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import py_dfg_da as pdd

from dfg_da.marginals_computers import LBPMarginalsByTotalProbBethe
from dfg_da.cluster_conditioning_lbp_ie import (
    MulticlusterEfficientMarginalsLBPInclusionExclusion,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUST_CRATE = REPO_ROOT / "dfg-da-rs"
FIXTURE_DIR = RUST_CRATE / "tests" / "fixtures"

# Rust-vs-Python parity. LBP iterates to a convergence threshold
# (max_prob_diff_from_conv=1e-3 by default), so on loopy problems the two
# implementations can stop at slightly different fixed points. Agreement is
# therefore bounded by ~1e-3, not machine precision; this tolerance sits well
# below any structural error (which would be O(0.1)) but above convergence noise.
# Tree-structured / sparse fixtures still agree to ~1e-13.
PARITY_RTOL = 1e-4
PARITY_ATOL = 1e-6

# Python-vs-golden regression: identical code path, so effectively exact.
GOLDEN_RTOL = 1e-9
GOLDEN_ATOL = 1e-12

FIXTURE_FILES = sorted(FIXTURE_DIR.glob("*.json"))


# --- Fixture parsing ----------------------------------------------------------

def load_fixture(path: Path):
    data = json.loads(path.read_text())
    # JSON null encodes -inf (a measurement a track does not gate).
    R_LC = np.array(
        [[-math.inf if x is None else float(x) for x in row] for row in data["R_LC"]],
        dtype=float,
    )
    assoc_local = np.array(data["assoc_local"], dtype=int)
    return data, R_LC, assoc_local


def build_hypotheses_list(priors):
    return pdd.hypothesis.HypothesesList([
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis(list(h["tracks"]), h["log_weight"]) for h in cluster
        ])
        for cluster in priors
    ])


def python_reference(data, R_LC, assoc_local):
    prior_list = build_hypotheses_list(deepcopy(data["prior_hypotheses_per_cluster"]))
    solver = MulticlusterEfficientMarginalsLBPInclusionExclusion(
        R_LC=R_LC.copy(),
        prior_hypotheses_per_cluster=prior_list,
        assocLocal=assoc_local.copy(),
        lbp_solver=LBPMarginalsByTotalProbBethe(),
    )
    out = solver.compute_marginals_likelihood()
    return {
        "marginals": np.asarray(out.marginals, dtype=float),
        "likelihood": float(out.likelihood),
        "theta_posteriors": [np.asarray(t, dtype=float) for t in out.theta_posteriors],
    }


# --- Rust runner (compiled example) -------------------------------------------

@pytest.fixture(scope="session")
def rust_binary():
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo not found; cannot run the Rust parity comparison")

    build = subprocess.run(
        [cargo, "build", "--example", "ie_fixture"],
        cwd=RUST_CRATE,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        pytest.fail(f"failed to build Rust example:\n{build.stderr}")

    meta = subprocess.run(
        [cargo, "metadata", "--format-version", "1", "--no-deps"],
        cwd=RUST_CRATE,
        capture_output=True,
        text=True,
        check=True,
    )
    target_dir = Path(json.loads(meta.stdout)["target_directory"])
    binary = target_dir / "debug" / "examples" / "ie_fixture"
    assert binary.exists(), f"built example not found at {binary}"
    return binary


def rust_output(rust_binary, fixture_path: Path):
    proc = subprocess.run(
        [str(rust_binary), str(fixture_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(f"Rust example failed on {fixture_path.name}:\n{proc.stderr}")
    out = json.loads(proc.stdout)
    return {
        "marginals": np.asarray(out["marginals"], dtype=float),
        "likelihood": float(out["likelihood"]),
        "theta_posteriors": [np.asarray(t, dtype=float) for t in out["theta_posteriors"]],
    }


# --- Comparison helpers -------------------------------------------------------

def assert_outputs_close(a, b, ctx, rtol, atol):
    np.testing.assert_allclose(
        a["marginals"], b["marginals"], rtol=rtol, atol=atol,
        err_msg=f"{ctx}: marginals differ",
    )
    assert a["likelihood"] == pytest.approx(b["likelihood"], rel=rtol, abs=atol), (
        f"{ctx}: likelihood {a['likelihood']} != {b['likelihood']}"
    )
    assert len(a["theta_posteriors"]) == len(b["theta_posteriors"]), (
        f"{ctx}: number of clusters differ"
    )
    for c, (ta, tb) in enumerate(zip(a["theta_posteriors"], b["theta_posteriors"])):
        np.testing.assert_allclose(
            ta, tb, rtol=rtol, atol=atol,
            err_msg=f"{ctx}: theta posterior for cluster {c} differs",
        )


@pytest.mark.parametrize("fixture_path", FIXTURE_FILES, ids=[p.stem for p in FIXTURE_FILES])
def test_rust_matches_python(fixture_path, rust_binary):
    data, R_LC, assoc_local = load_fixture(fixture_path)

    py = python_reference(data, R_LC, assoc_local)
    rs = rust_output(rust_binary, fixture_path)

    # 1. Rust output equals the Python reference output.
    assert_outputs_close(
        rs, py, ctx=f"{fixture_path.stem} (rust vs python)",
        rtol=PARITY_RTOL, atol=PARITY_ATOL,
    )

    # 2. Python reference still equals the stored golden values (regression).
    expected = data["expected"]
    golden = {
        "marginals": np.asarray(expected["marginals"], dtype=float),
        "likelihood": float(expected["likelihood"]),
        "theta_posteriors": [np.asarray(t, dtype=float) for t in expected["theta_posteriors"]],
    }
    assert_outputs_close(
        py, golden, ctx=f"{fixture_path.stem} (python vs golden)",
        rtol=GOLDEN_RTOL, atol=GOLDEN_ATOL,
    )


def test_fixtures_exist():
    assert FIXTURE_FILES, f"no fixtures found in {FIXTURE_DIR}"
