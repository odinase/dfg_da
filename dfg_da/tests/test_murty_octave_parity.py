"""Parity of the Python Murty port against the original MATLAB, run under Octave.

This is the correctness ground truth for ``dfg_da.murty_marginals``: the branch-and-bound
it builds on (``pmbm-cm-python/cm/branchbound.py``) states in its own docstring that it is
"equivalent, not bit-exact" and was never validated against a MATLAB reference.

Generate the reference once (needs Octave: ``brew install octave`` /
``sudo apt install octave``), from the repo root::

    octave --no-gui --quiet --eval "addpath('pmbm-cm-matlab', 'pmbm-cm-matlab/at612/jmpdFunctions', 'octave'); murty_reference('octave_murty_reference', [10 20 50 100 150], 'octave/reference_files.txt')"

The tests skip when that output is absent, so a machine without Octave still runs the
rest of the suite.

Two levels are checked:

* **Level A** -- the raw ``branchAndBoundExplore`` output per supercluster, which
  localises a failure to the branch-and-bound rather than to the marginal accumulation.
* **Level B** -- the end-to-end ``a`` (marginals) and ``Z`` (normalization constant).

Level A compares hypotheses *grouped by score*: ``assign2D.m`` and
``cm.bbhelpers.assign2d`` may resolve an equal-cost assignment differently, which
reorders equally-scored hypotheses without changing anything downstream (both ``a`` and
``Z`` are order-invariant sums). A difference in the multiset of scores, or in ``a``/``Z``,
is a real port bug.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import dfg_da.murty_marginals as mm

REPO_ROOT = Path(__file__).resolve().parents[2]
REF_ROOT = REPO_ROOT / "octave_murty_reference"
FILE_LIST = REPO_ROOT / "octave" / "reference_files.txt"

ATOL = 1e-9

pytestmark = pytest.mark.skipif(
    not REF_ROOT.exists(),
    reason=f"no Octave reference at {REF_ROOT}; see this module's docstring")


def _cases():
    if not REF_ROOT.exists() or not FILE_LIST.exists():
        return []
    names = [Path(l.strip()).name for l in FILE_LIST.read_text().splitlines() if l.strip()]
    out = []
    for k in mm.K_SWEEP:
        for name in names:
            ref = REF_ROOT / f"nHypoTotalMax_{k}" / name
            if ref.exists():
                out.append(pytest.param(k, name, id=f"K{k}-{name[:-4]}"))
    return out


CASES = _cases()


def _scan(name):
    return REPO_ROOT / "data" / "pmbm_output_files" / name


def _cell(ref, key):
    """Unpack an Octave 1xN cell array into a list of raveled float arrays."""
    raw = ref[key]
    return [np.asarray(c, dtype=float).ravel() for c in np.asarray(raw).ravel()]


def _group_by_score(scores, hypo_sets):
    """Map rounded score -> multiset of hypothesis track-sets sharing it."""
    groups = {}
    for s, h in zip(scores, hypo_sets):
        groups.setdefault(round(float(s), 9), []).append(frozenset(h))
    return {s: sorted(v, key=sorted) for s, v in groups.items()}


def _split(flat, cards):
    out, b = [], 0
    for c in cards.astype(int):
        out.append(flat[b:b + c])
        b += c
    return out


@pytest.mark.parametrize("k,name", CASES)
def test_branch_and_bound_parity(k, name):
    """Level A: per-supercluster branchAndBoundExplore output."""
    from cm.branchbound import branch_and_bound_explore

    ref = loadmat(str(REF_ROOT / f"nHypoTotalMax_{k}" / name))
    ws = loadmat(str(_scan(name)))
    ravel = lambda key, dt: np.asarray(ws[key], dtype=dt).ravel()

    ref_prob = _cell(ref, "bbProbLogLocal")
    ref_hypos = _cell(ref, "bbHyposLocal")
    ref_cards = _cell(ref, "bbHyposCardLocal")

    n_superclusters = int(np.asarray(ref["nSuperclusters"]).ravel()[0])
    assert len(ref_prob) == n_superclusters

    for iC in range(1, n_superclusters + 1):
        hl, hcl, pll = branch_and_bound_explore(
            ravel("hypos", int), ravel("hyposCard", int), ravel("clusters", int),
            ravel("clustersCard", int), ravel("probLogHypos", float), iC,
            np.asarray(ws["assocLocal"], dtype=float),
            np.asarray(ws["gainMatPostC"], dtype=float),
            ravel("indicesOfNewbornTracks", int), k,
            np.asarray(ws["trackNumberLookup"], dtype=float),
            int(ravel("k", float)[0]))

        r_pll, r_hl, r_hcl = ref_prob[iC - 1], ref_hypos[iC - 1], ref_cards[iC - 1]

        assert hcl.size == r_hcl.size, (
            f"{name} K={k} iC={iC}: {hcl.size} hypotheses vs {r_hcl.size} in Octave")
        np.testing.assert_allclose(
            np.sort(pll)[::-1], np.sort(r_pll)[::-1], rtol=0, atol=ATOL,
            err_msg=f"{name} K={k} iC={iC}: score multiset differs")

        got = _group_by_score(pll, _split(hl.astype(float), hcl))
        want = _group_by_score(r_pll, _split(r_hl, r_hcl))
        assert got == want, f"{name} K={k} iC={iC}: hypothesis track sets differ"


@pytest.mark.parametrize("k,name", CASES)
def test_marginals_and_z_parity(k, name):
    """Level B: end-to-end marginals and normalization constant."""
    ref = loadmat(str(REF_ROOT / f"nHypoTotalMax_{k}" / name))
    ws = loadmat(str(_scan(name)))

    num_measurements = ws["measurements"].shape[1]
    num_tracks = ws["trackNumberLookup"].shape[0] - num_measurements

    out = mm.murty_marginals_likelihood(ws, num_tracks, num_measurements, k)

    a_ref = np.asarray(ref["a"], dtype=float)
    assert out.marginals.shape == a_ref.shape
    np.testing.assert_allclose(out.marginals, a_ref, rtol=0, atol=ATOL,
                               err_msg=f"{name} K={k}: marginals differ")
    np.testing.assert_allclose(out.likelihood,
                               float(np.asarray(ref["Z"]).ravel()[0]),
                               rtol=1e-9, err_msg=f"{name} K={k}: Z differs")
