"""
solvers.py
==========

Single-cluster, multi-hypothesis association solvers.  Every solver implements
the same call contract used throughout ``odinase/dfg_da``::

    solver(R_cluster, prior_hypotheses, enforce_meas=()) ->
        (marginals, theta_posterior, likelihood)

where

* ``R_cluster``  : (n, m+1) log-reward matrix for the cluster (col 0 misdetect),
* ``prior_hypotheses`` : a :class:`prior_hypotheses.Hypotheses`,
* ``enforce_meas`` : set of 1-based measurement columns that, under the *disjoint*
  event space (thesis Eq. (7.5) / Section 7.4), are forced to be a detection of
  some existing track (false alarm disallowed).  Empty for the overlapping
  event space (Eq. (7.29)).
* ``marginals``  : (n, m+2) array, columns ``[misdetect, meas_1..m, nonexist]``,
* ``theta_posterior`` : (H,) posterior over the cluster's prior hypotheses,
* ``likelihood`` : the cluster-conditioned normalization constant
  ``p(Z^c | d, Z_{1:k-1})`` (thesis Eq. (7.17)).

Three solvers are provided:

``ExactEnumerationSolver``  - brute-force ground truth (marginals AND constant).
``EHM2Solver``              - marginals from ``pyehm``'s EHM2 (the thesis/codebase
                              baseline), constant from the exact permanent.
``LBPBetheSolver``          - Williams-style single-cluster LBP with the Bethe
                              pseudodual (thesis Corollary 2) as the constant.
"""
from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence, Tuple

import numpy as np

from .prior_hypotheses import Hypotheses

Result = Tuple[np.ndarray, np.ndarray, float]


# ---------------------------------------------------------------------------
# Exact single-hypothesis primitives
# ---------------------------------------------------------------------------
def _single_hyp_exact(R_sub: np.ndarray, enforce_meas: Iterable[int]) -> Tuple[np.ndarray, float]:
    """Exact JPDA marginals + normalization constant for ONE hypothesis.

    ``R_sub`` is (n_e, m+1) for the ``n_e`` existing tracks of the hypothesis.
    Returns ``(marg, Z)`` where ``marg`` is (n_e, m+1) unnormalized
    (sum over assignments of weight * indicator) and ``Z`` the constant.

    The association is a weighted partial matching: every track is either
    misdetected (column 0, weight ``exp(R[t,0])``) or detects a gated, still
    free measurement (weight ``exp(R[t,j])``); at most one track per measurement.
    """
    n_e, mp1 = R_sub.shape
    m = mp1 - 1
    enforce = set(int(j) for j in enforce_meas)
    W = np.exp(R_sub)  # weights; non-gated entries are exp(-inf) = 0

    Z = 0.0
    marg = np.zeros((n_e, mp1))

    used = np.zeros(m + 1, dtype=bool)  # measurement-used flags (index 1..m)

    def dfs2(t: int, weight: float, assign: list):
        nonlocal Z
        if t == n_e:
            if any(not used[j] for j in enforce):
                return
            Z += weight
            for (tt, jj) in assign:
                marg[tt, jj] += weight
            return
        w0 = W[t, 0]
        if w0 > 0.0:
            assign.append((t, 0))
            dfs2(t + 1, weight * w0, assign)
            assign.pop()
        for j in range(1, m + 1):
            if used[j] or W[t, j] <= 0.0:
                continue
            used[j] = True
            assign.append((t, j))
            dfs2(t + 1, weight * W[t, j], assign)
            assign.pop()
            used[j] = False

    dfs2(0, 1.0, [])
    return marg, Z


def _permanent_constant(R_sub: np.ndarray, enforce_meas: Iterable[int]) -> float:
    """Just the normalization constant (reuses the exact enumeration)."""
    _, Z = _single_hyp_exact(R_sub, enforce_meas)
    return Z


# ---------------------------------------------------------------------------
# Multi-hypothesis assembly (thesis Eq. (7.16)/(7.17), mirrors multihypothesis_ehm2)
# ---------------------------------------------------------------------------
def _assemble_multihypothesis(
    R_cluster: np.ndarray,
    prior_hypotheses: Hypotheses,
    single_hyp_fn,
    enforce_meas: Iterable[int],
    reindex: bool = True,
) -> Result:
    if reindex:
        prior_hypotheses.reindex_tracks()
    n, mp1 = R_cluster.shape
    all_idx = np.arange(n)
    marginals = np.zeros((n, mp1 + 1))            # + nonexistence column
    cond = np.empty((n, mp1 + 1))
    L_h = np.empty(len(prior_hypotheses))
    Zc = 0.0
    enforce = set(int(j) for j in enforce_meas)

    for k, h in enumerate(prior_hypotheses):
        ex = (np.array(h.tracks(), dtype=int) - 1)
        prob = h.probability()
        if ex.size > 0:
            R_sub = R_cluster[ex]
            # A measurement can only be enforced as a detection if some existing
            # track of THIS hypothesis gates it; otherwise this hypothesis is
            # incompatible with the conditioning -> likelihood 0.
            gated_here = np.isfinite(R_sub[:, 1:]).any(axis=0)
            if any((j - 1 >= R_sub.shape[1] - 1) or not gated_here[j - 1] for j in enforce):
                jpda = np.zeros((ex.size, mp1))
                Lk = 0.0
            else:
                marg_un, Lk = single_hyp_fn(R_sub, enforce)
                jpda = marg_un / Lk if Lk > 0 else np.zeros((ex.size, mp1))
        else:
            jpda = np.zeros((0, mp1))
            Lk = 0.0 if enforce else 1.0  # empty hyp can't host an enforced meas

        nonex = np.setdiff1d(all_idx, ex, assume_unique=False)
        cond[ex] = np.hstack((jpda, np.zeros((ex.size, 1))))
        cond[nonex] = np.hstack((np.zeros((nonex.size, mp1)), np.ones((nonex.size, 1))))

        L_h[k] = Lk
        if Lk > 0.0:
            marginals += cond * Lk * prob
            Zc += Lk * prob

    priors = prior_hypotheses.hypothesis_probabilites()
    theta_post = priors * L_h
    s = theta_post.sum()
    theta_post = theta_post / s if s > 0 else priors.copy()

    s = marginals.sum(axis=1, keepdims=True)
    marginals = np.divide(marginals, s, out=np.zeros_like(marginals), where=s > 0)
    return marginals, theta_post, Zc


# ---------------------------------------------------------------------------
# Public solvers
# ---------------------------------------------------------------------------
class ExactEnumerationSolver:
    """Ground-truth solver via exhaustive association enumeration."""

    def __call__(self, R_cluster, prior_hypotheses, enforce_meas=(), reindex=True) -> Result:
        return _assemble_multihypothesis(
            R_cluster, prior_hypotheses, _single_hyp_exact, enforce_meas, reindex
        )


class EHM2Solver:
    """Marginals from ``pyehm`` EHM2; normalization constant from the permanent.

    This is the baseline referenced by the thesis / codebase
    (``multihypothesis_ehm2``).  The installed ``pyehm`` (>=2.2) returns only
    association probabilities, so the per-hypothesis constant is obtained from
    the exact permanent (identical to the value EHM2's net computes internally).
    """

    def __init__(self):
        from pyehm.core import EHM2  # imported lazily so the package works w/o it
        self._EHM2 = EHM2

    def _single_hyp(self, R_sub, enforce):
        # EHM2 has no notion of "enforce a detection".  We realise enforcement on
        # the *track-oriented* matrix by setting, for each enforced measurement,
        # the misdetection contribution of all tracks to still allow misdetect
        # (EHM cannot forbid FA directly); so we fall back to the exact b-version
        # transposition only for the constant, and use EHM2 on the (un-enforced)
        # matrix for marginals when nothing is enforced.  When enforcement is
        # active we use the exact enumeration for marginals to stay correct.
        if enforce:
            return _single_hyp_exact(R_sub, enforce)
        lik = np.asfortranarray(np.exp(R_sub))
        val = np.asfortranarray((lik > 0.0).astype(np.int32))
        probs = self._EHM2().run(val, lik)             # (n_e, m+1) normalized
        Z = _permanent_constant(R_sub, enforce)        # exact constant
        return probs * Z, Z                            # unnormalized marg, Z

    def __call__(self, R_cluster, prior_hypotheses, enforce_meas=(), reindex=True) -> Result:
        return _assemble_multihypothesis(
            R_cluster, prior_hypotheses, self._single_hyp, enforce_meas, reindex
        )


class LBPBetheSolver:
    """Williams-style single-cluster LBP + Bethe-pseudodual constant.

    Implements the standard single-cluster, single-hypothesis loopy belief
    propagation of Williams & Lau [8] (thesis Eq. (6.1)) and estimates the
    per-hypothesis normalization constant with the Bethe pseudodual of
    Corollary 2 (Eq. (6.14)-(6.17)).  Wrapped by total probability over prior
    hypotheses (Eq. (7.16)).

    Note: under the *overlapping* event space (no enforcement) this solver never
    sees an infinite-SNR (enforced-detection) sub-problem, which is precisely
    why the thesis prefers it over the transposed exact formulation (Sec. 7.5).
    """

    def __init__(self, max_iter: int = 1000, tol: float = 1e-10):
        self.max_iter = max_iter
        self.tol = tol

    def _lbp_single(self, R_sub, enforce):
        n_e, mp1 = R_sub.shape
        m = mp1 - 1
        # Work with raw weights and keep the misdetection weight psi^t(0) = w0
        # EXPLICIT rather than normalizing it to 1.  The thesis normalizes by
        # misdetection for notational brevity (Eq. (6.15)-(6.17)), but that is
        # only valid when w0 > 0.  Under the overlapping event space a track may
        # be forbidden to misdetect (R[t,0] = -inf  =>  w0 = 0), so we carry w0
        # through every expression instead -- this reduces to the thesis form
        # when w0 = 1 and stays correct when w0 = 0.
        w0 = np.exp(R_sub[:, 0])                         # (n_e,), may be 0
        W = np.exp(R_sub[:, 1:])                         # (n_e, m), 0 where -inf
        W = np.nan_to_num(W, nan=0.0, posinf=0.0)
        gate = W > 0.0

        mu = W.copy()                                    # mu[t,j] = mu_{t->j}
        for _ in range(self.max_iter):
            col = mu.sum(axis=0)                          # (m,)
            nu = 1.0 / (1.0 + (col[None, :] - mu))       # (n_e, m)
            s = (W * nu).sum(axis=1)                      # (n_e,)
            denom = w0[:, None] + (s[:, None] - W * nu)
            mu_new = np.where(gate, W / np.where(denom > 0, denom, 1.0), 0.0)
            if np.max(np.abs(mu_new - mu)) < self.tol:
                mu = mu_new
                break
            mu = mu_new

        col = mu.sum(axis=0)
        nu = 1.0 / (1.0 + (col[None, :] - mu))

        # Marginals  p(a^t = 0) ∝ w0,  p(a^t = j) ∝ W(j) nu_{j->t}  (Eq. (6.2)).
        marg = np.zeros((n_e, mp1))
        marg[:, 0] = w0
        marg[:, 1:] = W * nu

        # Bethe pseudodual constant (Corollary 2, Eq. (6.14)-(6.17)), written with
        # explicit w0 and summed over ALL (t, j) pairs (the consistency factor
        # connects every track to every measurement; non-gated edges have W = 0).
        Zt = w0 + (W * nu).sum(axis=1)                    # (n_e,)
        Zj = 1.0 + mu.sum(axis=0)                         # (m,)
        sum_t_excl = col[None, :] - mu                    # (n_e, m)
        sum_j_excl = (W * nu).sum(axis=1)[:, None] - W * nu
        Ztj = (1.0 + sum_t_excl) * (w0[:, None] + sum_j_excl) + W
        with np.errstate(divide="ignore"):
            lnZt = np.where(Zt > 0, np.log(Zt), 0.0)
            lnZj = np.where(Zj > 0, np.log(Zj), 0.0)
            lnZtj = np.log(np.clip(Ztj, 1e-300, None))
        F = (m - 1) * lnZt.sum() + (n_e - 1) * lnZj.sum() - lnZtj.sum()
        Z = np.exp(-F)                                    # Bethe constant
        return marg * Z, Z

    def _single_hyp(self, R_sub, enforce):
        if enforce:
            # The overlapping event space never enforces; if a caller asks for
            # enforcement we defer to the exact primitive for correctness.
            return _single_hyp_exact(R_sub, enforce)
        return self._lbp_single(R_sub, enforce)

    def __call__(self, R_cluster, prior_hypotheses, enforce_meas=(), reindex=True) -> Result:
        return _assemble_multihypothesis(
            R_cluster, prior_hypotheses, self._single_hyp, enforce_meas, reindex
        )
