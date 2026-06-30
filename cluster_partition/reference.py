"""
reference.py
============

Brute-force exact multi-cluster, multi-hypothesis marginals computed directly on
the *full* association graph (no clustering, no conditioning).  This is the
ground truth against which the partitioning methods are verified.

It enumerates the joint over per-cluster prior hypotheses (their product, thesis
Eq. (5.21)) and, for each combination, the global single-hypothesis association
with the *global* at-most-one constraint over all measurements (which is exactly
the coupling that linking measurements introduce).
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .prior_hypotheses import Hypotheses
from .solvers import _single_hyp_exact


def multicluster_exact_reference(
    R_LC: np.ndarray,
    prior_hypotheses_per_cluster: Sequence[Hypotheses],
) -> Tuple[np.ndarray, Dict[int, np.ndarray], float]:
    """Exact marginals, per-cluster theta posteriors, and total likelihood."""
    n, mp1 = R_LC.shape

    # Reindex each cluster's hypotheses for stable global track ids (they index
    # rows of R_LC as track_id - 1; we keep the original global ids here).
    hyp_lists: List[List] = []
    prior_lists: List[np.ndarray] = []
    for ph in prior_hypotheses_per_cluster:
        hyp_lists.append(list(ph))
        prior_lists.append(ph.hypothesis_probabilites())

    marginals = np.zeros((n, mp1 + 1))
    theta_acc = {c: np.zeros(len(prior_hypotheses_per_cluster[c])) for c in range(len(hyp_lists))}
    Z = 0.0

    for combo in product(*[range(len(h)) for h in hyp_lists]):
        prior_prod = 1.0
        existing: List[int] = []
        for c, h_idx in enumerate(combo):
            prior_prod *= prior_lists[c][h_idx]
            existing.extend(int(t) for t in hyp_lists[c][h_idx].tracks())
        existing = sorted(set(existing))
        if existing:
            rows = np.array(existing, dtype=int) - 1
            R_sub = R_LC[rows]
            marg_un, Z_combo = _single_hyp_exact(R_sub, enforce_meas=())
        else:
            rows = np.array([], dtype=int)
            marg_un, Z_combo = np.zeros((0, mp1)), 1.0

        Z += prior_prod * Z_combo
        # track marginals (existing rows get assoc mass, others get nonexistence)
        if rows.size:
            marginals[rows, :mp1] += marg_un * prior_prod
        nonex = np.setdiff1d(np.arange(n), rows, assume_unique=False)
        marginals[nonex, mp1] += prior_prod * Z_combo  # nonexistence column

        for c, h_idx in enumerate(combo):
            theta_acc[c][h_idx] += prior_prod * Z_combo

    s = marginals.sum(axis=1, keepdims=True)
    marginals = np.divide(marginals, s, out=np.zeros_like(marginals), where=s > 0)
    theta_post = {}
    for c, p in theta_acc.items():
        tot = p.sum()
        theta_post[c] = p / tot if tot > 0 else p
    return marginals, theta_post, Z
