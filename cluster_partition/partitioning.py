"""
partitioning.py
===============

The cluster-partitioning / cluster-conditioning marginalization of thesis
Chapter 7, with the **alternative (overlapping) event space** of Section 7.5 and
the **inclusion-exclusion** treatment of Section 7.5.1.

Derivation recap
----------------
A linking measurement ``l`` couples a group of clusters ``C_l`` (those whose
tracks gate it).  Introduce the delegating variable ``d_l`` and condition on it
to restore cluster independence (Eq. (7.10)-(7.12)):

    Pr{a^t | Z} ∝ Σ_d Pr{a^t | Z, d} · p(Z | d)            (uniform Pr{d})
    p(Z | d)   = Π_c p(Z^c | d)                            (independence)

Two event spaces for ``d_l``:

* **Disjoint** (Eq. (7.5)):   d_l ∈ {{0}} ∪ {T^c : c ∈ C_l}.
  Delegating to ``c`` *forces* the measurement to be detected by a track in
  ``c`` (false alarm disallowed there).  Exact, but enforcing a detection drives
  the SNR of that measurement to infinity, which destabilizes LBP and voids its
  convergence guarantees (Sec. 7.5, 7.4).

* **Overlapping** (Eq. (7.29)):  d_l ∈ {{0}} ∪ {{0} ∪ T^c : c ∈ C_l}.
  Delegating to ``c`` lets the measurement be *either* a detection in ``c`` *or*
  a misdetection — so no infinite SNR.  But now the misdetection (``{0}``)
  outcome lies in every event, so naive summation over-counts it.  Correct
  summation needs the inclusion-exclusion principle.

Closed-form inclusion-exclusion
-------------------------------
Because tracks are exclusive to one cluster, *any* intersection of two or more
overlapping events collapses to ``{0}`` (Sec. 7.5).  The conditioned quantity
``P`` (a sum over allowed global assignments) is **additive** over disjoint
allowed-sets, so inclusion-exclusion applies exactly and, for a single linking
measurement gated by ``K_l = |C_l|`` clusters, telescopes to

    P(full) = Σ_{c∈C_l} P({0}∪T^c) + (1 - K_l)·P({0}).

Equivalently, over the enumeration tuple ``i_l ∈ {none} ∪ C_l`` used by the
codebase (``[-1, *C_l]``), every term carries a **signed weight**

    ω_l(c)    = +1            (c ∈ C_l)
    ω_l(none) = 1 - K_l

and, by multilinearity across measurements, the exact union is

    Exact = Σ_d ( Π_l ω_l(i_l) ) · Q(d),

with ``Q(d)`` the first-stage conditioned term.  Keeping only the first stage
(``ω ≡ 1``) is the thesis's published approximation; by the Bonferroni
inequalities (Sec. 7.5.1) it uses an *odd* number of stages and therefore
*overestimates* the Bethe constant / unnormalized mass.

Modes implemented
-----------------
``"disjoint_exact"``       : Eq. (7.5), enforce detection.  Exact.  (= upstream
                             ``MulticlusterEfficientMarginals``.)
``"overlap_firststage"``   : Eq. (7.29), first stage only.  Biased (overcounts).
                             (= upstream ``MulticlusterEfficientMarginalsLBP``.)
``"overlap_ie"``           : Eq. (7.29) + full inclusion-exclusion via the signed
                             weights above.  Exact with an exact inner solver;
                             de-biases an LBP inner solver.  (The thesis's
                             "future work".)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .cluster_links import ClusterLinks, LinkingMappings, cartesian_product
from .prior_hypotheses import Hypotheses
from .solvers import ExactEnumerationSolver

MODES = ("disjoint_exact", "overlap_firststage", "overlap_ie")
NONE_EVENT = -1  # the "{0}" / false-alarm-everywhere outcome of a delegating var


class ConditionedCluster:
    """A single cluster conditioned on a delegating-variable tuple.

    Owns the cluster's reward sub-matrix and prior hypotheses, plus the
    bookkeeping that maps a supercluster-wide measurement-assignment tuple to
    *this* cluster's view (which of its linking measurements are delegated to it).
    """

    def __init__(
        self,
        cluster_idx: int,
        R_cluster: np.ndarray,
        prior_hypotheses: Hypotheses,
        linking_map_mat: np.ndarray,  # rows: (global_meas_idx, supercluster_col)
        solver,
        mode: str,
    ):
        self.cluster_idx = cluster_idx
        self.R_cluster = R_cluster
        self.prior_hypotheses = prior_hypotheses
        # Capture GLOBAL 0-based rows BEFORE reindexing relabels tracks to local
        # 1..n_c (upstream captures t_idxs first for exactly this reason).
        self.t_idxs = prior_hypotheses.t_idxs()
        self.prior_hypotheses.reindex_tracks()
        self.actual_meas = linking_map_mat[:, 0]   # global 1-based meas indices
        self.reindex_meas = linking_map_mat[:, 1]  # supercluster tuple columns
        self.solver = solver
        self.mode = mode
        self.cache: Dict[Tuple[bool, ...], Tuple[np.ndarray, np.ndarray, float]] = {}

    def _owned_mask(self, tuple_assignment: np.ndarray) -> np.ndarray:
        """Boolean over this cluster's linking measurements: delegated to me?"""
        mine = tuple_assignment[self.reindex_meas]
        return mine == self.cluster_idx

    def _conditioned_reward(self, owned_mask: np.ndarray) -> Tuple[np.ndarray, tuple]:
        R = self.R_cluster.copy()
        # Linking measurements NOT delegated to this cluster cannot be detected here.
        not_owned = self.actual_meas[~owned_mask]
        R[:, not_owned] = -np.inf
        enforce = ()
        if self.mode == "disjoint_exact":
            # Owned measurements MUST be detected (disjoint event space).
            enforce = tuple(int(j) for j in self.actual_meas[owned_mask])
        return R, enforce

    def conditioned_marginals(self, tuple_assignment: np.ndarray):
        owned = self._owned_mask(tuple_assignment)
        key = tuple(bool(b) for b in owned)
        if key in self.cache:                       # dynamic programming (Sec. 7.3)
            return self.cache[key]
        R, enforce = self._conditioned_reward(owned)
        out = self.solver(R, self.prior_hypotheses, enforce_meas=enforce, reindex=False)
        self.cache[key] = out
        return out


class ConditionalSuperclusterMarginals:
    """Marginalizes one group of merging clusters by conditioning on all ``d_l``."""

    def __init__(
        self,
        R_LC: np.ndarray,
        prior_hypotheses_per_cluster: Sequence[Hypotheses],
        linking_mappings: LinkingMappings,
        solver,
        mode: str,
    ):
        assert mode in MODES, f"unknown mode {mode!r}"
        self.R_LC = R_LC
        self.prior_hypotheses_per_cluster = prior_hypotheses_per_cluster
        self.linking_mappings = linking_mappings
        self.mode = mode
        self.solver = solver

        self.lm2col = {
            lm: col for col, lm in enumerate(linking_mappings.all_linking_measurements_idxs())
        }
        self.t_idxs = self.supercluster_t_idxs()

        self.conditioned_clusters: List[ConditionedCluster] = []
        for cluster, lms in linking_mappings.cluster_to_linking_measurements.items():
            ph = prior_hypotheses_per_cluster[cluster]
            t_idxs = ph.t_idxs()
            R_cluster = R_LC[t_idxs]
            link_mat = np.array([(lm, self.lm2col[lm]) for lm in sorted(lms)], dtype=int)
            self.conditioned_clusters.append(
                ConditionedCluster(cluster, R_cluster, ph, link_mat, solver, mode)
            )

    def supercluster_t_idxs(self) -> np.ndarray:
        idxs = [
            ph.t_idxs()
            for c, ph in enumerate(self.prior_hypotheses_per_cluster)
            if c in self.linking_mappings.all_cluster_idxs()
        ]
        return np.sort(np.hstack(idxs))

    def _delegating_event_columns(self):
        """For each tuple column, the list of outcomes ``[-1, *gating clusters]``."""
        cols = [None] * len(self.lm2col)
        for lm, col in self.lm2col.items():
            clusters = sorted(self.linking_mappings.linking_measurements_to_clusters[lm])
            cols[col] = np.array([NONE_EVENT, *clusters], dtype=int)
        return cols

    def _ie_weights(self):
        """Per tuple-column signed weight tables ω_l(·) keyed by outcome value."""
        weights = []
        for lm in self.linking_mappings.all_linking_measurements_idxs():
            clusters = self.linking_mappings.linking_measurements_to_clusters[lm]
            K = len(clusters)
            w = {NONE_EVENT: (1 - K)}
            for c in clusters:
                w[c] = 1
            weights.append(w)
        return weights

    def enumerate_assignments(self) -> np.ndarray:
        return cartesian_product(*self._delegating_event_columns())

    def compute_marginals_likelihood(self):
        n, mp1 = self.R_LC.shape
        marginals = np.zeros((n, mp1 + 1))
        theta_post = defaultdict(lambda: 0.0)
        likelihood = 0.0

        assignments = self.enumerate_assignments()
        ie = self.mode == "overlap_ie"
        weight_tables = self._ie_weights() if ie else None

        marg_term = np.empty_like(marginals)
        for d in assignments:
            # Signed inclusion-exclusion weight for this tuple.
            W = 1
            if ie:
                for col, val in enumerate(d):
                    W *= weight_tables[col][int(val)]
                if W == 0:
                    continue

            # print(f"Weight: {W}")

            joint_lik = 1.0
            theta_terms = {}
            ok = True
            for cc in self.conditioned_clusters:
                marg_c, theta_c, lik_c = cc.conditioned_marginals(d)
                if lik_c <= 0.0:
                    ok = False
                    break
                marg_term[cc.t_idxs] = marg_c
                joint_lik *= lik_c
                theta_terms[cc.cluster_idx] = theta_c
            if not ok:
                continue

            signed = W * joint_lik
            marginals[self.t_idxs] += marg_term[self.t_idxs] * signed
            likelihood += signed
            for c, p in theta_terms.items():
                theta_post[c] = theta_post[c] + p * signed

        marg = marginals[self.t_idxs]
        s = marg.sum(axis=1, keepdims=True)
        marg = np.divide(marg, s, out=np.zeros_like(marg), where=np.abs(s) > 0)
        out_marg = np.empty((n, mp1 + 1))
        out_marg[self.t_idxs] = marg

        theta_out = {}
        for c, p in theta_post.items():
            tot = np.sum(p)
            theta_out[c] = p / tot if tot != 0 else p
        return out_marg, theta_out, likelihood


class MulticlusterPartitionedMarginals:
    """Top-level driver: superclusters (merging) + untouched clusters.

    Mirrors the upstream ``MulticlusterEfficientMarginals`` /
    ``MulticlusterEfficientMarginalsLBP`` classes, parameterized by event-space
    mode and inner solver.
    """

    def __init__(
        self,
        R_LC: np.ndarray,
        prior_hypotheses_per_cluster: Sequence[Hypotheses],
        solver=None,
        mode: str = "overlap_ie",
        assocLocal=None,
    ):
        assert mode in MODES, f"unknown mode {mode!r}"
        self.R_LC = R_LC
        self.prior_hypotheses_per_cluster = prior_hypotheses_per_cluster
        self.mode = mode
        self.solver = solver or ExactEnumerationSolver()
        self.cluster_links = ClusterLinks(R_LC, prior_hypotheses_per_cluster, assocLocal)
        self.superclusters = [
            ConditionalSuperclusterMarginals(
                R_LC, prior_hypotheses_per_cluster, lm, self.solver, mode
            )
            for lm in self.cluster_links.linking_mappings_per_merging_clusters()
        ]

    def compute_marginals_likelihood(self):
        n, mp1 = self.R_LC.shape
        marginals = np.empty((n, mp1 + 1))
        theta_posteriors: Dict[int, np.ndarray] = {}
        likelihood = 1.0

        for sc in self.superclusters:
            m_sc, th_sc, lik_sc = sc.compute_marginals_likelihood()
            marginals[sc.t_idxs] = m_sc[sc.t_idxs]
            likelihood *= lik_sc
            theta_posteriors.update(th_sc)

        for c in self.cluster_links.unmerging_clusters():
            ph = self.prior_hypotheses_per_cluster[c]
            t_idxs = ph.t_idxs()
            R_cluster = self.R_LC[t_idxs]
            m_c, th_c, lik_c = self.solver(R_cluster, ph, enforce_meas=(), reindex=True)
            marginals[t_idxs] = m_c
            likelihood *= lik_c
            theta_posteriors[c] = th_c

        s = marginals.sum(axis=1, keepdims=True)
        marginals = np.divide(marginals, s, out=np.zeros_like(marginals), where=np.abs(s) > 0)
        return marginals, theta_posteriors, likelihood
