"""
cluster_links.py
================

Discovery of the *sparse* coupling structure between prior clusters, and the
construction of the **delegating variables** ``d_l`` of thesis Section 7.1.

Background (thesis Ch. 3.6, 5.3, 7.1)
-------------------------------------
Prior clusters are independent *except* through **linking measurements**: a
measurement that is gated by tracks belonging to more than one cluster.  Such a
measurement is the only thing that couples the clusters, so conditioning on how
it is delegated restores independence (Eq. (7.10)-(7.12)).

For each linking measurement ``l`` (global, 1-based column index into the reward
matrix) we introduce a delegating variable ``d_l``.  Its event space partitions
the global assignment alphabet ``B = {0} u T^{c_1} u ... u T^{c_C}`` of the
measurement, where ``T^c`` is the set of tracks in cluster ``c`` and ``0`` is
the false-alarm / new-track outcome.

This module computes, for every group of clusters that merge:

* ``cluster_to_linking_measurements`` : c -> {linking meas owned by c}
* ``linking_measurements_to_clusters`` : l -> {clusters gating l}

which together define each ``d_l``'s event space.  These are exactly the
``LinkingMappings`` of the upstream repo.

This implementation discovers merging purely from the gating structure of the
reward matrix ``R_LC`` (a measurement column with finite rewards in >1 cluster),
which is what the upstream ``find_linking_measurements`` ultimately uses.  An
``assocLocal`` array can optionally be supplied for bit-exact compatibility with
the upstream dataset pipeline, but is not required.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from .prior_hypotheses import Hypotheses, HypothesesList


@dataclass
class LinkingMappings:
    """The delegating-variable structure for one group of merging clusters."""

    cluster_to_linking_measurements: Dict[int, Set[int]]
    linking_measurements_to_clusters: Dict[int, Set[int]]

    def all_linking_measurements_idxs(self) -> List[int]:
        # Deterministic ordering -> reproducible enumeration columns.
        return sorted(self.linking_measurements_to_clusters)

    def all_cluster_idxs(self) -> Set[int]:
        return set(self.cluster_to_linking_measurements)


def cartesian_product(*arrays: np.ndarray) -> np.ndarray:
    """Cartesian product of 1-D integer arrays -> ``(prod_len, n_arrays)``.

    Mirrors ``cluster_bayes_tree.cartesian_product`` from the repo.
    """
    la = len(arrays)
    dtype = np.result_type(*[a.dtype for a in arrays]) if arrays else np.int64
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


class ClusterLinks:
    """Finds merging clusters and their linking measurements from gating.

    Parameters
    ----------
    R_LC : (n, m+1) reward matrix in log space; column 0 is misdetection,
           columns ``1..m`` are detection log-rewards (``-inf`` if not gated).
    prior_hypotheses_per_cluster : list of :class:`Hypotheses`, one per cluster.
    assocLocal : optional, kept for upstream compatibility (unused here).
    """

    def __init__(
        self,
        R_LC: np.ndarray,
        prior_hypotheses_per_cluster: Sequence[Hypotheses],
        assocLocal: Optional[np.ndarray] = None,
    ):
        self.R_LC = R_LC
        self.prior_hypotheses_per_cluster = prior_hypotheses_per_cluster
        self.all_clusters: Set[int] = set(range(len(prior_hypotheses_per_cluster)))

        self._tracks_per_cluster = {
            c: np.sort(np.fromiter(ph.tracks(), dtype=int))
            for c, ph in enumerate(prior_hypotheses_per_cluster)
        }
        self._lm2c = self._find_linking_measurements()
        self._c2lm = self._invert(self._lm2c)
        self.merging_clusters = self._group_merging_clusters(self._lm2c)

    # -- gating --------------------------------------------------------------
    def _gated_measurements(self, track_ids_1based: np.ndarray) -> np.ndarray:
        """Boolean mask over measurements (1..m) gated by the given tracks."""
        rows = track_ids_1based - 1
        Rd = self.R_LC[rows, 1:]
        return np.isfinite(Rd).any(axis=0)

    def _find_linking_measurements(self) -> Dict[int, Set[int]]:
        """Map each linking measurement -> set of clusters that gate it."""
        m = self.R_LC.shape[1] - 1
        # gating[c] = boolean mask over measurements gated by cluster c
        gating = {
            c: self._gated_measurements(tracks)
            for c, tracks in self._tracks_per_cluster.items()
        }
        lm2c: Dict[int, Set[int]] = {}
        for j in range(m):
            clusters = {c for c, g in gating.items() if g[j]}
            if len(clusters) > 1:  # gated by >1 cluster -> linking measurement
                lm2c[j + 1] = clusters  # 1-based measurement index
        return lm2c

    @staticmethod
    def _invert(lm2c: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
        c2lm: Dict[int, Set[int]] = defaultdict(set)
        for lm, clusters in lm2c.items():
            for c in clusters:
                c2lm[c].add(lm)
        return c2lm

    @staticmethod
    def _group_merging_clusters(lm2c: Dict[int, Set[int]]) -> List[Set[int]]:
        """Union-find over clusters connected by shared linking measurements."""
        parent: Dict[int, int] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        for clusters in lm2c.values():
            cs = list(clusters)
            for c in cs[1:]:
                union(cs[0], c)

        groups: Dict[int, Set[int]] = defaultdict(set)
        for c in list(parent):
            groups[find(c)].add(c)
        return [g for g in groups.values() if len(g) > 1]

    # -- public API (mirrors upstream) --------------------------------------
    def clusters_that_merge(self) -> Set[int]:
        return set().union(*self.merging_clusters) if self.merging_clusters else set()

    def unmerging_clusters(self) -> Set[int]:
        return self.all_clusters - self.clusters_that_merge()

    def linking_mappings_per_merging_clusters(self) -> List[LinkingMappings]:
        """One :class:`LinkingMappings` per independent group of merging clusters."""
        out: List[LinkingMappings] = []
        for clusters in self.merging_clusters:
            c2lm = {c: set(self._c2lm[c]) for c in clusters}
            lm2c = self._invert(c2lm)
            out.append(
                LinkingMappings(
                    cluster_to_linking_measurements=c2lm,
                    linking_measurements_to_clusters=lm2c,
                )
            )
        return out

    def delegating_variable_event_space(self, l: int) -> List[frozenset]:
        """Disjoint event space of ``d_l`` (thesis Eq. (7.5)).

        ``{{0}} u {T^c : c gates l}``  -- the false-alarm event and one block of
        track ids per gating cluster.
        """
        clusters = self._lm2c[l]
        events: List[frozenset] = [frozenset({0})]
        for c in sorted(clusters):
            events.append(frozenset(int(t) for t in self._tracks_per_cluster[c]))
        return events
