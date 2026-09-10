"""Topology of the association graphs the marginals solvers message-pass on.

Every method in this repo runs on one of two graphs, and the cyclomatic number
``mu = E - V + C`` (circuit rank / first Betti number,
https://en.wikipedia.org/wiki/Circuit_rank) of that graph is the natural
explanatory variable for its error: ``mu = 0`` iff the graph is a forest, for
which LBP is exact, and larger ``mu`` means more independent cycles.

The two graphs, for a cluster whose track set is ``T`` (1-indexed track ids; row
``t - 1`` of ``R_LC``) with feasible edges
``E = {(t, j) : isfinite(R_LC[t-1, j]), 1 <= j <= m}`` and gated measurements
``M = {j : (t, j) in E}``:

**Multihypothesis** -- what ``lbp_single_cluster`` / ``lbp_multicluster``
(``src/lbp.cpp``) run on. Nodes ``T | M | {theta}``, edges ``E`` plus
``(t, theta)`` for every ``t`` in ``T``: the hypothesis factor touches every
track of the cluster, which is what ``t2h`` / ``t2h_not`` and the ``sigma``
update encode. Theta joins every component, so ``C = 1`` and the value collapses
to ``|E| - |M|``.

**Hypothesis-conditioned** -- what exact EHM2 and the Williams LBP run on. Given
one prior hypothesis, ``MulticlusterExactEHM2`` slices rows only,
``R_LC[np.array(h.tracks()) - 1, :]`` (``marginals_computers.py``), and
``LBPMarginalsByTotalProbBethe`` does the same before calling ``lbp_marginal``.
So the graph is bipartite over the tracks existing in that hypothesis and their
gated measurements. One number per hypothesis.

Isolated vertices never change ``mu`` (they add one to both ``V`` and ``C``), so
they are dropped throughout -- the same convention as the scan-level
``cyclomatic_number`` this module took over from
``ravens_parser_parallell_multicluster.py``.
"""

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .stats_logger import ClusterGraphStats, MulticlusterGraphStats


def _cyclomatic_from_edges(u: np.ndarray, v: np.ndarray, n_nodes: int) -> int:
    """``mu = E - V + C`` for the undirected graph on ``n_nodes`` labelled nodes
    with edges ``zip(u, v)``. Isolated nodes are excluded from both ``V`` and ``C``.

    Edges must be unique: ``coo_matrix`` sums duplicates into one entry, while
    ``E`` is taken as ``u.size``, so a repeated pair would be counted twice.
    """
    n_edges = int(u.size)
    if n_edges == 0:
        return 0

    adjacency = coo_matrix((np.ones(n_edges, dtype=np.int8), (u, v)),
                           shape=(n_nodes, n_nodes))
    n_components, _ = connected_components(adjacency, directed=False)

    touched = np.zeros(n_nodes, dtype=bool)
    touched[u] = True
    touched[v] = True
    n_vertices = int(touched.sum())

    # connected_components counts every isolated node as its own component.
    n_components -= n_nodes - n_vertices

    return int(n_edges - n_vertices + n_components)


def _num_components(adjacency: np.ndarray) -> int:
    """Number of connected components of a small dense symmetric boolean adjacency.

    Called once per hypothesis, on a matrix whose side is the number of gated
    measurements (single digits for these scans), so a closure over numpy rows
    beats paying scipy's sparse-graph setup cost per call.
    """
    n_nodes = adjacency.shape[0]
    seen = np.zeros(n_nodes, dtype=bool)
    n_components = 0
    for node in range(n_nodes):
        if seen[node]:
            continue
        n_components += 1
        component = np.zeros(n_nodes, dtype=bool)
        frontier = adjacency[node]
        while True:
            new = frontier & ~component
            if not new.any():
                break
            component |= new
            frontier = adjacency[new].any(axis=0)
        seen |= component

    return n_components


def cyclomatic_number_from_gate(gate: np.ndarray) -> int:
    """``mu`` of the bipartite track<->measurement graph described by ``gate``.

    ``gate`` is the ``(num_tracks, m)`` boolean feasibility mask, i.e. the
    measurement columns of ``R_LC`` with the misdetection column already dropped.

    Components are counted on the measurement projection -- measurements ``j`` and
    ``j'`` adjacent iff some track gates both -- which has exactly the components
    of the bipartite graph, since every component left after dropping isolated
    vertices contains at least one measurement. The projection is at most ``m``
    nodes wide where the bipartite graph is ``num_tracks + m``.
    """
    gate = np.asarray(gate, dtype=bool)
    if gate.size == 0:
        return 0

    n_edges = int(gate.sum())
    if n_edges == 0:
        return 0

    meas_used = gate.any(axis=0)
    n_vertices = int(gate.any(axis=1).sum()) + int(meas_used.sum())

    gate_used = gate[:, meas_used].astype(np.int8)
    co_gated = (gate_used.T @ gate_used) > 0

    return int(n_edges - n_vertices + _num_components(co_gated))


def gate_mask(R_LC: np.ndarray, rows: Optional[np.ndarray] = None) -> np.ndarray:
    """Feasible track->measurement mask of ``R_LC``, optionally row-sliced.

    Column 0 of ``R_LC`` is misdetection and is never an edge. Feasibility is
    ``np.isfinite``, matching ``ClusterLinks.gated_measurements``; EHM2 instead
    tests ``exp(R) > 0``, which agrees on everything these .mat files contain
    (the only non-finite entries are the ``-inf`` ungated ones).
    """
    R_LC = np.asarray(R_LC)
    if rows is not None:
        rows = np.asarray(rows, dtype=int)
        if rows.size == 0:
            return np.zeros((0, R_LC.shape[1] - 1), dtype=bool)
        R_LC = R_LC[rows]

    return np.isfinite(R_LC[:, 1:])


def cyclomatic_number(R_LC: np.ndarray) -> int:
    """Scan-level ``mu`` of the plain bipartite track<->measurement graph.

    Kept for the CSV-based scripts (``accuracy_metrics_multicluster.py``,
    ``chapter10_plots.py``), which reach it through
    ``ravens_parser_parallell_multicluster.cyclomatic_number``.
    """
    return cyclomatic_number_from_gate(gate_mask(R_LC))


def cluster_graph_stats(R_LC: np.ndarray, prior_hypotheses,
                        member_prior_clusters: Tuple[int, ...] = ()) -> ClusterGraphStats:
    """Both cyclomatic numbers for one cluster's ``py_dfg_da`` ``Hypotheses``.

    ``prior_hypotheses`` may be a prior cluster or a merged (posterior) one --
    the merged case is what ``MulticlusterExactEHM2`` conditions on, and
    ``member_prior_clusters`` records which prior clusters it was built from.
    """
    R_LC = np.asarray(R_LC)

    # t_idxs() is the C++ helper: sorted union of track ids over all hypotheses,
    # minus one, i.e. rows of the global R_LC.
    t_idxs = np.asarray(prior_hypotheses.t_idxs(), dtype=int)
    gate = gate_mask(R_LC, t_idxs)

    n_tracks = int(t_idxs.size)
    n_edges = int(gate.sum())
    n_gated_measurements = int(gate.any(axis=0).sum()) if gate.size else 0

    mu_bipartite = cyclomatic_number_from_gate(gate)

    # Multihypothesis graph: the same edges plus one theta node wired to every
    # track of the cluster. Node layout [tracks | measurements | theta].
    num_measurements = R_LC.shape[1] - 1
    track_idx, meas_idx = np.nonzero(gate)
    theta = n_tracks + num_measurements
    u = np.concatenate((track_idx, np.arange(n_tracks)))
    v = np.concatenate((n_tracks + meas_idx, np.full(n_tracks, theta)))
    mu_multihypothesis = _cyclomatic_from_edges(u, v, theta + 1)
    # Theta joins every component, so this always reduces to |E| - |M|.
    assert n_tracks == 0 or mu_multihypothesis == n_edges - n_gated_measurements

    # Conditioned graphs are row subsets of the cluster's own gate mask, so index
    # into it rather than slicing R_LC again per hypothesis. A hypothesis's tracks
    # are by construction a subset of the cluster's, so searchsorted is exact.
    n_hypotheses = len(prior_hypotheses)
    mu_conditioned = np.zeros(n_hypotheses, dtype=np.int32)
    weights = np.zeros(n_hypotheses, dtype=float)
    for k, hypothesis in enumerate(prior_hypotheses):
        weights[k] = hypothesis.probability()
        rows = np.sort(np.asarray(hypothesis.tracks(), dtype=int)) - 1
        if rows.size == 0:
            continue
        positions = np.searchsorted(t_idxs, rows)
        assert np.array_equal(t_idxs[positions], rows)
        mu_conditioned[k] = cyclomatic_number_from_gate(gate[positions])

    nan = float("nan")
    weight_sum = float(weights.sum())
    return ClusterGraphStats(
        track_ids=(t_idxs + 1).astype(np.int32),
        member_prior_clusters=tuple(int(c) for c in member_prior_clusters),
        n_tracks=n_tracks,
        n_gated_measurements=n_gated_measurements,
        n_edges=n_edges,
        n_hypotheses=n_hypotheses,
        mu_multihypothesis=mu_multihypothesis,
        mu_bipartite=mu_bipartite,
        mu_conditioned=mu_conditioned,
        mu_conditioned_max=int(mu_conditioned.max()) if n_hypotheses else 0,
        mu_conditioned_mean=float(mu_conditioned.mean()) if n_hypotheses else nan,
        mu_conditioned_prior_mean=(float((mu_conditioned * weights).sum() / weight_sum)
                                   if weight_sum > 0.0 else nan),
        tree_hypothesis_fraction=(float((mu_conditioned == 0).mean())
                                  if n_hypotheses else nan),
    )


def scan_multihypothesis_cyclomatic(R_LC: np.ndarray, prior_hypotheses_per_cluster) -> int:
    """``mu`` of the whole factor graph ``lbp_multicluster`` runs on.

    Every cluster contributes its feasible edges plus one theta node wired to its
    own tracks; measurement nodes are shared across clusters, so this is strictly
    more than the sum of the per-cluster values whenever clusters gate a common
    measurement.
    """
    R_LC = np.asarray(R_LC)
    num_tracks, num_measurements = R_LC.shape[0], R_LC.shape[1] - 1

    us: List[np.ndarray] = []
    vs: List[np.ndarray] = []
    for c, prior_hypotheses in enumerate(prior_hypotheses_per_cluster):
        t_idxs = np.asarray(prior_hypotheses.t_idxs(), dtype=int)
        if t_idxs.size == 0:
            continue
        gate = gate_mask(R_LC, t_idxs)
        local_track, meas_idx = np.nonzero(gate)
        theta = num_tracks + num_measurements + c
        us.append(t_idxs[local_track])
        vs.append(num_tracks + meas_idx)
        us.append(t_idxs)
        vs.append(np.full(t_idxs.size, theta))

    if not us:
        return 0

    u = np.concatenate(us)
    v = np.concatenate(vs)
    # Clusters partition the tracks, so pairs are already unique; dedupe anyway
    # so a malformed clustering cannot silently inflate E.
    edges = np.unique(np.stack((u, v)), axis=1)
    n_nodes = num_tracks + num_measurements + len(prior_hypotheses_per_cluster)

    return _cyclomatic_from_edges(edges[0], edges[1], n_nodes)


def multicluster_graph_stats(R_LC: np.ndarray,
                             prior_hypotheses_per_cluster,
                             merged_clusters=None,
                             merged_members: Optional[Sequence[Iterable[int]]] = None,
                             merged_skipped_reason: str = "") -> MulticlusterGraphStats:
    """Graph topology of one scan, over both cluster partitions.

    ``prior_hypotheses_per_cluster`` is the unmerged list ``lbp_multicluster`` is
    handed. ``merged_clusters`` is the posterior list
    ``ClusterHypothesesPosterior.prior_hypotheses_per_cluster_posterior``, which
    is what ``MulticlusterExactEHM2`` conditions on hypothesis-for-hypothesis;
    pass ``None`` (with a ``merged_skipped_reason``) when it is unavailable or
    too large to build.
    """
    per_prior_cluster = [cluster_graph_stats(R_LC, prior_hypotheses)
                         for prior_hypotheses in prior_hypotheses_per_cluster]

    per_merged_cluster = None
    if merged_clusters is not None:
        if merged_members is None:
            merged_members = [()] * len(merged_clusters)
        per_merged_cluster = [
            cluster_graph_stats(R_LC, prior_hypotheses, tuple(members))
            for prior_hypotheses, members in zip(merged_clusters, merged_members)
        ]

    return MulticlusterGraphStats(
        per_prior_cluster=per_prior_cluster,
        per_merged_cluster=per_merged_cluster,
        mu_scan_multihypothesis=scan_multihypothesis_cyclomatic(
            R_LC, prior_hypotheses_per_cluster),
        mu_scan_bipartite=cyclomatic_number(R_LC),
        merged_skipped_reason=merged_skipped_reason,
    )


def merged_cluster_members(assocLocal: np.ndarray) -> List[Tuple[int, ...]]:
    """Prior clusters making up each merged (posterior) cluster, in posterior order.

    ``assocLocal[0]`` is the 1-indexed master cluster of each prior cluster and
    ``assocLocal[1]`` flags the masters; the posterior index of a master is its
    rank among the masters. Same convention as
    ``ClusterHypothesesPosterior.__init__``, which shifts row 0 to 0-based before
    using it -- note the standalone ``merge_clusters()`` copies in the repo omit
    that shift.
    """
    assocLocal = np.asarray(assocLocal)
    masters = assocLocal[0].astype(int) - 1
    is_master = assocLocal[1].astype(int)
    posterior_idx = np.cumsum(is_master) - 1

    members: List[List[int]] = [[] for _ in range(int(is_master.sum()))]
    for cluster, master in enumerate(masters):
        members[posterior_idx[master]].append(cluster)

    return [tuple(m) for m in members]
