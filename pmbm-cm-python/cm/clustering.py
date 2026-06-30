"""Clustering preprocessing: assign measurements to clusters and break weak
inter-cluster edges.

Port of ``clusteringPreprocess.m``. The MATLAB uses graph-toolbox functions
(``graph``, ``centrality 'betweenness'``, ``rmedge``, ``conncomp``,
``groupcounts``); here we use ``networkx``.

Stage-1 note: at k=1 there are no clusters, so only the trivial ``else`` branch
runs (returns the gain matrix unchanged with empty ``masters``/``assocLocal``).
The heavy graph branch (including the supercluster size/cardinality balancing
loop) is ported for Stage 2 but is not bit-exact against MATLAB (betweenness /
connected-component labelling order may differ).
"""

import numpy as np

from .clouds import (
    cluster2tracks,
    pick_ind_c,
    tcloud_to_beg,
    tcloud_to_end,
    track2cluster,
    union_sorted,
)
from .mathutils import find_colmajor, v2m


# Supercluster balancing heuristics (script_pmbm91 lines ~74-289). MATLAB magic
# numbers, kept verbatim.
MAX_CLUSTERS_PER_SUPERCLUSTER = 5      # GC > 5 test
MAX_CARD_HEURISTIC_MARGIN = 5          # maxCardHeuri = max(maxCard) + 5
MAX_COUNT_HEURISTIC_MARGIN = 100       # maxCountHeuri = max(maxCount) + 100
EDGE_GAIN_HEURISTIC_WEIGHT = 0.75      # worstBreakEdgeGains * 3/4
MAX_BALANCING_ITERS = 1200             # nycaCount < 1200 guard


def _conncomp_labels(graph, n_nodes):
    """Return a 1-based component label per node (1..n_nodes), mimicking the
    way MATLAB ``conncomp`` + the cluster-assoc loop derive a representative."""
    import networkx as nx

    labels = np.zeros(n_nodes + 1, dtype=int)  # 1-based node ids
    comp_id = 0
    for comp in nx.connected_components(graph):
        comp_id += 1
        for node in comp:
            labels[node] = comp_id
    return labels


def _cluster_assoc_from_graph(H, nC, n_nodes):
    """Component representative per cluster node, mirroring MATLAB ``conncomp`` +
    ``clusterAssoc(q) = q(1)``: each of the first ``nC`` nodes is mapped to the
    lowest node id in its connected component. Returns a length-``nC`` 1-based
    array."""
    labels = _conncomp_labels(H, n_nodes)
    comp_first = {}
    assoc_full = np.zeros(n_nodes + 1, dtype=int)
    for node in range(1, n_nodes + 1):
        lab = labels[node]
        if lab not in comp_first:
            comp_first[lab] = node
        assoc_full[node] = comp_first[lab]
    return assoc_full[1:nC + 1]


def clustering_preprocess(hypos, hypos_card, clusters, clusters_card,
                          gain_mat_full, mea_hist_col, m, pre_cluster_threshold):
    """Port of ``clusteringPreprocess.m``.

    Returns (assoc_local (2, nC), gain_mat_post_c, masters (1-based cluster ids)).
    """
    gain_mat_full = np.array(gain_mat_full, dtype=float, copy=True)
    clusters = np.asarray(clusters, dtype=int).ravel()
    clusters_card = np.asarray(clusters_card, dtype=int).ravel()
    hypos = np.asarray(hypos, dtype=int).ravel()
    hypos_card = np.asarray(hypos_card, dtype=int).ravel()

    nT = mea_hist_col.shape[1]
    l_mat_full = gain_mat_full[:nT, :m] if nT > 0 else np.zeros((0, m))
    nC = clusters_card.size

    has_finite = m > 0 and np.any(~np.isinf(l_mat_full))

    if has_finite:
        gain_mat_post_c, cluster_assoc = _clustering_heavy(
            hypos, hypos_card, clusters, clusters_card, gain_mat_full,
            l_mat_full, nT, nC, m, pre_cluster_threshold)
    else:
        # m == 0, or measurements present but none validated.
        gain_mat_post_c = gain_mat_full
        cluster_assoc = np.arange(1, nC + 1)

    uniq, ia = np.unique(cluster_assoc, return_index=True)
    masters = cluster_assoc[np.sort(ia)] if cluster_assoc.size else np.zeros(0, dtype=int)

    assoc_local = np.zeros((2, cluster_assoc.size))
    if cluster_assoc.size:
        assoc_local[0, :] = cluster_assoc
        assoc_local[1, masters - 1] = 1
    return assoc_local, gain_mat_post_c, masters


def _clustering_heavy(hypos, hypos_card, clusters, clusters_card, gain_mat_full,
                      l_mat_full, nT, nC, m, pre_cluster_threshold):
    """Heavy graph branch of clusteringPreprocess.

    Builds the cluster-measurement graph, breaks weak connecting edges via
    betweenness, then runs the iterative supercluster size/cardinality balancing
    loop (MATLAB lines 74-289) that keeps superclusters small. Not bit-exact vs
    MATLAB (betweenness / connected-component ordering may differ)."""
    import networkx as nx

    begsC = tcloud_to_beg(clusters_card)
    endsC = tcloud_to_end(clusters_card)

    l_mat_bool = ~np.isinf(l_mat_full)
    ix = find_colmajor(l_mat_bool)            # 1-based column-major
    subs = v2m(ix, nT)
    track_numbers = subs[0, :]                # 1-based track ids
    mea_numbers = nC + subs[1, :]             # measurement node ids

    cluster_numbers, _ = track2cluster(track_numbers, hypos, hypos_card,
                                       clusters, clusters_card)
    cluster_numbers = cluster_numbers.astype(int)

    # Edge values = the gating gains, sorted descending; keep unique cluster-mea edges.
    tempvals = l_mat_full.flatten(order="F")[ix - 1]
    order = np.argsort(-tempvals, kind="stable")
    cm = np.vstack([cluster_numbers[order], mea_numbers[order]])
    _, uniq_idx = np.unique(cm.T, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    edge_arr = cm[:, uniq_idx]
    vals = tempvals[order][uniq_idx]

    mis_det_part = np.diag(gain_mat_full[:nT, m:])
    trackwise_threshold = mis_det_part - pre_cluster_threshold

    G = nx.Graph()
    G.add_nodes_from(range(1, nC + m + 1))
    for e in range(edge_arr.shape[1]):
        G.add_edge(int(edge_arr[0, e]), int(edge_arr[1, e]), weight=float(vals[e]))

    bc = nx.betweenness_centrality(G, normalized=False)
    wbc = np.array([bc.get(int(n), 0.0) for n in range(1, nC + m + 1)])
    connecting_edges = wbc[edge_arr[1, :] - 1] > 0
    thresholded = vals < trackwise_threshold[edge_arr[0, :] - 1]
    to_be_broken = thresholded & connecting_edges

    # Keep each measurement attached to at least its best cluster.
    for jj in range(1, m + 1):
        j_edges = (edge_arr[1, :] == (jj + nC)) & (~to_be_broken)
        if not np.any(j_edges):
            relevant = np.nonzero(edge_arr[1, :] == (jj + nC))[0]
            if relevant.size:
                best = relevant[np.argmax(vals[relevant])]
                to_be_broken[best] = False

    H = G.copy()
    for e in np.nonzero(to_be_broken)[0]:
        if H.has_edge(int(edge_arr[0, e]), int(edge_arr[1, e])):
            H.remove_edge(int(edge_arr[0, e]), int(edge_arr[1, e]))

    # --- Supercluster size/cardinality balancing loop (clusteringPreprocess.m
    # lines 74-289) --------------------------------------------------------------
    n_nodes = nC + m

    # Per-cluster max hypothesis cardinality and hypothesis count (lines 74-82).
    max_card_clusters = np.zeros(nC, dtype=float)
    for ii in range(nC):
        seg = hypos_card[begsC[ii] - 1:endsC[ii]]
        max_card_clusters[ii] = seg.max() if seg.size else 0
    max_count_clusters = clusters_card.astype(float)
    max_card_heuri = (max_card_clusters.max() + MAX_CARD_HEURISTIC_MARGIN) if nC else 0.0
    max_count_heuri = (max_count_clusters.max() + MAX_COUNT_HEURISTIC_MARGIN) if nC else 0.0

    def _mea_neighbors(node):
        return set(int(v) - nC for v in H.neighbors(node) if v > nC)

    # Precompute per-cluster-pair break info + worst-edge gains (lines 86-169).
    init_assoc = _cluster_assoc_from_graph(H, nC, n_nodes)
    worst_break_edge_gains = np.full((nC, nC), -np.inf)
    break_edge_stuff = {}
    edges_cand = []
    for ii in range(1, nC + 1):
        for jj in range(ii + 1, nC + 1):
            if init_assoc[ii - 1] != init_assoc[jj - 1]:
                continue
            mea_common = np.array(sorted(_mea_neighbors(ii) & _mea_neighbors(jj)),
                                  dtype=int)
            if mea_common.size == 0:
                continue
            tracks_i = cluster2tracks(ii, hypos, hypos_card, clusters, clusters_card)
            tracks_j = cluster2tracks(jj, hypos, hypos_card, clusters, clusters_card)
            tracks_union, a_match, b_match = union_sorted(tracks_i, tracks_j)
            rew_sub = gain_mat_full[np.ix_(tracks_union - 1, mea_common - 1)].copy()
            winning = np.zeros(mea_common.size, dtype=int)
            worst = -np.inf
            for zz in range(mea_common.size):
                gating = np.nonzero(rew_sub[:, zz] > -np.inf)[0] + 1  # 1-based union pos
                tracks_iz = np.intersect1d(a_match, gating)
                tracks_jz = np.intersect1d(b_match, gating)
                max_i = rew_sub[tracks_iz - 1, zz].max() if tracks_iz.size else -np.inf
                max_j = rew_sub[tracks_jz - 1, zz].max() if tracks_jz.size else -np.inf
                if max_i > max_j:
                    winning[zz] = ii
                    rew_sub[tracks_jz - 1, zz] = -np.inf
                    worst = max(worst, max_j)
                else:
                    winning[zz] = jj
                    rew_sub[tracks_iz - 1, zz] = -np.inf
                    worst = max(worst, max_i)
            break_edge_stuff[(ii, jj)] = {
                "mea_common": mea_common, "rew_sub": rew_sub,
                "winning": winning, "tracks_union": tracks_union}
            worst_break_edge_gains[ii - 1, jj - 1] = worst
            edges_cand.append((ii, jj))

    # Iteratively cut the weakest inter-cluster connection until every
    # supercluster is acceptably small (lines 185-286).
    acceptable = (nC == 0)
    count = 1
    while not acceptable and count < MAX_BALANCING_ITERS:
        count += 1
        cluster_assoc = _cluster_assoc_from_graph(H, nC, n_nodes)
        uca, gc = np.unique(cluster_assoc, return_counts=True)
        test_size = not np.any(gc > MAX_CLUSTERS_PER_SUPERCLUSTER)
        card_ok = True
        count_ok = True
        for rep in uca:
            members = np.nonzero(cluster_assoc == rep)[0]
            if members.size > 1:
                if max_card_clusters[members].sum() > max_card_heuri:
                    card_ok = False
                if max_count_clusters[members].sum() > max_count_heuri:
                    count_ok = False
        if test_size and card_ok and count_ok:
            acceptable = True
            continue
        if not edges_cand:
            break

        heuristic = np.empty(len(edges_cand))
        for idx, (ii, jj) in enumerate(edges_cand):
            size_contrib = np.sum(cluster_assoc == cluster_assoc[ii - 1]) \
                / MAX_CLUSTERS_PER_SUPERCLUSTER
            card_contrib = (max_card_clusters[ii - 1] + max_card_clusters[jj - 1]) \
                / max_card_heuri
            count_contrib = (max_count_clusters[ii - 1] + max_count_clusters[jj - 1]) \
                / max_count_heuri
            edge_contrib = worst_break_edge_gains[ii - 1, jj - 1] \
                * EDGE_GAIN_HEURISTIC_WEIGHT
            heuristic[idx] = size_contrib + card_contrib + count_contrib - edge_contrib

        best_idx = int(np.argmax(heuristic))
        ii, jj = edges_cand.pop(best_idx)
        stuff = break_edge_stuff[(ii, jj)]
        # Apply the (loser-masked) reward submatrix into the gain matrix.
        gain_mat_full[np.ix_(stuff["tracks_union"] - 1, stuff["mea_common"] - 1)] = \
            stuff["rew_sub"]
        # Remove the losing cluster's edge to each affected measurement in H.
        for z in range(stuff["mea_common"].size):
            winner = stuff["winning"][z]
            loser = jj if winner == ii else ii
            mnode = nC + int(stuff["mea_common"][z])
            if H.has_edge(loser, mnode):
                H.remove_edge(loser, mnode)

    # Mark every original edge no longer present in H as broken (lines 288-289).
    for e in range(edge_arr.shape[1]):
        if not H.has_edge(int(edge_arr[0, e]), int(edge_arr[1, e])):
            to_be_broken[e] = True

    cluster_assoc = _cluster_assoc_from_graph(H, nC, n_nodes)

    # Remove assignments lacking cluster support.
    gain_mat_post_c = np.array(gain_mat_full, copy=True)
    l_mat_post_c = gain_mat_post_c[:nT, :m]
    for e in np.nonzero(to_be_broken)[0]:
        c = int(edge_arr[0, e])
        jj = int(edge_arr[1, e]) - nC
        h = clusters[begsC[c - 1] - 1:endsC[c - 1]]
        a_remove = pick_ind_c(h, hypos_card)
        if a_remove.size:
            t = np.unique(hypos[a_remove - 1])
            for tt in t:
                l_mat_post_c[tt - 1, jj - 1] = -np.inf
    gain_mat_post_c[:nT, :m] = l_mat_post_c
    return gain_mat_post_c, cluster_assoc
