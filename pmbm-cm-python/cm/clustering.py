"""Clustering preprocessing: assign measurements to clusters and break weak
inter-cluster edges.

Port of ``clusteringPreprocess.m``. The MATLAB uses graph-toolbox functions
(``graph``, ``centrality 'betweenness'``, ``rmedge``, ``conncomp``,
``groupcounts``); here we use ``networkx``.

Stage-1 note: at k=1 there are no clusters, so only the trivial ``else`` branch
runs (returns the gain matrix unchanged with empty ``masters``/``assocLocal``).
The heavy graph branch is ported for Stage 2 but is NOT yet validated against
MATLAB (betweenness / connected-component labelling order may differ).
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
    """Heavy graph branch of clusteringPreprocess (Stage-2; unvalidated)."""
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

    # NOTE: the iterative super-cluster size/cardinality balancing loop
    # (script_pmbm91 lines ~74-289) is deferred to Stage 2. For now we accept the
    # initial edge-breaking and derive cluster associations from H.
    labels = _conncomp_labels(H, nC + m)
    cluster_assoc_full = np.zeros(nC + m + 1, dtype=int)
    comp_first = {}
    for node in range(1, nC + m + 1):
        lab = labels[node]
        if lab not in comp_first:
            comp_first[lab] = node
        cluster_assoc_full[node] = comp_first[lab]
    cluster_assoc = cluster_assoc_full[1:nC + 1]

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
