"""Branch-and-bound hypothesis generation for one supercluster.

Faithful port of ``branchAndBoundExplore.m`` and its priority-queue helpers
(``pqInitialize``, ``pqsolve``, ``pqsolve2``, ``pqAppend``, ``pqAppendSwitch``,
``constructSwitchReward``, ``boundCalculate``, ``testBAndCpqI``).

The algorithm (Brekke, Miller-style outer/inner Murty over clusters) finds the
``nHypoTotalMax`` best global hypotheses formed by combining one child
hypothesis from each old cluster in the supercluster, respecting measurement
contention between clusters.

This is **equivalent, not bit-exact**: there is no MATLAB reference to validate
against, and the assignment tie-breaking differs from Crouse's solver in
degenerate cases. A priority-queue entry is a ``dict``; ``clustercontrib`` is a
list of ``dict`` (one per old cluster); search-tree indices
(``parentInSearchTree``/``childrenInSearchTree``) are 1-based into the queue,
remapped on every re-sort exactly as MATLAB does.
"""

import copy

import numpy as np

from .bbhelpers import crouse2d
from .clouds import pick_ind_c, tcloud_to_beg, tcloud_to_end


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _ismember_pos(a, b):
    """1-based position of each element of ``a`` in ``b`` (first match, 0 if
    absent). Mirrors the second output of MATLAB ``ismember``."""
    a = np.atleast_1d(np.asarray(a)).ravel()
    pos = {}
    for i, v in enumerate(np.asarray(b).ravel()):
        if v not in pos:
            pos[v] = i + 1
    return np.array([pos.get(v, 0) for v in a], dtype=int)


def _expandability(reward_matrix, person_to_item, super_min_val):
    """Per-row expandability: a track is expandable if, after removing its
    assigned cell, it still has an admissible (> superMinVal) measurement."""
    temp = np.array(reward_matrix, dtype=float)
    for i in range(person_to_item.size):
        temp[i, person_to_item[i] - 1] = super_min_val
    if temp.size == 0:
        return np.zeros(temp.shape[0], dtype=bool)
    return (temp > super_min_val).any(axis=1)


def _matlab_max_assign_grow(row, idx1, val):
    """MATLAB ``row(idx1) = val`` with out-of-bounds auto-grow (zero padding),
    then ``max(row)``. Reproduces the indexing quirk in the expansion heuristic
    of ``branchAndBoundExplore`` (``tRow(p.meaEntireOpt(tt)) = superMinVal``)."""
    row = np.asarray(row, dtype=float).ravel()
    if idx1 > row.size:
        grown = np.zeros(idx1)
        grown[:row.size] = row
        grown[idx1 - 1] = val
        return grown.max()
    row = row.copy()
    row[idx1 - 1] = val
    return row.max()


def _paired_sum(matrix, rows1, cols1):
    """Sum of ``matrix[rows, cols]`` paired (MATLAB ``sum(M(m2v([r;c],size)))``).
    ``rows1``/``cols1`` are 1-based."""
    rows1 = np.asarray(rows1, dtype=int).ravel()
    cols1 = np.asarray(cols1, dtype=int).ravel()
    if rows1.size == 0:
        return 0.0
    return float(matrix[rows1 - 1, cols1 - 1].sum())


# ---------------------------------------------------------------------------
# pqInitialize
# ---------------------------------------------------------------------------

def _empty_contrib():
    return {"tracks": np.zeros(0, dtype=int), "parentInBestList": 1,
            "parentPriorScore": 0.0, "expandable": np.zeros(0, dtype=bool),
            "switchable": False, "tracksLocalInSuper": np.zeros(0, dtype=int),
            "meaIndOptLocalInSuper": np.zeros(0, dtype=int), "posteriorScore": 0.0}


def pq_initialize(best_local, label_hypo, gain_mat_post_c, super_min_val):
    """Port of ``pqInitialize.m``: build the root priority-queue entry."""
    nCL = len(best_local)
    n_hl = np.array([best_local[ii]["scores"].size for ii in range(nCL)], dtype=int)
    max_nhl = int(n_hl.max()) if nCL else 0

    tracks_row, tracks_cl, tracks_blevel, mea_row = [], [], [], []
    for ii in range(nCL):
        cc1 = best_local[ii]["clustercontrib"][0]
        t = np.asarray(cc1["tracks"], dtype=int).ravel()
        tracks_row.extend(t.tolist())
        tracks_cl.extend([ii + 1] * t.size)
        tracks_blevel.extend(range(1, t.size + 1))
        ma = np.asarray(cc1["meaIndAll"], dtype=int).ravel()
        mea_row.extend(ma.tolist())
    tracks_row = np.array(tracks_row, dtype=int)
    tracks_cl = np.array(tracks_cl, dtype=int)
    tracks_blevel = np.array(tracks_blevel, dtype=int)
    mea_ind_unique = np.unique(np.array(mea_row, dtype=int)) if mea_row \
        else np.zeros(0, dtype=int)

    pq1 = {
        "score": 0.0, "bound": 0.0, "rewardMatrix": np.zeros((0, 0)),
        "tracksEntire": tracks_row, "meaEntire": mea_ind_unique,
        "meaEntireOpt": np.zeros(0, dtype=int), "clustercontrib": [],
        "tracksBLevel": tracks_blevel, "tracksCLevel": tracks_cl,
        "customer2Item": np.zeros(0, dtype=int), "parentInSearchTree": 0,
        "childrenInSearchTree": np.zeros(0, dtype=int), "childrenCount": 0,
        "labelHypo": label_hypo,
        "switchOrExpand": {"switch": False, "expand": False},
        "bestCaseLoss": -np.inf * np.ones((nCL, max_nhl)),
        "upperBoundsOfScores": np.zeros((0, 0)), "hasBeenUUB": False,
    }
    for ii in range(nCL):
        pq1["bestCaseLoss"][ii, :n_hl[ii]] = best_local[ii]["scores"]

    if tracks_row.size:
        reward_matrix = gain_mat_post_c[np.ix_(tracks_row - 1, mea_ind_unique - 1)]
        person_to_item, opt_reward, ubs = crouse2d(reward_matrix)
        if opt_reward < -500:
            raise ValueError("too low optreward in pqInitialize")

        bound = sum(best_local[ii]["scores"][0] for ii in range(nCL))
        score = opt_reward + sum(
            best_local[ii]["clustercontrib"][0]["parentPriorScore"] for ii in range(nCL))
        mea_entire_opt = mea_ind_unique[person_to_item - 1]
        expandability = _expandability(reward_matrix, person_to_item, super_min_val)

        pq1.update({"score": score, "bound": bound, "rewardMatrix": reward_matrix,
                    "meaEntireOpt": mea_entire_opt, "customer2Item": person_to_item,
                    "upperBoundsOfScores": ubs})
        for ii in range(nCL):
            cc1 = best_local[ii]["clustercontrib"][0]
            tracks = np.asarray(cc1["tracks"], dtype=int).ravel()
            tlis = _ismember_pos(tracks, tracks_row)
            contrib = {"tracks": tracks, "parentInBestList": 1,
                       "parentPriorScore": cc1["parentPriorScore"],
                       "tracksLocalInSuper": tlis,
                       "expandable": expandability[tlis - 1] if tlis.size
                       else np.zeros(0, dtype=bool),
                       "switchable": len(best_local[ii]["clustercontrib"]) > 1}
            if tracks.size:
                meo = mea_entire_opt[tracks_cl == ii + 1]
                miols = _ismember_pos(meo, mea_ind_unique)
                contrib["meaIndOptLocalInSuper"] = miols
                contrib["posteriorScore"] = _paired_sum(reward_matrix, tlis, miols) \
                    + cc1["parentPriorScore"]
            else:
                contrib["meaIndOptLocalInSuper"] = np.zeros(0, dtype=int)
                contrib["posteriorScore"] = cc1["parentPriorScore"]
            pq1["clustercontrib"].append(contrib)
    else:
        score = sum(best_local[ii]["clustercontrib"][0]["parentPriorScore"]
                    for ii in range(nCL))
        bound = sum(best_local[ii]["scores"][0] for ii in range(nCL))
        pq1.update({"score": score, "bound": bound})
        for ii in range(nCL):
            cc1 = best_local[ii]["clustercontrib"][0]
            pq1["clustercontrib"].append({
                "tracks": np.zeros(0, dtype=int), "parentInBestList": 1,
                "parentPriorScore": cc1["parentPriorScore"],
                "expandable": np.zeros(0, dtype=bool),
                "switchable": len(best_local[ii]["clustercontrib"]) > 1,
                "tracksLocalInSuper": np.zeros(0, dtype=int),
                "meaIndOptLocalInSuper": np.zeros(0, dtype=int),
                "posteriorScore": cc1["parentPriorScore"]})
    return [pq1]


# ---------------------------------------------------------------------------
# testBAndCpqI
# ---------------------------------------------------------------------------

def _test_b_and_c(pq_entry):
    """Port of ``testBAndCpqI.m``: B-level indices within each cluster must be a
    contiguous 1..n run."""
    c_level = np.asarray(pq_entry["tracksCLevel"], dtype=int).ravel()
    b_level = np.asarray(pq_entry["tracksBLevel"], dtype=int).ravel()
    if c_level.size == 0:
        return True
    for ii in range(1, int(c_level.max()) + 1):
        b_entries = b_level[c_level == ii]
        if b_entries.size:
            if b_entries.max() != b_entries.size:
                return False
    return True


# ---------------------------------------------------------------------------
# constructSwitchReward
# ---------------------------------------------------------------------------

def construct_switch_reward(pqI, to_be_switched, next_parents, best_local,
                            super_min_val, gain_mat_post_c):
    """Port of ``constructSwitchReward.m``."""
    nCL = len(best_local)
    to_be_switched = np.asarray(to_be_switched, dtype=int).ravel()
    next_parents = np.asarray(next_parents, dtype=int).ravel()

    tracks_i_switch, pre_sort = [], []
    for jj in range(to_be_switched.size):
        cl, npar = to_be_switched[jj], next_parents[jj]
        t = np.asarray(best_local[cl - 1]["clustercontrib"][npar - 1]["tracks"],
                       dtype=int).ravel()
        tracks_i_switch.extend(t.tolist())
        pre_sort.extend([cl] * t.size)
    tracks_i_switch = np.array(tracks_i_switch, dtype=int)
    pre_sort = np.array(pre_sort, dtype=int)
    tracks_b_inds = list(range(1, pre_sort.size + 1))

    tracks_legacy, tracks_legacy_local = [], []
    not_switched = [j for j in range(1, nCL + 1) if j not in to_be_switched.tolist()]
    for jj in not_switched:
        cc = pqI["clustercontrib"][jj - 1]
        t = np.asarray(cc["tracks"], dtype=int).ravel()
        tracks_legacy.extend(t.tolist())
        tracks_legacy_local.extend(np.asarray(cc["tracksLocalInSuper"], dtype=int).ravel().tolist())
        pre_sort = np.append(pre_sort, [jj] * t.size)
        tracks_b_inds.extend(range(1, t.size + 1))
    tracks_legacy = np.array(tracks_legacy, dtype=int)
    tracks_legacy_local = np.array(tracks_legacy_local, dtype=int)
    tracks_b_inds = np.array(tracks_b_inds, dtype=int)

    rm = pqI["rewardMatrix"]
    if tracks_legacy_local.size:
        mil_super = np.nonzero(
            (rm[tracks_legacy_local - 1, :] > super_min_val).any(axis=0))[0] + 1
    else:
        mil_super = np.zeros(0, dtype=int)
    mea_legacy = pqI["meaEntire"][mil_super - 1] if mil_super.size \
        else np.zeros(0, dtype=int)
    if tracks_i_switch.size:
        mi_switch = np.nonzero(
            (gain_mat_post_c[tracks_i_switch - 1, :] > super_min_val).any(axis=0))[0] + 1
    else:
        mi_switch = np.zeros(0, dtype=int)
    mea_all = np.union1d(mi_switch, mea_legacy).astype(int)
    mea_i_not_legacy = np.setdiff1d(mi_switch, mea_legacy).astype(int)

    rew_i_switch = gain_mat_post_c[np.ix_(tracks_i_switch - 1, mea_all - 1)] \
        if tracks_i_switch.size else np.zeros((0, mea_all.size))
    rew_legacy_raw = rm[np.ix_(tracks_legacy_local - 1, mil_super - 1)] \
        if tracks_legacy_local.size else np.zeros((0, mil_super.size))

    ix = np.argsort(pre_sort, kind="stable")
    cluster_per_track = pre_sort[ix]
    tracks_all_pre = np.concatenate([tracks_i_switch, tracks_legacy])
    tracks_all_switch = tracks_all_pre[ix]

    n_leg_rows = rew_legacy_raw.shape[0]
    n_cols = rew_legacy_raw.shape[1] + mea_i_not_legacy.size
    rew_legacy_exp = super_min_val * np.ones((n_leg_rows, n_cols))
    if mea_legacy.size:
        b = _ismember_pos(mea_legacy, mea_all)
        rew_legacy_exp[:, b - 1] = rew_legacy_raw
    rew_combined_raw = np.vstack([rew_i_switch, rew_legacy_exp])
    reward_matrix_combined = rew_combined_raw[ix, :]
    tracks_b_inds = tracks_b_inds[ix]

    return (reward_matrix_combined, tracks_all_switch, mea_all, tracks_b_inds,
            cluster_per_track)


# ---------------------------------------------------------------------------
# pqAppend / pqAppendSwitch
# ---------------------------------------------------------------------------

def _pq_append_common(pqI, parent_in_tree, best_local, to_be_switched, next_parents,
                      t2m_forbidden, reward_matrix_combined, opt_reward,
                      tracks_all_switch, tracks_b_inds, mea_ind_all_switch,
                      cluster_per_track, expandability_test, person_to_item,
                      label_hypo, ubs, is_expand, keep_legacy_switchable):
    nCL = len(best_local)
    to_be_switched = np.asarray(to_be_switched, dtype=int).ravel()
    next_parents = np.asarray(next_parents, dtype=int).ravel()
    cluster_per_track = np.asarray(cluster_per_track, dtype=int).ravel()
    person_to_item = np.asarray(person_to_item, dtype=int).ravel()
    mea_eo = mea_ind_all_switch[person_to_item - 1] if person_to_item.size \
        else np.zeros(0, dtype=int)
    bcl_copy = np.array(pqI["bestCaseLoss"], dtype=float)

    newcontrib = [None] * nCL
    for jj in range(to_be_switched.size):
        j_switch = to_be_switched[jj]
        npar = next_parents[jj]
        newcontrib[j_switch - 1] = {
            "parentPriorScore": best_local[j_switch - 1]["clustercontrib"][npar - 1]["parentPriorScore"],
            "parentInBestList": npar,
            "switchable": bool(np.any(~np.isinf(bcl_copy[j_switch - 1, npar:])))}
    for jj in range(1, nCL + 1):
        if jj in to_be_switched.tolist():
            continue
        if keep_legacy_switchable:
            sw = pqI["clustercontrib"][jj - 1]["switchable"]
        else:
            sw = False
        newcontrib[jj - 1] = {
            "switchable": sw,
            "parentInBestList": pqI["clustercontrib"][jj - 1]["parentInBestList"],
            "parentPriorScore": pqI["clustercontrib"][jj - 1]["parentPriorScore"]}

    score = opt_reward if opt_reward is not None else 0.0
    for jj in range(1, nCL + 1):
        c = newcontrib[jj - 1]
        tlis = np.nonzero(cluster_per_track == jj)[0] + 1
        c["tracksLocalInSuper"] = tlis
        c["tracks"] = tracks_all_switch[cluster_per_track == jj]
        c["expandable"] = expandability_test[tlis - 1] if tlis.size \
            else np.zeros(0, dtype=bool)
        score += c["parentPriorScore"]
        if tlis.size:
            miols = _ismember_pos(mea_eo[cluster_per_track == jj], mea_ind_all_switch)
            c["meaIndOptLocalInSuper"] = miols
            c["posteriorScore"] = _paired_sum(reward_matrix_combined, tlis, miols) \
                + c["parentPriorScore"]
        else:
            c["meaIndOptLocalInSuper"] = np.zeros(0, dtype=int)
            c["posteriorScore"] = c["parentPriorScore"]

    label_hypo += 1
    pqO = {
        "clustercontrib": newcontrib, "score": score,
        "rewardMatrix": reward_matrix_combined, "tracksEntire": tracks_all_switch,
        "meaEntire": mea_ind_all_switch, "meaEntireOpt": mea_eo,
        "tracksCLevel": cluster_per_track, "tracksBLevel": tracks_b_inds,
        "customer2Item": person_to_item, "parentInSearchTree": parent_in_tree,
        "childrenInSearchTree": np.zeros(0, dtype=int), "childrenCount": 0,
        "labelHypo": label_hypo,
        "switchOrExpand": {"switch": to_be_switched.size > 0, "expand": is_expand},
        "bestCaseLoss": bcl_copy, "bound": 0.0, "upperBoundsOfScores": ubs,
        "hasBeenUUB": False}
    if not _test_b_and_c(pqO):
        raise ValueError("B/C level index inconsistency in pqAppend")
    pqO = bound_calculate(pqO, best_local)
    return pqO, label_hypo


def pq_append(pqI, parent_in_tree, best_local, to_be_switched, next_parents,
              t2m_forbidden, reward_matrix_combined, opt_reward, tracks_all_switch,
              tracks_b_inds, mea_ind_all_switch, cluster_per_track,
              expandability_test, person_to_item, label_hypo, ubs):
    """Port of ``pqAppend.m`` (expansion path; legacy clusters become
    non-switchable)."""
    is_expand = np.asarray(t2m_forbidden).size > 0
    return _pq_append_common(pqI, parent_in_tree, best_local, to_be_switched,
                             next_parents, t2m_forbidden, reward_matrix_combined,
                             opt_reward, tracks_all_switch, tracks_b_inds,
                             mea_ind_all_switch, cluster_per_track,
                             expandability_test, person_to_item, label_hypo, ubs,
                             is_expand=is_expand, keep_legacy_switchable=False)


def pq_append_switch(pqI, parent_in_tree, best_local, to_be_switched, next_parents,
                     reward_matrix_combined, opt_reward, tracks_all_switch,
                     tracks_b_inds, mea_ind_all_switch, cluster_per_track,
                     expandability_test, person_to_item, label_hypo, ubs):
    """Port of ``pqAppendSwitch.m`` (switch path; legacy switchability kept)."""
    return _pq_append_common(pqI, parent_in_tree, best_local, to_be_switched,
                             next_parents, np.zeros(0, dtype=int),
                             reward_matrix_combined, opt_reward, tracks_all_switch,
                             tracks_b_inds, mea_ind_all_switch, cluster_per_track,
                             expandability_test, person_to_item, label_hypo, ubs,
                             is_expand=False, keep_legacy_switchable=True)


# ---------------------------------------------------------------------------
# boundCalculate
# ---------------------------------------------------------------------------

def bound_calculate(pqI, best_local):
    """Port of ``boundCalculate.m``: tighten the bound of a freshly-built entry."""
    soe = pqI["switchOrExpand"]
    nCL = len(best_local)
    if soe["switch"] and not soe["expand"]:
        if pqI["bound"] != 0:
            raise ValueError("boundCalculate expects zero bound")
        bound = 0.0
        for ii in range(nCL):
            pib = pqI["clustercontrib"][ii]["parentInBestList"]
            bound += best_local[ii]["scores"][pib - 1]
        pqI["bound"] = bound

        bcl = pqI["bestCaseLoss"]
        bound_reductions = np.inf * np.ones(nCL)
        for jj in range(nCL):
            if pqI["clustercontrib"][jj]["switchable"]:
                pib = pqI["clustercontrib"][jj]["parentInBestList"]
                nxt = bcl[jj, pib] if pib < bcl.shape[1] else -np.inf
                bound_reductions[jj] = bcl[jj, pib - 1] - nxt
                if np.isinf(bound_reductions[jj]):
                    pqI["clustercontrib"][jj]["switchable"] = False
        best_reduction = bound_reductions.min()
        best_candidate = int(np.argmin(bound_reductions))

        any_switch = any(c["switchable"] for c in pqI["clustercontrib"])
        if any_switch and not np.isinf(best_reduction):
            reduced = 0.0
            for jj in range(nCL):
                pib = pqI["clustercontrib"][jj]["parentInBestList"]
                if jj == best_candidate:
                    reduced += bcl[jj, pib]
                else:
                    reduced += bcl[jj, pib - 1]
        else:
            reduced = -np.inf
        if best_reduction < 0:
            raise ValueError("bestBoundReduction should be non-negative")
        pqI["bound"] = max(reduced, pqI["score"])
    elif soe["expand"] and not soe["switch"]:
        pqI["bound"] = pqI["score"]
    else:
        raise ValueError("boundCalculate: must be either switch or expand")
    return pqI


# ---------------------------------------------------------------------------
# pqsolve / pqsolve2
# ---------------------------------------------------------------------------

def pqsolve(pqI, best_local, switches, t2m_forbidden, t2m_enforced, super_min_val,
            gain_mat_post_c, label_hypo, parent_in_tree):
    """Port of ``pqsolve.m`` (switch)."""
    switches = np.asarray(switches, dtype=float).ravel()
    to_be_switched = np.nonzero(~np.isnan(switches))[0] + 1
    next_parents = switches[to_be_switched - 1].astype(int)
    (rmc, tracks_all, mea_all, tb_inds, cpt) = construct_switch_reward(
        pqI, to_be_switched, next_parents, best_local, super_min_val, gain_mat_post_c)
    person_to_item, _, ubs = crouse2d(rmc)
    opt_reward = float(sum(rmc[i, person_to_item[i] - 1] for i in range(person_to_item.size)))
    expandability_test = _expandability(rmc, person_to_item, super_min_val)
    return pq_append_switch(pqI, parent_in_tree, best_local, to_be_switched,
                            next_parents, rmc, opt_reward, tracks_all, tb_inds,
                            mea_all, cpt, expandability_test, person_to_item,
                            label_hypo, ubs)


def pqsolve2(pqI, best_local, switches, t2m_forbidden, t2m_enforced, super_min_val,
             gain_mat_post_c, label_hypo, parent_in_tree, min_val):
    """Port of ``pqsolve2.m`` (switch OR expand; returns ``None`` if the
    solution's reward falls below ``min_val``)."""
    switches = np.asarray(switches, dtype=float).ravel()
    to_be_switched = np.nonzero(~np.isnan(switches))[0] + 1
    next_parents = switches[to_be_switched - 1].astype(int)
    t2m_forbidden = np.atleast_1d(np.asarray(t2m_forbidden, dtype=int)).ravel()

    if np.any(~np.isnan(switches)) and t2m_forbidden.size == 0:
        (rmc, tracks_all, mea_all, tb_inds, cpt) = construct_switch_reward(
            pqI, to_be_switched, next_parents, best_local, super_min_val,
            gain_mat_post_c)
    elif t2m_forbidden.size > 0 and not np.any(~np.isnan(switches)):
        rmc = np.array(pqI["rewardMatrix"], dtype=float)
        col = pqI["customer2Item"][t2m_forbidden - 1]
        rmc[t2m_forbidden - 1, col - 1] = super_min_val
        tracks_all = pqI["tracksEntire"]
        mea_all = pqI["meaEntire"]
        tb_inds = pqI["tracksBLevel"]
        cpt = pqI["tracksCLevel"]
    else:
        raise ValueError("pqsolve2: can only switch or expand separately")

    person_to_item, opt_reward, ubs = crouse2d(rmc)
    if opt_reward >= min_val:
        expandability_test = _expandability(rmc, person_to_item, super_min_val)
        return pq_append(pqI, parent_in_tree, best_local, to_be_switched,
                         next_parents, t2m_forbidden, rmc, opt_reward, tracks_all,
                         tb_inds, mea_all, cpt, expandability_test, person_to_item,
                         label_hypo, ubs)
    return None, label_hypo


# ---------------------------------------------------------------------------
# search-tree re-indexing on re-sort
# ---------------------------------------------------------------------------

def _reindex(pq, ix0):
    """Reorder ``pq`` by 0-based permutation ``ix0`` (``pq_new[i]=pq_old[ix0[i]]``)
    and remap 1-based parent/children search-tree indices. Returns ``(new_pq,
    inv)`` where ``inv[old0]=new0``."""
    n = len(pq)
    inv = np.zeros(n, dtype=int)
    inv[ix0] = np.arange(n)
    new_pq = [pq[i] for i in ix0]
    for e in new_pq:
        p = e["parentInSearchTree"]
        e["parentInSearchTree"] = 0 if p == 0 else int(inv[p - 1]) + 1
        ch = np.asarray(e["childrenInSearchTree"], dtype=int).ravel()
        e["childrenInSearchTree"] = (inv[ch - 1] + 1) if ch.size else np.zeros(0, dtype=int)
    return new_pq, inv


# ---------------------------------------------------------------------------
# branchAndBoundExplore
# ---------------------------------------------------------------------------

def branch_and_bound_explore(hypos, hypos_card, clusters, clusters_card,
                             prob_log_hypos, iC, assoc_local, gain_mat_post_c,
                             indices_of_newborn_tracks, n_hypo_total_max,
                             track_number_lookup, k):
    """Port of ``branchAndBoundExplore.m``. ``iC`` is the 1-based supercluster
    (master) index. Returns ``(hypos_local, hypos_card_local, prob_log_local)``.
    """
    hypos = np.asarray(hypos, dtype=int).ravel()
    hypos_card = np.asarray(hypos_card, dtype=int).ravel()
    clusters = np.asarray(clusters, dtype=int).ravel()
    clusters_card = np.asarray(clusters_card, dtype=int).ravel()
    prob_log_hypos = np.asarray(prob_log_hypos, dtype=float).ravel()
    assoc_local = np.asarray(assoc_local, dtype=float)
    gain_mat_post_c = np.array(gain_mat_post_c, dtype=float)
    indices_of_newborn_tracks = np.asarray(indices_of_newborn_tracks, dtype=int).ravel()
    track_number_lookup = np.asarray(track_number_lookup, dtype=float)

    min_val = -30.0
    super_min_val = -1000.0
    m = indices_of_newborn_tracks.size
    nT = gain_mat_post_c.shape[0] - m
    l_mat_post_c = gain_mat_post_c[:nT, :m].copy()  # BEFORE inf -> superMinVal
    masters = np.nonzero(assoc_local[1, :] == 1)[0] + 1

    gain_mat_post_c[np.isinf(gain_mat_post_c)] = super_min_val

    into_this = np.nonzero(assoc_local[0, :] == masters[iC - 1])[0] + 1
    nCL = into_this.size

    begs_hypo = tcloud_to_beg(hypos_card)
    ends_hypo = tcloud_to_end(hypos_card)

    empty = (np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0))

    # -- Phase 1: best child hypothesis for each parent in each cluster --------
    best_local = []
    for ii in range(nCL):
        parent_hypos = clusters[pick_ind_c(into_this[ii], clusters_card) - 1]
        scores = []
        contribs = []
        for iH in range(parent_hypos.size):
            ph = parent_hypos[iH]
            tracks_h = hypos[begs_hypo[ph - 1] - 1:ends_hypo[ph - 1]]
            if tracks_h.size:
                gated_h = np.nonzero(
                    (~np.isinf(l_mat_post_c[tracks_h - 1, :])).any(axis=0))[0] + 1
                mea_ind_h = np.concatenate([gated_h, m + tracks_h])
                gain_mat_h = gain_mat_post_c[np.ix_(tracks_h - 1, mea_ind_h - 1)]
                person_to_item, opt_reward, ubs = crouse2d(gain_mat_h)
                scores.append(opt_reward + prob_log_hypos[ph - 1])
                contribs.append({
                    "tracks": tracks_h, "meaIndAll": mea_ind_h,
                    "meaIndOpt": mea_ind_h[person_to_item - 1],
                    "meaIndOptLocal": person_to_item, "parentHypo": ph,
                    "parentPriorScore": prob_log_hypos[ph - 1],
                    "upperBoundsOfScores": ubs})
            else:
                scores.append(prob_log_hypos[ph - 1])
                contribs.append({
                    "tracks": np.zeros(0, dtype=int), "meaIndAll": np.zeros(0, dtype=int),
                    "meaIndOpt": np.zeros(0, dtype=int), "meaIndOptLocal": np.zeros(0, dtype=int),
                    "parentHypo": ph, "parentPriorScore": prob_log_hypos[ph - 1],
                    "upperBoundsOfScores": np.zeros((0, 0))})
        scores = np.array(scores, dtype=float)
        order = np.argsort(-scores, kind="stable")
        best_local.append({"scores": scores[order],
                           "clustercontrib": [contribs[o] for o in order]})

    if not best_local:
        return empty

    # -- Phase 2: branch-and-bound --------------------------------------------
    label_hypo = 1
    pq = pq_initialize(best_local, label_hypo, gain_mat_post_c, super_min_val)
    label_hypo = pq[0]["labelHypo"]
    pq[0]["hasBeenUUB"] = True
    if not pq:
        return empty

    uub = pq[0]["bound"]
    lpb = super_min_val / 2.0
    uub_holder = 1   # 1-based index into pq
    it = 1
    max_iter = 10000

    while uub > lpb and it <= max_iter:
        it += 1

        # ---- Aggressive switching ------------------------------------------
        if pq[uub_holder - 1]["score"] < pq[uub_holder - 1]["bound"] - 1e-13:
            best_score_as = pq[uub_holder - 1]["score"]
            unresolved = [uub_holder]
            s_iter = 0
            while s_iter < max_iter and unresolved:
                s_iter += 1
                bounds_unres = np.array([pq[u - 1]["bound"] for u in unresolved])
                ps_entry = unresolved[int(np.argmax(bounds_unres))]
                p = copy.deepcopy(pq[ps_entry - 1])

                bcl_actual = -np.inf * np.ones(nCL)
                for ii in range(nCL):
                    pib = p["clustercontrib"][ii]["parentInBestList"]
                    if pib + 1 <= best_local[ii]["scores"].size:
                        bcl_actual[ii] = p["bestCaseLoss"][ii, pib] - p["bestCaseLoss"][ii, pib - 1]
                six = np.argsort(-bcl_actual, kind="stable") + 1
                svals = bcl_actual[six - 1]
                switchabilities = np.array([c["switchable"] for c in p["clustercontrib"]])
                keep = (~np.isinf(svals)) & (p["bound"] + svals > p["score"]) \
                    & switchabilities[six - 1]
                as_murty_order = six[keep]

                for ii in range(as_murty_order.size):
                    y_cluster = as_murty_order[ii]
                    z_for_y = p["clustercontrib"][y_cluster - 1]["parentInBestList"]
                    p_marked = copy.deepcopy(p)
                    p_marked["bestCaseLoss"][y_cluster - 1, z_for_y - 1] = -np.inf

                    a = p_marked["bestCaseLoss"].max(axis=1)
                    s_mar = p_marked["bestCaseLoss"].argmax(axis=1) + 1  # 1-based
                    cur_parents = np.array(
                        [c["parentInBestList"] for c in p_marked["clustercontrib"]])
                    switch_bool = (~np.isinf(a)) & (s_mar != cur_parents)

                    if np.any(switch_bool):
                        switches = np.full(nCL, np.nan)
                        switches[switch_bool] = s_mar[switch_bool]
                        subs2 = cur_parents.copy()
                        subs2[switch_bool] = s_mar[switch_bool]
                        losses = p_marked["bestCaseLoss"][np.arange(nCL), subs2 - 1]

                        if losses.sum() >= lpb:
                            s_marked, label_hypo = pqsolve(
                                p_marked, best_local, switches, np.zeros(0, dtype=int),
                                np.zeros(0, dtype=int), super_min_val,
                                gain_mat_post_c, label_hypo, ps_entry)
                            pq.append(s_marked)
                            new_idx = len(pq)
                            pq[ps_entry - 1]["childrenInSearchTree"] = np.append(
                                pq[ps_entry - 1]["childrenInSearchTree"], new_idx).astype(int)
                            pq[ps_entry - 1]["childrenCount"] += 1
                            best_score_as = max(best_score_as, s_marked["score"])

                            ch = pq[ps_entry - 1]["childrenInSearchTree"]
                            if ps_entry in unresolved:
                                pass
                            if s_marked["bound"] > pq[uub_holder - 1]["score"]:
                                unresolved.append(new_idx)
                            unres_bounds = np.array([pq[u - 1]["bound"] for u in unresolved])
                            unresolved = [u for u, b in zip(unresolved, unres_bounds)
                                          if not (b < s_marked["score"])]
                        # enforce y-z in p
                        p["bestCaseLoss"][y_cluster - 1, :z_for_y - 1] = -np.inf
                        p["bestCaseLoss"][y_cluster - 1, z_for_y:] = -np.inf
                    pq[ps_entry - 1]["clustercontrib"][as_murty_order[ii] - 1]["switchable"] = False

                # drop p's label from unresolved
                unresolved = [u for u in unresolved if pq[u - 1]["labelHypo"] != p["labelHypo"]]

                score_sorted = np.sort([e["score"] for e in pq])[::-1]
                if score_sorted.size >= n_hypo_total_max:
                    lpb = max(lpb, score_sorted[n_hypo_total_max - 1])

            # sort pq by [bound desc, score desc]
            bounds = np.array([e["bound"] for e in pq])
            scores = np.array([e["score"] for e in pq])
            ixV = np.lexsort((-scores, -bounds))
            pq, inv = _reindex(pq, ixV)
            uub_holder = int(inv[uub_holder - 1]) + 1

        # ---- Lazy switches -------------------------------------------------
        p = copy.deepcopy(pq[uub_holder - 1])
        if any(c["switchable"] for c in p["clustercontrib"]):
            bcl_actual = -np.inf * np.ones(nCL)
            for ii in range(nCL):
                pib = p["clustercontrib"][ii]["parentInBestList"]
                if pib + 1 <= best_local[ii]["scores"].size:
                    bcl_actual[ii] = p["bestCaseLoss"][ii, pib] - p["bestCaseLoss"][ii, pib - 1]
            six = np.argsort(-bcl_actual, kind="stable") + 1
            svals = bcl_actual[six - 1]
            switchabilities = np.array([c["switchable"] for c in p["clustercontrib"]])
            keep = (~np.isinf(svals)) & switchabilities[six - 1]
            as_murty_order = six[keep]

            for ii in range(as_murty_order.size):
                y_cluster = as_murty_order[ii]
                z_for_y = p["clustercontrib"][y_cluster - 1]["parentInBestList"]
                p_marked = copy.deepcopy(p)
                p_marked["bestCaseLoss"][y_cluster - 1, z_for_y - 1] = -np.inf
                a = p_marked["bestCaseLoss"].max(axis=1)
                s_mar = p_marked["bestCaseLoss"].argmax(axis=1) + 1
                cur_parents = np.array(
                    [c["parentInBestList"] for c in p_marked["clustercontrib"]])
                switch_bool = (~np.isinf(a)) & (s_mar != cur_parents)
                if np.any(switch_bool):
                    switches = np.full(nCL, np.nan)
                    switches[switch_bool] = s_mar[switch_bool]
                    subs2 = cur_parents.copy()
                    subs2[switch_bool] = s_mar[switch_bool]
                    losses = p_marked["bestCaseLoss"][np.arange(nCL), subs2 - 1]
                    if losses.sum() >= lpb:
                        s_marked, label_hypo = pqsolve(
                            p_marked, best_local, switches, np.zeros(0, dtype=int),
                            np.zeros(0, dtype=int), super_min_val, gain_mat_post_c,
                            label_hypo, uub_holder)
                        pq.append(s_marked)
                        new_idx = len(pq)
                        pq[uub_holder - 1]["childrenInSearchTree"] = np.append(
                            pq[uub_holder - 1]["childrenInSearchTree"], new_idx).astype(int)
                        pq[uub_holder - 1]["childrenCount"] += 1
                    p["bestCaseLoss"][y_cluster - 1, :z_for_y - 1] = -np.inf
                    p["bestCaseLoss"][y_cluster - 1, z_for_y:] = -np.inf
                pq[uub_holder - 1]["clustercontrib"][as_murty_order[ii] - 1]["switchable"] = False

        # ---- Expansions ----------------------------------------------------
        expandable_any = any(
            np.asarray(c["expandable"]).any() for c in p["clustercontrib"]
            if np.asarray(c["expandable"]).size)
        if expandable_any:
            tracks_entire = p["tracksEntire"]
            score_bounds = -np.inf * np.ones(tracks_entire.size)
            for tt in range(tracks_entire.size):
                c_level = p["tracksCLevel"][tt]
                b_level = p["tracksBLevel"][tt]
                exp_arr = np.asarray(p["clustercontrib"][c_level - 1]["expandable"])
                if exp_arr.size and exp_arr[b_level - 1]:
                    t_row = p["upperBoundsOfScores"][tt, :]
                    score_bounds[tt] = _matlab_max_assign_grow(
                        t_row, int(p["meaEntireOpt"][tt]), super_min_val)
            track_order = np.argsort(score_bounds, kind="stable") + 1  # ascend
            sbs = score_bounds[track_order - 1]

            for tt in range(track_order.size):
                if sbs[tt] > lpb:
                    pqO, label_hypo = pqsolve2(
                        copy.deepcopy(p), best_local, np.full(nCL, np.nan),
                        np.array([track_order[tt]], dtype=int), np.zeros(0, dtype=int),
                        super_min_val, gain_mat_post_c, label_hypo, uub_holder, min_val)
                    if pqO is not None:
                        if pqO["bound"] >= lpb:
                            pq.append(pqO)
                            new_idx = len(pq)
                            pq[uub_holder - 1]["childrenInSearchTree"] = np.append(
                                pq[uub_holder - 1]["childrenInSearchTree"], new_idx).astype(int)
                            pq[uub_holder - 1]["childrenCount"] += 1
                        else:
                            label_hypo -= 1
                else:
                    c_level = pq[uub_holder - 1]["tracksCLevel"][tt]
                    b_level = pq[uub_holder - 1]["tracksBLevel"][tt]
                    pq[uub_holder - 1]["clustercontrib"][c_level - 1]["expandable"][b_level - 1] = False

                # enforce t in p before next iteration
                y_for_p = track_order[tt]
                z_for_p = p["customer2Item"][y_for_p - 1]
                p["rewardMatrix"][:y_for_p - 1, z_for_p - 1] = super_min_val
                p["rewardMatrix"][y_for_p:, z_for_p - 1] = super_min_val
                p["rewardMatrix"][y_for_p - 1, :z_for_p - 1] = super_min_val
                p["rewardMatrix"][y_for_p - 1, z_for_p:] = super_min_val
            for ii in range(nCL):
                ea = np.asarray(pq[uub_holder - 1]["clustercontrib"][ii]["expandable"])
                pq[uub_holder - 1]["clustercontrib"][ii]["expandable"] = np.zeros(ea.size, dtype=bool)

        # ---- Identify new UUB ----------------------------------------------
        value_list = np.array([e["bound"] for e in pq])
        if np.any(np.isnan(value_list)):
            raise ValueError("nan in valueList")
        ixV = np.argsort(-value_list, kind="stable")
        new_value_list = value_list[ixV]
        pq, inv = _reindex(pq, ixV)

        score_sorted = np.sort([e["score"] for e in pq])[::-1]
        if score_sorted.size >= n_hypo_total_max:
            lpb = max(lpb, score_sorted[n_hypo_total_max - 1])

        has_been_uub = np.array([e["hasBeenUUB"] for e in pq])
        worse = np.nonzero((new_value_list < uub + 100 * np.finfo(float).eps)
                           & ~has_been_uub)[0]
        if worse.size and uub > lpb:
            uub_holder = int(worse[0]) + 1
            pq[uub_holder - 1]["hasBeenUUB"] = True
            uub = new_value_list[uub_holder - 1]
        else:
            break

    # -- Sort final queue by score, decode hypotheses --------------------------
    value_list = np.array([e["score"] for e in pq])
    ixV = np.argsort(-value_list, kind="stable")
    pq, inv = _reindex(pq, ixV)

    final_count = min(len(pq), n_hypo_total_max)
    h_in_c = clusters[pick_ind_c(into_this, clusters_card) - 1]
    t_in_c = hypos[pick_ind_c(h_in_c, hypos_card) - 1]
    gated_in_c = np.nonzero((~np.isinf(l_mat_post_c[t_in_c - 1, :])).any(axis=0))[0] + 1

    hypos_local, hypos_card_local, prob_log_local = [], [], []
    for jj in range(final_count):
        entry = pq[jj]
        h_new_legacy, mea_by_legacy = [], []
        for ii in range(nCL):
            cc = entry["clustercontrib"][ii]
            down = np.asarray(cc["tracks"], dtype=int).ravel()
            miols = np.asarray(cc["meaIndOptLocalInSuper"], dtype=int).ravel()
            along = entry["meaEntire"][miols - 1] if miols.size else np.zeros(0, dtype=int)
            for iL in range(down.size):
                h_new_legacy.append(track_number_lookup[down[iL] - 1, along[iL] - 1])
                mea_by_legacy.append(along[iL])
        mea_by_legacy = np.array(mea_by_legacy, dtype=int)
        unclaimed = np.setdiff1d(gated_in_c, mea_by_legacy)
        newborn = indices_of_newborn_tracks[unclaimed - 1] if unclaimed.size \
            else np.zeros(0, dtype=int)
        h_new = np.concatenate([np.array(h_new_legacy, dtype=int), newborn]).astype(int)
        hypos_local.extend(h_new.tolist())
        hypos_card_local.append(h_new.size)
        prob_log_local.append(entry["score"])

    return (np.array(hypos_local, dtype=int), np.array(hypos_card_local, dtype=int),
            np.array(prob_log_local, dtype=float))
