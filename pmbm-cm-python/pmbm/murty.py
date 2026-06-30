"""Murty's algorithm for the k-best 2D assignments.

Port of the functionality of ``murty.m`` (Murty, 1968). The original MATLAB
code shipped with a hand-written implementation that wraps a Munkres solver
(``assignmentoptimal``). Here we use :func:`scipy.optimize.linear_sum_assignment`
as the base optimal-assignment solver and implement the standard
partitioning of the solution space to rank the assignments by increasing cost.

The cost matrix is minimised. Forbidden assignments are encoded as ``+inf``.
For the rectangular case the number of rows must not exceed the number of
columns, so that every row is assigned to a distinct column (this is always
the situation in the PMBM data-association problem).
"""

import heapq
import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

# Large finite value substituted for +inf so that linear_sum_assignment can run.
_LARGE = 1e12


def _solve_node(cost, included, excluded):
    """Optimal assignment subject to forced (included) and forbidden (excluded)
    row/column pairs.

    Parameters
    ----------
    cost : ndarray (n, m), n <= m
    included : list of (row, col) pairs that must be in the assignment.
    excluded : set of (row, col) pairs that must not be in the assignment.

    Returns
    -------
    (total_cost, assignment) where ``assignment`` is an int array of length n
    giving the column assigned to each row, or ``None`` if infeasible.
    """
    n, m = cost.shape

    assignment = -np.ones(n, dtype=int)
    base_cost = 0.0
    forced_rows = set()
    forced_cols = set()
    for (r, c) in included:
        if not np.isfinite(cost[r, c]):
            return None
        assignment[r] = c
        base_cost += cost[r, c]
        forced_rows.add(r)
        forced_cols.add(c)

    free_rows = [r for r in range(n) if r not in forced_rows]
    free_cols = [c for c in range(m) if c not in forced_cols]

    if not free_rows:
        return base_cost, assignment

    sub = cost[np.ix_(free_rows, free_cols)].astype(float).copy()
    for (r, c) in excluded:
        if r in forced_rows or c in forced_cols:
            continue
        ri = free_rows.index(r)
        ci = free_cols.index(c)
        sub[ri, ci] = np.inf

    work = np.where(np.isinf(sub), _LARGE, sub)
    rows, cols = linear_sum_assignment(work)

    total = base_cost
    for a, b in zip(rows, cols):
        rr = free_rows[a]
        cc = free_cols[b]
        if not np.isfinite(sub[a, b]):
            return None
        assignment[rr] = cc
        total += sub[a, b]
    return total, assignment


def murty(cost, k_best):
    """Return the ``k_best`` lowest-cost assignments of rows to columns.

    Parameters
    ----------
    cost : ndarray (n, m), n <= m
        Cost matrix to minimise. ``+inf`` marks forbidden assignments.
    k_best : int
        Number of best assignments requested.

    Returns
    -------
    assignments : ndarray (n_sol, n)
        Each row is an assignment: column index (0-based) for each row.
    costs : ndarray (n_sol,)
        Total cost of each assignment, sorted ascending.

    ``n_sol`` may be smaller than ``k_best`` if fewer feasible assignments
    exist.
    """
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape

    if k_best <= 0 or n == 0:
        return np.zeros((0, n), dtype=int), np.zeros(0)

    first = _solve_node(cost, [], set())
    if first is None:
        return np.zeros((0, n), dtype=int), np.zeros(0)

    # Priority queue of nodes: (cost, tie_breaker, included, excluded, assignment)
    counter = itertools.count()
    heap = []
    c0, a0 = first
    heapq.heappush(heap, (c0, next(counter), [], set(), a0))

    assignments = []
    costs = []

    while heap and len(assignments) < k_best:
        node_cost, _, included, excluded, assignment = heapq.heappop(heap)
        assignments.append(assignment.copy())
        costs.append(node_cost)

        # Partition the solution space relative to this node's assignment.
        included_rows = {r for r, _ in included}
        free = [r for r in range(n) if r not in included_rows]

        new_included = list(included)
        new_excluded = set(excluded)
        for r in free:
            c = assignment[r]
            # Branch: exclude this (row, col) pair, keep earlier ones included.
            branch_excluded = set(new_excluded)
            branch_excluded.add((r, c))
            child = _solve_node(cost, new_included, branch_excluded)
            if child is not None:
                cc, ca = child
                heapq.heappush(
                    heap,
                    (cc, next(counter), list(new_included), branch_excluded, ca),
                )
            # For subsequent branches this pair becomes forced.
            new_included = new_included + [(r, c)]
            new_excluded.add((r, c))

    return np.array(assignments, dtype=int), np.array(costs, dtype=float)
