"""Generalized Optimal Sub-Pattern Assignment (GOSPA) metric.

Port of ``GOSPA.m`` (Rahmathullah et al., 2017) and ``ComputeGOSPAerror.m``.
The optimal assignment is computed with :func:`scipy.optimize.linear_sum_assignment`
(equivalent to the auction algorithm used in the MATLAB code). The base
distance is the Euclidean distance between the (2D) position vectors.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class GospaDecomposition:
    localisation: float = 0.0
    missed: float = 0.0
    false: float = 0.0


def gospa(x_mat, y_mat, p=2, c=10.0, alpha=2):
    """Compute the GOSPA distance between two finite sets of column vectors.

    Parameters
    ----------
    x_mat : ndarray (d, nx)
        Ground-truth set, one column per element.
    y_mat : ndarray (d, ny)
        Estimated set, one column per element.
    p : float
        Exponent (1 <= p < inf).
    c : float
        Cut-off distance (c > 0).
    alpha : float
        Cardinality-penalty factor (0 < alpha <= 2). Use 2 for the
        localisation / missed / false decomposition.

    Returns
    -------
    d_gospa : float
    assignment : ndarray (nx,)
        Column of ``y_mat`` assigned to each column of ``x_mat`` (-1 = none).
    decomposed : GospaDecomposition
        Valid only when ``alpha == 2``.
    """
    x_mat = np.atleast_2d(np.asarray(x_mat, dtype=float))
    y_mat = np.atleast_2d(np.asarray(y_mat, dtype=float))
    if x_mat.shape[0] == 0:
        x_mat = x_mat.reshape(0, 0)
    nx = x_mat.shape[1] if x_mat.size else 0
    ny = y_mat.shape[1] if y_mat.size else 0

    decomposed = GospaDecomposition()
    assignment = -np.ones(nx, dtype=int)
    dummy_cost = (c ** p) / alpha

    if nx == 0 and ny == 0:
        return 0.0, assignment, decomposed
    if nx == 0:
        decomposed.false = ny * dummy_cost
        return (ny * dummy_cost) ** (1.0 / p), assignment, decomposed
    if ny == 0:
        decomposed.missed = nx * dummy_cost
        return (nx * dummy_cost) ** (1.0 / p), assignment, decomposed

    # Capped base distances raised to power p.
    diff = x_mat[:, :, None] - y_mat[:, None, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=0))
    capped = np.minimum(dist, c) ** p  # (nx, ny)

    row_ind, col_ind = linear_sum_assignment(capped)

    cp = c ** p
    total = 0.0
    assigned_cols = set()
    for i, j in zip(row_ind, col_ind):
        assignment[i] = j
        assigned_cols.add(j)
        dij = capped[i, j]
        total += dij
        if dij < cp:
            decomposed.localisation += dij
        else:  # too far -> counts as a missed and a false detection
            decomposed.missed += dummy_cost
            decomposed.false += dummy_cost

    # Unassigned ground-truth targets -> missed.
    n_unassigned_x = nx - len(row_ind)
    decomposed.missed += n_unassigned_x * dummy_cost
    total += n_unassigned_x * dummy_cost

    # Unassigned estimates -> false.
    n_unassigned_y = ny - len(assigned_cols)
    decomposed.false += n_unassigned_y * dummy_cost
    total += n_unassigned_y * dummy_cost

    d_gospa = total ** (1.0 / p)
    return d_gospa, assignment, decomposed


def compute_gospa_error(x_estimate, x_truth, t_birth, t_death, c_gospa, k,
                        pos_idx=(0, 2)):
    """Squared GOSPA position error (alpha=2) and its decomposition at time ``k``.

    Port of ``ComputeGOSPAerror.m``.

    Parameters
    ----------
    x_estimate : ndarray (4 * n_est,)
        Stacked estimated states per target.
    x_truth : ndarray (4 * n_targets, Nsteps)
        Ground-truth states.
    t_birth, t_death : array_like
        Birth/death time step of each ground-truth target (1-based, as in the
        scenario definition).
    c_gospa : float
        GOSPA cut-off parameter.
    k : int
        Time step (1-based).
    pos_idx : tuple of int
        Row indices of the (x, y) position within each 4-element target block.
        Use ``(0, 2)`` for the Williams-15 ``[px, vx, py, vy]`` layout and
        ``(0, 1)`` for the ``[px, py, vx, vy]`` layout of scenario1MC.mat.
    """
    t_birth = np.asarray(t_birth)
    t_death = np.asarray(t_death)
    alive = (k >= t_birth) & (k < t_death)
    n_targets = t_birth.size
    pos_idx = list(pos_idx)

    truth_cols = []
    for i in range(n_targets):
        if alive[i]:
            block = x_truth[4 * i:4 * i + 4, k - 1]
            truth_cols.append(block[pos_idx])
    x_pos_truth = np.array(truth_cols).T if truth_cols else np.zeros((2, 0))

    x_estimate = np.asarray(x_estimate, dtype=float).ravel()
    n_est = x_estimate.size // 4
    if n_est > 0:
        est_full = x_estimate.reshape(n_est, 4).T  # (4, n_est)
        # Positions are at the rows given by pos_idx (default (0, 2) = px, py).
        # NB: the original ComputeGOSPAerror.m used rows [0, 1] = (px, vx) for
        # the Williams layout, which is an indexing bug; we use the true
        # positions instead.
        x_pos_est = est_full[pos_idx, :]
    else:
        x_pos_est = np.zeros((2, 0))

    d_gospa, _, decomp = gospa(x_pos_truth, x_pos_est, p=2, c=c_gospa, alpha=2)

    return d_gospa ** 2, decomp.localisation, decomp.missed, decomp.false
