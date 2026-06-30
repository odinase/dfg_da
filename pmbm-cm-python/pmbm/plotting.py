"""Plotting helpers. Port of ``DrawFilterEstimates.m``."""

import numpy as np


def draw_filter_estimates(x_truth, t_birth, t_death, x_estimate, xlim, ylim, z, ax=None):
    """Plot ground truth, current estimates and measurements at one time step.

    Parameters
    ----------
    x_truth : ndarray (4 * n_targets, Nsteps)
    t_birth, t_death : array_like (1-based time steps)
    x_estimate : ndarray (4 * n_est,)
        Stacked estimated states.
    xlim, ylim : (lo, hi) axis limits.
    z : ndarray (2, m) measurement set.
    ax : optional matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.grid(True)

    t_birth = np.asarray(t_birth)
    t_death = np.asarray(t_death)
    n_targets = x_truth.shape[0] // 4
    for i in range(n_targets):
        b = t_birth[i] - 1
        d = t_death[i] - 1
        xs = x_truth[4 * i + 0, b:d]
        ys = x_truth[4 * i + 2, b:d]
        ax.plot(xs, ys, "b", linewidth=1.2)
        if xs.size:
            ax.text(xs[0], ys[0], str(i + 1), color="b")

    x_estimate = np.asarray(x_estimate, dtype=float).ravel()
    n_est = x_estimate.size // 4
    if n_est > 0:
        est = x_estimate.reshape(n_est, 4)
        ax.plot(est[:, 0], est[:, 2], "or", fillstyle="none", label="estimate")

    z = np.asarray(z, dtype=float)
    if z.size:
        ax.plot(z[0, :], z[1, :], "ok", fillstyle="none", markersize=4, label="measurement")

    return ax
