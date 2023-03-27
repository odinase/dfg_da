import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import scipy.linalg as la
from .stats_logger import TrackEstimate, PredictedMeasurement, MatFileParser
from typing import List


def ellipse(mu, P, s, n):
    thetas = np.linspace(0, 2*np.pi, n)
    ell = mu + s * (la.cholesky(P).T @ np.array([np.cos(thetas), np.sin(thetas)])).T
    return ell


def draw_estimate(x: np.ndarray, P: np.ndarray, ax: plt.Axes, c = None, label = "_"):
    ell = ellipse(x[:2], P[:2, :2], 3.0, 200)
    if not (c is None):
        ax.plot(*x, 'x', color=c, label=label, ms=10)
        ax.plot(*ell.T, color=c)
    else:
        ax.plot(*x, label=label)
        ax.plot(*ell.T)
