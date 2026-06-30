"""Cardinality distribution of a multi-Bernoulli RFS.

Port of ``CardinalityMB.m``.
"""

import numpy as np


def cardinality_mb(r):
    """Cardinality distribution of a multi-Bernoulli RFS via FFT.

    Parameters
    ----------
    r : array_like
        Existence probabilities of the Bernoulli components.

    Returns
    -------
    pcard : ndarray, shape (N + 1,)
        ``pcard[n]`` is the probability of cardinality ``n`` (n = 0..N).
    """
    r = np.asarray(r, dtype=float).ravel()
    N = r.size
    if N == 0:
        return np.array([1.0])

    exp_omega = np.exp(-1j * np.arange(N + 1) / (N + 1) * 2 * np.pi)
    F = np.ones(N + 1, dtype=complex)
    for i in range(N):
        F = F * ((1.0 - r[i]) + r[i] * exp_omega)
    pcard = np.real(np.fft.ifft(F))
    return pcard
