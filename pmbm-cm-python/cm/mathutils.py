"""Math / linear-algebra helpers for the CM PMBM port.

Faithful ports of the small MATLAB utilities used by the prior/likelihood
pipeline:

* ``covMat2Vec.m`` / ``covVec2Mat.m`` — symmetric-covariance (de)vectorisation.
* ``at612/jmpdFunctions/v2m.m`` — linear index -> 2D subscript.
* ``at612/jmpdFunctions/normpdfLog.m`` — log Gaussian density.
* ``at612/jmpdFunctions/c2p.m`` — Cartesian -> polar.
* ``cmAlternative.m`` — debiased polar->Cartesian measurement-covariance.
* ``gmReduce.m`` — Gaussian-mixture moment-matching reduction.

All index helpers preserve MATLAB's 1-based, column-major semantics where the
indices are *data* (so the dumped ``.mat`` matches).
"""

import numpy as np
from scipy.linalg import block_diag  # noqa: F401  (re-exported for callers)


def cov_mat_to_vec(cov):
    """Port of ``covMat2Vec.m``. ``cov`` is (d, d) or (d, d, n).

    Returns (n_cov, n) where n_cov = d + d(d-1)/2: the diagonal followed by the
    strict lower triangle in column order [(2,1),(3,1),(4,1),(3,2),(4,2),(4,3)].
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 2:
        cov = cov[:, :, None]
    d = cov.shape[0]
    n = cov.shape[2]
    diag_part = np.zeros((d, n))
    lower_part = np.zeros((d * (d - 1) // 2, n))
    for ii in range(n):
        diag_part[:, ii] = np.diag(cov[:, :, ii])
        start = 0
        for kk in range(d - 1):
            seg = cov[kk + 1:d, kk, ii]
            lower_part[start:start + seg.size, ii] = seg
            start += seg.size
    return np.vstack([diag_part, lower_part])


def cov_vec_to_mat(vec):
    """Port of ``covVec2Mat.m``. ``vec`` is (n_cov,) or (n_cov, n).

    Returns (d, d, n) symmetric matrices (or (d, d) if a single vector given).
    """
    vec = np.asarray(vec, dtype=float)
    single = vec.ndim == 1
    if single:
        vec = vec[:, None]
    n_cov, n = vec.shape
    d = int((np.sqrt(8 * n_cov + 1) - 1) / 2)
    out = np.zeros((d, d, n))
    for ii in range(n):
        m = np.zeros((d, d))
        start = d
        for kk in range(d - 1):
            length = d - kk - 1
            m[kk + 1:d, kk] = vec[start:start + length, ii]
            start += length
        m = m + m.T + np.diag(vec[:d, ii])
        out[:, :, ii] = m
    return out[:, :, 0] if single else out


def v2m(candvec, xmax):
    """Port of ``v2m.m``: 1-based column-major linear index -> [row; col].

    ``candvec`` is a 1-D array of 1-based linear indices; ``xmax`` is the number
    of rows. Returns a (2, n) array of 1-based [row; col] subscripts.
    """
    candvec = np.asarray(candvec, dtype=int).ravel()
    out = np.zeros((2, candvec.size), dtype=int)
    out[0, :] = np.mod(candvec - 1, xmax) + 1
    out[1, :] = np.ceil(candvec / xmax).astype(int)
    return out


def m2v(subs, siz):
    """Port of ``m2v.m``: 1-based [row; col] subscripts -> column-major linear
    index. Inverse of :func:`v2m`.

    ``subs`` is a (2, n) array of 1-based [row; col] subscripts; ``siz`` is the
    matrix shape ``(nrows, ncols)``. Returns a 1-D array of 1-based linear
    indices (``sub2ind`` semantics).
    """
    subs = np.asarray(subs, dtype=int).reshape(2, -1)
    nrows = int(siz[0])
    return (subs[1, :] - 1) * nrows + subs[0, :]


def find_colmajor(mask):
    """Return MATLAB ``find`` for a 2-D boolean/array: 1-based column-major
    linear indices of the true/non-zero entries, in ascending order."""
    mask = np.asarray(mask)
    # Column-major flatten (order='F'); +1 for 1-based.
    flat = mask.flatten(order="F")
    return np.nonzero(flat)[0] + 1


def normpdf_log(x, mu, sigma, sigma_inv=None):
    """Port of ``normpdfLog.m``: log of the multivariate normal density.

    Either ``x`` or ``mu`` may be a single column broadcast against the other.
    Returns a 1-D array of length max(#cols(x), #cols(mu)).
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    mu = np.atleast_2d(np.asarray(mu, dtype=float))
    if x.shape[0] == 1 and x.shape[1] != 1:  # passed a row by accident
        x = x.T
    if mu.shape[0] == 1 and mu.shape[1] != 1:
        mu = mu.T
    sigma = np.asarray(sigma, dtype=float)
    p = x.shape[0]
    mydet = np.linalg.det(sigma)

    if x.shape[1] > mu.shape[1] and mu.shape[1] == 1:
        x_arr = x
        mu_arr = np.tile(mu, (1, x.shape[1]))
    elif mu.shape[1] > x.shape[1] and x.shape[1] == 1:
        mu_arr = mu
        x_arr = np.tile(x, (1, mu.shape[1]))
    else:
        mu_arr = mu
        x_arr = x

    log_val = -np.log((2 * np.pi) ** (p / 2) * np.sqrt(mydet))
    nu = x_arr - mu_arr
    if sigma_inv is None:
        sol = np.linalg.solve(sigma, nu)
    else:
        sol = np.asarray(sigma_inv, dtype=float) @ nu
    exp_val = -np.sum(sol * nu, axis=0) / 2
    return log_val + exp_val


def c2p(cartesian):
    """Port of ``c2p.m``: Cartesian -> polar [r; theta], theta in [0, 2*pi)."""
    cart = np.atleast_2d(np.asarray(cartesian, dtype=float))
    if cart.shape[0] == 1:
        cart = cart.reshape(2, -1)
    r = np.sqrt(cart[0, :] ** 2 + cart[1, :] ** 2)
    theta = np.zeros_like(r)
    for i in range(cart.shape[1]):
        x, y = cart[0, i], cart[1, i]
        if x > 0:
            theta[i] = np.arctan(y / x)
        elif x < 0:
            theta[i] = np.arctan(y / x) + np.pi
        else:
            if y < 0:
                theta[i] = -np.pi / 2
            elif y > 0:
                theta[i] = np.pi / 2
            else:
                theta[i] = 0.0
    out = np.zeros((2, cart.shape[1]))
    out[0, :] = r
    out[1, :] = np.mod(theta, 2 * np.pi)
    return out


def cm_alternative(polar, R):
    """Port of ``cmAlternative.m``: Bar-Shalom debiased polar->Cartesian.

    Returns (xa, Ra): converted Cartesian measurement and its covariance.
    """
    polar = np.asarray(polar, dtype=float).ravel()
    R = np.asarray(R, dtype=float)
    rm, thetam = polar[0], polar[1]
    R11 = rm ** 2 * R[1, 1] * np.sin(thetam) ** 2 + R[0, 0] * np.cos(thetam) ** 2
    R22 = rm ** 2 * R[1, 1] * np.cos(thetam) ** 2 + R[0, 0] * np.sin(thetam) ** 2
    R12 = (R[0, 0] - rm ** 2 * R[1, 1]) * np.sin(thetam) * np.cos(thetam)
    Ra = np.array([[R11, R12], [R12, R22]])
    xa = np.array([rm * np.cos(thetam), rm * np.sin(thetam)])
    return xa, Ra


def gm_reduce(eta_list, cov_list, weights):
    """Port of ``gmReduce.m``: moment-matching of a Gaussian mixture.

    ``eta_list`` is (dim, n), ``cov_list`` is (dim, dim, n), ``weights`` is
    length n (will be normalised). Returns (eta_ave, cov_ave).
    """
    eta_list = np.atleast_2d(np.asarray(eta_list, dtype=float))
    cov_list = np.asarray(cov_list, dtype=float)
    if cov_list.ndim == 2:
        cov_list = cov_list[:, :, None]
    weights = np.asarray(weights, dtype=float).ravel()
    n = eta_list.shape[1]
    dim = eta_list.shape[0]
    weights = weights / np.sum(weights)
    active_weight = np.sum(weights[:n])
    eta_ave = eta_list @ weights[:n] / active_weight
    cov_ave = np.zeros((dim, dim))
    for ii in range(n):
        nu = eta_list[:, ii] - eta_ave
        cov_ave += weights[ii] * np.outer(nu, nu) + weights[ii] * cov_list[:, :, ii]
    return eta_ave, cov_ave
