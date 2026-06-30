"""Unit checks for the CM PMBM port's helper functions."""

import numpy as np

from cm.clouds import (
    a2bc,
    bc2a,
    pick_ind_c,
    tcloud_to_beg,
    tcloud_to_end,
    union_sorted,
)
from cm.columns import InCol
from cm.mathutils import (
    c2p,
    cm_alternative,
    cov_mat_to_vec,
    cov_vec_to_mat,
    find_colmajor,
    gm_reduce,
    normpdf_log,
    v2m,
)


def test_cov_roundtrip():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    P = A @ A.T + np.eye(4)
    vec = cov_mat_to_vec(P)
    assert vec.shape == (10, 1)
    P2 = cov_vec_to_mat(vec[:, 0])
    assert np.allclose(P, P2)


def test_cov_vec_layout():
    # Diagonal first, then strict lower triangle in column order.
    P = np.array([[1., 2., 3., 4.],
                  [2., 5., 6., 7.],
                  [3., 6., 8., 9.],
                  [4., 7., 9., 10.]])
    vec = cov_mat_to_vec(P)[:, 0]
    assert np.allclose(vec[:4], [1, 5, 8, 10])           # diagonal
    assert np.allclose(vec[4:], [2, 3, 4, 6, 7, 9])      # (2,1)(3,1)(4,1)(3,2)(4,2)(4,3)


def test_v2m_and_find_colmajor():
    # 3x2 matrix, MATLAB column-major linear indices.
    M = np.array([[0, 0], [1, 0], [0, 1]])
    idx = find_colmajor(M != 0)
    # nonzero at (2,1)->2 and (3,2)->6 (1-based col-major)
    assert list(idx) == [2, 6]
    subs = v2m(idx, 3)
    assert subs[0, 0] == 2 and subs[1, 0] == 1
    assert subs[0, 1] == 3 and subs[1, 1] == 2


def test_normpdf_log_matches_scipy():
    from scipy.stats import multivariate_normal
    mu = np.array([1.0, -2.0])
    S = np.array([[2.0, 0.3], [0.3, 1.0]])
    x = np.array([0.5, -1.0])
    got = normpdf_log(x.reshape(2, 1), mu.reshape(2, 1), S)[0]
    exp = multivariate_normal(mean=mu, cov=S).logpdf(x)
    assert abs(got - exp) < 1e-10


def test_c2p_quadrants():
    out = c2p(np.array([[1.0, -1.0, 0.0], [1.0, 1.0, -2.0]]))
    assert abs(out[0, 0] - np.sqrt(2)) < 1e-12
    assert abs(out[1, 0] - np.pi / 4) < 1e-12          # +x +y
    assert abs(out[1, 1] - 3 * np.pi / 4) < 1e-12      # -x +y
    assert abs(out[1, 2] - 3 * np.pi / 2) < 1e-12      # x=0, y<0 -> -pi/2 mod 2pi


def test_cm_alternative_symmetric():
    polar = np.array([100.0, 0.7])
    R = np.diag([2.0 ** 2, 0.0035 ** 2])
    xa, Ra = cm_alternative(polar, R)
    assert np.allclose(Ra, Ra.T)
    assert abs(xa[0] - 100 * np.cos(0.7)) < 1e-9
    assert abs(xa[1] - 100 * np.sin(0.7)) < 1e-9


def test_gm_reduce_single():
    eta = np.array([[1.0], [2.0]])
    cov = np.eye(2)[:, :, None]
    m, P = gm_reduce(eta, cov, np.array([1.0]))
    assert np.allclose(m, [1.0, 2.0])
    assert np.allclose(P, np.eye(2))


def test_gm_reduce_moments():
    eta = np.array([[0.0, 2.0]])           # 1-D states at 0 and 2
    cov = np.zeros((1, 1, 2))
    m, P = gm_reduce(eta, cov, np.array([0.5, 0.5]))
    assert abs(m[0] - 1.0) < 1e-12
    assert abs(P[0, 0] - 1.0) < 1e-12      # variance = E[x^2]-E[x]^2 = 2-1


def test_tcloud_helpers():
    card = np.array([2, 3, 1])
    assert list(tcloud_to_beg(card)) == [1, 3, 6]
    assert list(tcloud_to_end(card)) == [2, 5, 6]
    # pick_ind_c of cluster 2 -> A-level indices 3,4,5
    assert list(pick_ind_c([2], card)) == [3, 4, 5]


def test_a2bc_bc2a_roundtrip():
    card = np.array([2, 3, 1])
    a = np.array([1, 3, 5, 6])
    b, c = a2bc(a, card)
    a2 = bc2a(b, c, card)
    assert list(a2) == list(a)


def test_union_sorted():
    c, am, bm = union_sorted([20, 30, 40], [10, 20, 30])
    assert list(c) == [10, 20, 30, 40]
    assert list(c[am - 1]) == [20, 30, 40]
    assert list(c[bm - 1]) == [10, 20, 30]


def test_incol_layout():
    ic = InCol()
    assert ic.last == 21
    assert ic.meaLast == 14   # 0-based
    s = ic.to_matlab_struct()
    assert s["last"] == 21.0 and s["meaLast"] == 15.0
    assert list(s["tarX"].ravel()) == [1, 2, 3, 4]
    assert list(s["tarP"].ravel()) == list(range(5, 15))


if __name__ == "__main__":
    import sys
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
