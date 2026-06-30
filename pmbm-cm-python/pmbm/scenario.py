"""Simulation scenario: model parameters, ground-truth trajectories and
measurement generation.

Port of ``ScenarioWilliams15.m``, ``TrajectoryWilliams15.m`` and
``CreateMeasurement.m``. The single-target state is [x, vx, y, vy].
"""

from dataclasses import dataclass, field

import numpy as np


def _kron_block(diag2, block):
    return np.kron(np.eye(2), np.asarray(block, dtype=float)) if diag2 else None


@dataclass
class Scenario:
    """Container for the model parameters and the ground truth."""

    # Motion / measurement model
    Nx: int = 4
    T: float = 1.0
    F: np.ndarray = field(default=None)
    Q: np.ndarray = field(default=None)
    H: np.ndarray = field(default=None)
    R: np.ndarray = field(default=None)
    chol_R: np.ndarray = field(default=None)

    p_d: float = 0.90
    p_s: float = 0.99
    Area: tuple = (300.0, 300.0)
    Nsteps: int = 81
    l_clutter: float = 10.0
    intensity_clutter: float = None

    # Birth model (Poisson)
    weights_b: np.ndarray = field(default=None)
    means_b: np.ndarray = field(default=None)
    covs_b: list = field(default=None)
    lambda0: float = 3.0

    # Ground truth
    numtruth: int = 4
    X_truth: np.ndarray = field(default=None)
    t_birth: np.ndarray = field(default=None)
    t_death: np.ndarray = field(default=None)

    # Evaluation
    c_gospa: float = 10.0
    Nmc: int = 10


def williams15_trajectory(scenario_number, Nsteps, F, numtruth, Q, Area, rng):
    """Port of ``TrajectoryWilliams15.m``."""
    if scenario_number == 1:
        birthtime = np.zeros(numtruth)
        Pmid = 0.1 * np.eye(4)
        t_birth = np.ones(numtruth, dtype=int)
    else:
        birthtime = 10 * np.arange(numtruth)
        Pmid = 0.25 * np.eye(4)
        t_birth = np.arange(1, 10 * numtruth + 1, 10)

    t_death = np.concatenate(
        [[(Nsteps + 1) // 2], (Nsteps + 1) * np.ones(numtruth - 1, dtype=int)]
    ).astype(int)

    midpoint = (Nsteps + 1) // 2
    numfb = midpoint - 1
    chol_Q = np.linalg.cholesky(Q)  # lower triangular

    centre = np.array([Area[0] / 2, 0.0, Area[1] / 2, 0.0]).reshape(4, 1)
    x = np.linalg.cholesky(Pmid) @ rng.standard_normal((4, numtruth)) + centre

    xf = x.copy()
    xb = x.copy()
    X_truth = np.zeros((4 * numtruth, Nsteps))
    X_truth[:, midpoint - 1] = x.flatten(order="F")

    Finv = np.linalg.inv(F)
    for t in range(1, numfb + 1):
        xf = F @ xf + chol_Q @ rng.standard_normal((4, numtruth))
        xb = Finv @ (xb + chol_Q @ rng.standard_normal((4, numtruth)))
        X_truth[:, midpoint - 1 - t] = xb.flatten(order="F")
        X_truth[:, midpoint - 1 + t] = xf.flatten(order="F")

    # Zero out entries before birth / at and after death.
    for i in range(numtruth):
        block = X_truth[4 * i:4 * i + 4, :]
        block[:, : t_birth[i] - 1] = 0
        block[:, t_death[i] - 1:] = 0
        X_truth[4 * i:4 * i + 4, :] = block

    return X_truth, t_birth, t_death


def build_scenario(rng=None, seed=9):
    """Build the Williams-15 scenario (scenario number 1).

    Port of ``ScenarioWilliams15.m``.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    s = Scenario()
    s.T = 1.0
    s.F = np.kron(np.eye(2), np.array([[1.0, s.T], [0.0, 1.0]]))
    s.Q = 0.01 * np.kron(
        np.eye(2),
        np.array([[s.T ** 3 / 3, s.T ** 2 / 2], [s.T ** 2 / 2, s.T]]),
    )
    s.H = np.kron(np.eye(2), np.array([[1.0, 0.0]]))
    s.R = 1.0 * np.eye(2)
    s.chol_R = np.linalg.cholesky(s.R)  # lower triangular (= chol(R)' in MATLAB)

    s.p_d = 0.90
    s.p_s = 0.99
    s.Area = (300.0, 300.0)
    s.Nsteps = 81
    s.l_clutter = 10.0
    s.intensity_clutter = s.l_clutter / (s.Area[0] * s.Area[1])

    # Birth model
    s.weights_b = np.array([0.005])
    s.means_b = np.array([[100.0], [0.0], [100.0], [0.0]])
    P_ini = np.diag(np.array([150.0, 1.0, 150.0, 1.0]) ** 2)
    s.covs_b = [P_ini]
    s.lambda0 = 3.0

    s.numtruth = 4
    s.c_gospa = 10.0
    s.Nmc = 10

    s.X_truth, s.t_birth, s.t_death = williams15_trajectory(
        1, s.Nsteps, s.F, s.numtruth, s.Q, s.Area, rng
    )
    return s


def create_measurement(X_multi_k, t_birth, t_death, p_d, l_clutter, Area, k,
                       H, chol_R, Nx, rng):
    """Generate the measurement set at time ``k``.

    Port of ``CreateMeasurement.m``. ``X_multi_k`` is the column of
    ``X_truth`` at time ``k`` (length ``4 * n_targets``). Returns a (2, m)
    array of measurements (target detections followed by clutter).
    """
    t_birth = np.asarray(t_birth)
    t_death = np.asarray(t_death)
    index_targets = (t_birth <= k) & (t_death > k)

    detected = (rng.random(index_targets.size) < p_d) & index_targets
    n_detected = int(np.sum(detected))

    n_clutter = rng.poisson(l_clutter)

    X = np.asarray(X_multi_k, dtype=float).reshape(Nx, -1, order="F")
    Nz = H.shape[0]

    if n_detected > 0:
        z_targets = H @ X[:, detected] + chol_R @ rng.standard_normal((Nz, n_detected))
    else:
        z_targets = np.zeros((Nz, 0))

    if n_clutter > 0:
        z_clutter = np.vstack([
            Area[0] * rng.random(n_clutter),
            Area[1] * rng.random(n_clutter),
        ])
    else:
        z_clutter = np.zeros((2, 0))

    return np.hstack([z_targets, z_clutter])
