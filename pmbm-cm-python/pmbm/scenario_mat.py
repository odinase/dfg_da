"""Loader for the ``scenario1MC.mat`` benchmark dataset.

The dataset is the "9 ravens" long simulation scenario from

    E. F. Brekke and A. G. Hem, "A long simulation scenario for evaluation of
    multi-target tracking methods," Proc. ICECCME 2023.

It contains 1397 Cartesian radar scans produced by a moving ownship, 8 targets
moving in close formation, ground-truth trajectories and clutter. This module
converts the MATLAB structures (``params``, ``system``, ``scenario``) into the
quantities the ported PMBM filter consumes.

The single-target state here is ``[px, py, vx, vy]`` (positions at indices 0 and
1) — note this differs from the Williams-15 scenario in ``scenario.py`` which
uses ``[px, vx, py, vy]``.

The data extraction mirrors the "Angel" (reference-filter) branch of
``pmbm-cm-matlab/script_pmbm91.m`` (lines ~700-895): the model comes straight
from ``system``; the clutter intensity is ``faRate / areaCircle``; and the PPP
birth is a single wide Gaussian on a disk of radius ``rMax`` centred on the
ownship position at each step (the "flat-P" approximation).
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.io as sio


@dataclass
class MatScenario:
    """Scenario parameters and data loaded from ``scenario1MC.mat``."""

    F: np.ndarray
    Q: np.ndarray
    H: np.ndarray
    R: np.ndarray
    p_d: float
    p_s: float
    intensity_clutter: float
    Nsteps: int
    Nx: int
    c_gospa: float
    pos_idx: tuple

    X_truth: np.ndarray
    t_birth: np.ndarray
    t_death: np.ndarray
    ownship: np.ndarray  # (>=2, Nsteps), rows 0,1 are ownship x,y position

    # Internal: measurement layout and birth-model parameters.
    _zList: np.ndarray = field(repr=False, default=None)
    _zBegs: np.ndarray = field(repr=False, default=None)
    _zEnds: np.ndarray = field(repr=False, default=None)
    _rMax: float = field(repr=False, default=100.0)
    _pInitVel: float = field(repr=False, default=900.0)
    _areaCircle: float = field(repr=False, default=1.0)
    _birthRate: np.ndarray = field(repr=False, default=None)

    def measurements_at(self, k):
        """Measurement set at time step ``k`` (1-based), shape (2, m)."""
        b = self._zBegs[k - 1]
        e = self._zEnds[k - 1]
        return self._zList[:, b:e]

    def birth_at(self, k):
        """PPP birth (weights_b, means_b, covs_b) at time step ``k`` (1-based).

        Mirrors ``script_pmbm91.m`` lines 715-727 / 888-895: a single Gaussian
        on a disk of radius ``rMax`` centred on the ownship position, with a
        weight set so its 2D peak value equals the uniform birth intensity.
        """
        ox = self.ownship[0, k - 1]
        oy = self.ownship[1, k - 1]
        x_min, x_max = ox - self._rMax, ox + self._rMax
        y_min, y_max = oy - self._rMax, oy + self._rMax

        means_b = np.array([[(x_max + x_min) / 2.0],
                            [(y_max + y_min) / 2.0],
                            [0.0],
                            [0.0]])
        P_ini = np.diag(4.0 ** 2 * np.array([
            (x_max - x_min) ** 2 / 12.0,
            4.0 ** 2 * (y_max - y_min) ** 2 / 12.0,
            self._pInitVel,
            self._pInitVel,
        ]))
        covs_b = [P_ini]

        lambda0 = self._birthRate[k - 1] / self._areaCircle
        gauss_peak_2d = 1.0 / (np.sqrt(np.linalg.det(P_ini[:2, :2])) * (2.0 * np.pi))
        weights_b = np.array([lambda0 / gauss_peak_2d])

        return weights_b, means_b, covs_b

    def initial_filter(self):
        """Initial (pre-first-update) filter state seeded from the birth model."""
        weights_b, means_b, covs_b = self.birth_at(1)
        return {
            "weightPois": [float(weights_b[0])],
            "meanPois": [means_b[:, 0].astype(float).copy()],
            "covPois": [covs_b[0].astype(float).copy()],
            "tracks": [],
            "globHyp": np.zeros((0, 0), dtype=int),
            "globHypWeight": np.zeros(0),
        }


def load_mat_scenario(path, pd_value=0.9):
    """Load ``scenario1MC.mat`` and return a :class:`MatScenario`.

    Parameters
    ----------
    path : str
        Path to ``scenario1MC.mat``.
    pd_value : float
        Detection probability to use (the dataset ships ``PDList = [0.5, 0.9]``;
        the moving-sensor configuration in the MATLAB script selects 0.9).
    """
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    params = m["params"]
    system = m["system"]
    scenario = m["scenario"]

    F = np.asarray(system.fMat, dtype=float)
    Q = np.asarray(system.qMat, dtype=float)
    H = np.asarray(system.hMat, dtype=float)
    R = np.asarray(system.rCart, dtype=float)
    Nx = F.shape[0]

    area_circle = float(params.areaCircle)
    fa_rate = float(params.faRate)
    intensity_clutter = fa_rate / area_circle

    p_s = float(params.pS)

    # Measurement layout: cumulative offsets from per-step cardinalities.
    zList = np.asarray(scenario.zList, dtype=float)
    if zList.ndim == 1:
        zList = zList.reshape(2, -1)
    zCard = np.asarray(scenario.zCard, dtype=int).ravel()
    zEnds = np.cumsum(zCard)
    zBegs = zEnds - zCard
    Nsteps = zCard.size

    # Ground truth from the per-target structs.
    targets = np.atleast_1d(scenario.targetsTrue)
    numtruth = targets.size
    X_truth = np.zeros((4 * numtruth, Nsteps))
    t_birth = np.zeros(numtruth, dtype=int)
    t_death = np.zeros(numtruth, dtype=int)
    for tt in range(numtruth):
        tgt = targets[tt]
        kList = np.asarray(tgt.k, dtype=int).ravel()
        xMat = np.asarray(tgt.x, dtype=float)
        if xMat.ndim == 1:
            xMat = xMat.reshape(4, -1)
        for ci, kk in enumerate(kList):
            X_truth[4 * tt:4 * tt + 4, kk - 1] = xMat[:, ci]
        t_birth[tt] = kList[0]
        t_death[tt] = kList[-1] + 1

    ownship = np.asarray(params.stateFullOwn, dtype=float)

    p_init_vel = float(np.atleast_2d(np.asarray(params.pInitVel, dtype=float))[0, 0])
    birth_rate = np.asarray(params.birthRateHistory, dtype=float).ravel()

    return MatScenario(
        F=F, Q=Q, H=H, R=R,
        p_d=float(pd_value), p_s=p_s,
        intensity_clutter=intensity_clutter,
        Nsteps=Nsteps, Nx=Nx, c_gospa=10.0, pos_idx=(0, 1),
        X_truth=X_truth, t_birth=t_birth, t_death=t_death, ownship=ownship,
        _zList=zList, _zBegs=zBegs, _zEnds=zEnds,
        _rMax=float(params.rMax), _pInitVel=p_init_vel,
        _areaCircle=area_circle, _birthRate=birth_rate,
    )
