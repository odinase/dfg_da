"""Load ``scenario1MC.mat`` into the raw ``system`` / ``params`` / ``scenario``
dicts the CM pipeline expects (mirroring the MATLAB variable names).
"""

import numpy as np
import scipy.io as sio


class MatScenarioCM:
    """Holds the raw structs plus per-step measurement slicing."""

    def __init__(self, system, params, scenario, pd_value):
        self.system = system
        self.params = params
        self.scenario = scenario
        self.pD = float(pd_value)

        self.zList = np.asarray(scenario["zList"], dtype=float)
        if self.zList.ndim == 1:
            self.zList = self.zList.reshape(2, -1)
        zCard = np.asarray(scenario["zCard"], dtype=int).ravel()
        self.zCard = zCard
        self.zEnds = np.cumsum(zCard)
        self.zBegs = self.zEnds - zCard
        self.Nsteps = zCard.size

        self.hTrue = np.asarray(scenario["hTrue"], dtype=float)

    def measurements_at(self, k):
        b = self.zBegs[k - 1]
        e = self.zEnds[k - 1]
        return self.zList[:, b:e]

    def mu_birth(self, k):
        area = float(self.params["areaCircle"])
        rate = np.asarray(self.params["birthRateHistory"], dtype=float).ravel()
        return rate[k - 1] / area

    @property
    def lambda_fa(self):
        return float(self.params["faRate"]) / float(self.params["areaCircle"])


def load_cm_scenario(path, pd_value=0.9):
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    sys_s = m["system"]
    par_s = m["params"]
    scn_s = m["scenario"]

    system = {
        "fMat": np.asarray(sys_s.fMat, dtype=float),
        "qMat": np.asarray(sys_s.qMat, dtype=float),
        "hMat": np.asarray(sys_s.hMat, dtype=float),
        "rCart": np.asarray(sys_s.rCart, dtype=float),
        "rPol": np.asarray(sys_s.rPol, dtype=float),
    }
    params = {
        "areaCircle": float(par_s.areaCircle),
        "faRate": float(par_s.faRate),
        "pS": float(par_s.pS),
        "rMax": float(par_s.rMax),
        "lambdaFa": float(par_s.faRate) / float(par_s.areaCircle),
        "birthRateHistory": np.asarray(par_s.birthRateHistory, dtype=float).ravel(),
        "stateFullOwn": np.asarray(par_s.stateFullOwn, dtype=float),
        "pInitVel": np.atleast_2d(np.asarray(par_s.pInitVel, dtype=float)),
        "PDList": np.asarray(par_s.PDList, dtype=float).ravel(),
    }
    scenario = {
        "zList": np.asarray(scn_s.zList, dtype=float),
        "zCard": np.asarray(scn_s.zCard, dtype=int).ravel(),
        "hTrue": np.asarray(scn_s.hTrue, dtype=float),
    }
    return MatScenarioCM(system, params, scenario, pd_value)
