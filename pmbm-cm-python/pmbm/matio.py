"""Helpers for writing PMBM filter state to MATLAB ``.mat`` files.

The reference MATLAB code dumps per-time-step state in
``script_pmbm91.m:1236`` via ``save('priorLikelihood{k}.mat', ...)``. Those
variables (``hypos``, ``clusters``, ``clustersCard`` …) belong to the
cluster-management filter architecture and have no counterpart in the ported
reference filter. This module dumps the closest analog: the reference filter's
multi-Bernoulli-mixture state (global hypotheses + tracks) together with the
current measurements and estimate.
"""

import os
from datetime import datetime

import numpy as np
import scipy.io as sio


def make_run_dir(base="runs"):
    """Create and return a fresh per-run output directory.

    The directory is ``{base}/run_YYYYMMDD_HHMMSS`` (a numeric suffix is added
    if that name already exists), so dumps from a run never clutter an existing
    folder and are easy to locate.
    """
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    path = os.path.join(base, stamp)
    if os.path.exists(path):
        i = 1
        while os.path.exists(f"{path}_{i}"):
            i += 1
        path = f"{path}_{i}"
    os.makedirs(path)
    return path


def _obj_array(items):
    """Pack a list of ndarrays into a 1xN object array for ``savemat``.

    ``None`` entries (e.g. ungated single-target hypotheses that were never
    populated) become empty arrays, which MATLAB reads as ``[]``.
    """
    arr = np.empty((1, len(items)), dtype=object)
    for i, it in enumerate(items):
        arr[0, i] = np.zeros((0, 0)) if it is None else np.asarray(it, dtype=float)
    return arr


def _struct_array(dicts):
    """Pack a list of dicts into a 1xN object array (a MATLAB struct array)."""
    arr = np.empty((1, len(dicts)), dtype=object)
    for i, d in enumerate(dicts):
        arr[0, i] = d
    return arr


def _col(v):
    return None if v is None else np.asarray(v, dtype=float).reshape(-1, 1)


def _row(v):
    return None if v is None else np.asarray(v, dtype=float).reshape(1, -1)


def _track_to_struct(track):
    """Convert one track dict into a savemat-friendly dict (MATLAB struct)."""
    return {
        "meanB": _obj_array([_col(v) for v in track["meanB"]]),
        "covB": _obj_array(list(track["covB"])),
        "eB": np.asarray(track["eB"], dtype=float).reshape(1, -1),
        "aHis": _obj_array([_row(a) for a in track["aHis"]]),
        "weightBLog": np.asarray(track["weightBLog"], dtype=float).reshape(1, -1),
        "t_ini": float(track["t_ini"]),
    }


def filter_state_to_matdict(flt, k, measurements, X_estimate, pd, ownship_k=None):
    """Build a dict of MATLAB-friendly variables describing the filter state."""
    tracks = flt.get("tracks", [])
    track_structs = _struct_array([_track_to_struct(t) for t in tracks])

    glob_hyp = np.asarray(flt.get("globHyp", np.zeros((0, 0))), dtype=float)
    glob_w = np.asarray(flt.get("globHypWeight", np.zeros(0)), dtype=float).reshape(1, -1)

    out = {
        "k": float(k),
        "pD": float(pd),
        "measurements": np.asarray(measurements, dtype=float),
        "X_estimate": np.asarray(X_estimate, dtype=float).reshape(-1, 1),
        # globHyp uses 0-based hypothesis indices with -1 = absent (MATLAB used
        # 1-based with 0 = absent); kept as-is from the Python filter.
        "globHyp": glob_hyp,
        "globHypWeight": glob_w,
        "tracks": track_structs,
        # PPP (Poisson) component of the intensity.
        "weightPois": np.asarray(flt.get("weightPois", []), dtype=float).reshape(1, -1),
        "meanPois": _obj_array([_col(v) for v in flt.get("meanPois", [])]),
        "covPois": _obj_array(list(flt.get("covPois", []))),
    }
    if ownship_k is not None:
        out["ownship"] = np.asarray(ownship_k, dtype=float).reshape(-1, 1)
    return out


def save_priorlikelihood(flt, k, outdir, measurements, X_estimate, pd, ownship_k=None):
    """Write ``{outdir}/priorLikelihood{k}.mat`` for the given time step."""
    matdict = filter_state_to_matdict(flt, k, measurements, X_estimate, pd, ownship_k)
    path = os.path.join(outdir, f"priorLikelihood{k}.mat")
    sio.savemat(path, matdict, do_compression=True)
    return path
