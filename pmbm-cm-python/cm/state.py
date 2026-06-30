"""Persistent CM-filter state and its empty initialisation.

Port of the state set up in ``script_pmbm91.m:567-613``. At k=1 everything is
empty / zero.
"""

from dataclasses import dataclass, field

import numpy as np

from .columns import MAX_LAG, InCol


@dataclass
class CMState:
    incol: InCol
    max_lag: int = MAX_LAG

    track_file: np.ndarray = None        # (incol.last, nTracks)
    track_file_shadow: np.ndarray = None  # (incol.last, nTracks, maxLag)
    mea_hist_col: np.ndarray = None      # (maxLag, nTracks)

    clusters: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    clusters_card: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    hypos: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    hypos_card: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    prob_log_hypos: np.ndarray = field(default_factory=lambda: np.zeros(0))

    phd_tracks: np.ndarray = None        # (incol.last, nTPHD)
    muPPP: float = 0.0
    label_gen: int = 0


def initial_state(incol=None, max_lag=MAX_LAG):
    incol = incol or InCol()
    n = incol.last
    return CMState(
        incol=incol,
        max_lag=max_lag,
        track_file=np.zeros((n, 0)),
        track_file_shadow=np.zeros((n, 0, max_lag)),
        mea_hist_col=np.zeros((max_lag, 0)),
        phd_tracks=np.zeros((n, 0)),
        muPPP=0.0,
        label_gen=0,
    )
