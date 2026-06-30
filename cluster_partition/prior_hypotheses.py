"""
prior_hypotheses.py
===================

Pure-Python re-implementation of the prior-hypothesis containers used by
``odinase/dfg_da`` (in the repo these live in the compiled ``py_dfg_da``
pybind module as ``pdd.hypothesis.Hypotheses`` / ``HypothesesList``).

A *cluster* carries a discrete distribution over **prior hypotheses**.  Each
prior hypothesis ``theta`` is the subset of (globally indexed, 1-based) tracks
that *exist* under that hypothesis, together with its prior probability
``Pr{theta | Z_{1:k-1}}`` (the factor ``phi(theta)`` in thesis Eq. (5.12)).

The single-cluster, multi-hypothesis solvers consume these objects through the
following contract, identical to the upstream repo:

    prior_hypotheses.tracks()                  -> iterable of global track ids
    prior_hypotheses.t_idxs()                  -> sorted 0-based row indices
    prior_hypotheses.reindex_tracks()          -> relabel tracks to 1..n_c
    prior_hypotheses.hypothesis_probabilites() -> np.ndarray of priors
    for h in prior_hypotheses:                 -> Hypothesis objects with
        h.tracks(), h.probability()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Sequence, Set

import numpy as np


@dataclass
class Hypothesis:
    """A single prior hypothesis: the set of existing tracks + its prior prob."""

    _tracks: List[int]
    _log_prob: float

    def tracks(self) -> List[int]:
        return self._tracks

    def probability(self) -> float:
        return float(np.exp(self._log_prob))

    def log_probability(self) -> float:
        return float(self._log_prob)


class Hypotheses:
    """Distribution over prior hypotheses for a *single* cluster."""

    def __init__(self, hypotheses: Sequence[Hypothesis]):
        self._hypotheses: List[Hypothesis] = list(hypotheses)
        # Map used by reindex_tracks(); identity until reindex_tracks is called.
        self._reindex_map: dict | None = None

    # -- iteration -----------------------------------------------------------
    def __iter__(self) -> Iterator[Hypothesis]:
        return iter(self._hypotheses)

    def __len__(self) -> int:
        return len(self._hypotheses)

    # -- track bookkeeping ---------------------------------------------------
    def tracks(self) -> Set[int]:
        """Union of all track ids appearing in any hypothesis (current labels)."""
        out: Set[int] = set()
        for h in self._hypotheses:
            out.update(h.tracks())
        return out

    def t_idxs(self) -> np.ndarray:
        """Sorted 0-based row indices into the *global* reward matrix.

        Returned *before* reindexing, these are ``global_track_id - 1``.
        """
        return np.sort(np.fromiter(self.tracks(), dtype=int)) - 1

    def reindex_tracks(self) -> None:
        """Relabel the track ids to a dense ``1..n_c`` range.

        After taking ``R_cluster = R_LC[t_idxs]`` the cluster's reward sub-matrix
        has rows ``0..n_c-1``; reindexing makes the hypotheses refer to those
        rows as ``1..n_c`` (1-based), exactly as the upstream solvers expect.
        Idempotent: a second call is a no-op.
        """
        current = sorted(self.tracks())
        if current == list(range(1, len(current) + 1)):
            return  # already dense 1..n_c
        remap = {old: new for new, old in enumerate(current, start=1)}
        for h in self._hypotheses:
            h._tracks = [remap[t] for t in h._tracks]
        self._reindex_map = remap

    def hypothesis_probabilites(self) -> np.ndarray:  # (sic) upstream spelling
        return np.array([h.probability() for h in self._hypotheses], dtype=float)

    def copy(self) -> "Hypotheses":
        return Hypotheses(
            [Hypothesis(list(h.tracks()), h.log_probability()) for h in self._hypotheses]
        )


class HypothesesList(list):
    """List of per-cluster :class:`Hypotheses` (indexed by cluster id)."""

    pass


def make_hypotheses(hyps: Iterable[tuple]) -> Hypotheses:
    """Convenience builder.

    Each element is ``(tracks, prob)`` with ``prob`` a *linear* probability.
    """
    out = []
    for tracks, prob in hyps:
        out.append(Hypothesis(list(map(int, tracks)), float(np.log(prob))))
    return Hypotheses(out)
