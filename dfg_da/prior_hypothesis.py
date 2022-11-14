import numpy as np
from dataclasses import dataclass
from typing import List
from scipy.special import logsumexp


def is_int(t) -> bool:
    return (
        isinstance(t, int) or
        isinstance(t, np.int64) or
        isinstance(t, np.uint64) or
        isinstance(t, np.int64) or
        isinstance(t, np.uint32) or
        isinstance(t, np.int32) or
        isinstance(t, np.uint16) or
        isinstance(t, np.int8) or
        isinstance(t, np.uint8)        
    )


@dataclass
class PriorHypothesis:
    tracks: np.ndarray
    log_prob: float

    def __post_init__(self):
        assert all(is_int(t) and t > 0 for t in self.tracks)

    @property
    def probability(self) -> float:
        """
        Assumes that the log prob is properly normalized.
        """
        p = np.exp(self.log_prob)
        assert 0.0 <= p <= 1.0 + 1e-6 # Apparently we sometimes get just over 1.0 due to numeric inprecision?
        return p


@dataclass
class PriorHypothesisIter:
    prior_hypotheses: List[PriorHypothesis]
    __index: int = 0

    def __next__(self):
        if self.__index < len(self.prior_hypotheses):
            ph = self.prior_hypotheses[self.__index]
            self.__index += 1
            return ph.tracks, ph.probability

        raise StopIteration


@dataclass
class PriorHypotheses:
    prior_hypotheses: List[PriorHypothesis]

    def __post__init__(self):
        lognorm = logsumexp([h.log_prob for h in self.prior_hypotheses])
        for k in len(self.prior_hypotheses):
            self.prior_hypotheses[k].log_prob -= lognorm
            assert 0 <= self.prior_hypotheses[k].probability <= 1.0

    def hypothesis_probabilities(self) -> np.ndarray:
        return np.array([h.probability for h in self.prior_hypotheses])

    @property
    def num_hypotheses(self) -> int:
        return len(self.prior_hypotheses)

    def __iter__(self):
        return PriorHypothesisIter(self.prior_hypotheses)

    def __len__(self):
        return len(self.prior_hypotheses)