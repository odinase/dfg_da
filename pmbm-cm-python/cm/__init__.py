"""Python port of the cluster-management (CM) PMBM filter prior/likelihood
pipeline (Stage 1).

Reproduces the per-step variables dumped at
``pmbm-cm-matlab/script_pmbm91.m:1236-1237`` (``priorLikelihood{k}.mat``).
"""

from .columns import DIM_TAR, DIM_Z, MAX_LAG, InCol
from .dump import DUMP_VARS, save_priorlikelihood
from .loader import load_cm_scenario
from .pipeline import run_prior_likelihood
from .precompute import compute_pre_cluster_threshold
from .state import CMState, initial_state

__all__ = [
    "InCol", "DIM_TAR", "DIM_Z", "MAX_LAG",
    "load_cm_scenario", "initial_state", "CMState",
    "run_prior_likelihood", "compute_pre_cluster_threshold",
    "save_priorlikelihood", "DUMP_VARS",
]
