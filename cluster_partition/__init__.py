"""
cluster_partition
=================

Self-contained, pure-Python implementation of the cluster-partitioning /
cluster-conditioning marginalization of Chapter 7 (Sections 7.1-7.5) of

    O. A. Severinsen, "Efficient cluster marginalization for multi-cluster,
    multi-hypothesis data association", master's thesis, NTNU.

Compatible in structure and call-contracts with the reference codebase
https://github.com/odinase/dfg_da (the ``cluster_bayes_tree`` and
``cluster_conditioning_lbp`` modules), but with no dependency on the compiled
``py_dfg_da`` pybind module.
"""
from .prior_hypotheses import Hypothesis, Hypotheses, HypothesesList, make_hypotheses
from .cluster_links import ClusterLinks, LinkingMappings, cartesian_product
from .solvers import ExactEnumerationSolver, EHM2Solver, LBPBetheSolver
from .partitioning import (
    MODES,
    NONE_EVENT,
    ConditionedCluster,
    ConditionalSuperclusterMarginals,
    MulticlusterPartitionedMarginals,
)
from .reference import multicluster_exact_reference

__all__ = [
    "Hypothesis", "Hypotheses", "HypothesesList", "make_hypotheses",
    "ClusterLinks", "LinkingMappings", "cartesian_product",
    "ExactEnumerationSolver", "EHM2Solver", "LBPBetheSolver",
    "MODES", "NONE_EVENT", "ConditionedCluster",
    "ConditionalSuperclusterMarginals", "MulticlusterPartitionedMarginals",
    "multicluster_exact_reference",
]
