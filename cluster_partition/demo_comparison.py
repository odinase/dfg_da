"""
demo_comparison.py
==================

Runs the three event-space modes -- ``disjoint_exact`` (Eq. (7.5)),
``overlap_firststage`` (Eq. (7.29), thesis-published) and ``overlap_ie``
(Eq. (7.29) + inclusion-exclusion, the thesis "future work") -- against the
brute-force full-graph exact reference, with each of the three inner solvers
(exact enumeration, EHM2, LBP+Bethe).

For each (case, solver, mode) it reports the normalization constant Z and the
maximum absolute error of the association marginals versus the exact reference.

Run with:  python demo_comparison.py
"""
import numpy as np

from cluster_partition import (
    EHM2Solver,
    ExactEnumerationSolver,
    LBPBetheSolver,
    MulticlusterPartitionedMarginals,
    multicluster_exact_reference,
)
from cluster_partition.tests.test_partitioning import ALL_CASES  # reuse fixtures

SOLVERS = {
    "exact": ExactEnumerationSolver,
    "ehm2": EHM2Solver,
    "lbp": LBPBetheSolver,
}
MODES = ("disjoint_exact", "overlap_firststage", "overlap_ie")


def run_case(name):
    R, mk = ALL_CASES[name]()
    ref_marg, _, ref_Z = multicluster_exact_reference(R, mk())
    print(f"\n=== case '{name}'  (exact reference Z = {ref_Z:.6f}) ===")
    header = f"{'solver':<8}{'mode':<20}{'Z':>14}{'Z err':>14}{'marg max err':>14}"
    print(header)
    print("-" * len(header))
    for sname, sclass in SOLVERS.items():
        for mode in MODES:
            try:
                solver = sclass()
            except Exception as exc:  # e.g. pyehm missing
                print(f"{sname:<8}{mode:<20}{'(solver unavailable: ' + str(exc) + ')'}")
                continue
            # disjoint_exact enforces detections; LBP defers those to the exact
            # primitive, so it is exact there too -- still worth showing.
            m = MulticlusterPartitionedMarginals(R, mk(), solver, mode=mode)
            marg, _, Z = m.compute_marginals_likelihood()
            z_err = abs(Z - ref_Z)
            marg_err = np.max(np.abs(marg - ref_marg))
            print(f"{sname:<8}{mode:<20}{Z:>14.6f}{z_err:>14.6e}{marg_err:>14.6e}")


def main():
    for name in ALL_CASES:
        run_case(name)
    print(
        "\nReading the table:\n"
        "  * disjoint_exact and overlap_ie reproduce the exact constant and\n"
        "    marginals to solver precision (exact / EHM2 inner solvers).\n"
        "  * overlap_firststage always reports a LARGER-than-exact constant\n"
        "    (Bonferroni over-estimate from using a single, odd I-E stage);\n"
        "    its marginals differ because the spurious mass is re-normalized.\n"
        "  * with the LBP+Bethe inner solver the constants are Bethe estimates;\n"
        "    overlap_ie removes the over-counting bias that overlap_firststage\n"
        "    carries, so its constant sits closest to the exact reference.\n"
    )


if __name__ == "__main__":
    main()
