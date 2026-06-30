"""Driver for the cluster-management (CM) PMBM prior/likelihood pipeline.

Loads ``scenario1MC.mat``, runs the per-step prior/likelihood pipeline from an
empty initial state, and dumps ``priorLikelihood{k}.mat`` — the reproduction of
``script_pmbm91.m:1236-1237``.

With the Stage 2b carry-over (branch-and-bound hypothesis generation + pruning /
n-scan merge / cluster splitting / recycling) in place, ``--steps`` may be any
number of steps; each step's dump is produced in turn.

Usage:
    python run_cm.py [--mat ../scenario1MC.mat] [--pd 0.9] [--steps 3]
                     [--outbase runs] [--no-dump]
"""

import argparse

import numpy as np

from cm import (
    InCol,
    compute_pre_cluster_threshold,
    initial_state,
    load_cm_scenario,
    run_prior_likelihood,
    save_priorlikelihood,
)
from cm.carryover import carry_over
from pmbm.matio import make_run_dir

# Scalar parameters from script_pmbm91.m (the doAngel=false / CM branch).
G_GATE = 3.0
GAMMA_GATE = G_GATE ** 2  # 9
DO_MECHI = True
DO_SEMI_TWO_POINT = True
ADOLE_THRES = 1.0
NHYP_MAX = 30
NHYP_TOTAL_MAX = 150
SIG = 0.05               # majority-track significance (script:153)
SPLITTING_THRESHOLD = 0.018   # script:178
MAHA_CS_THRES = 3        # value used at the CS call site (script:1950)
NON_SPLIT_LAG = 2        # script:181 (Mechi data)


def run(mat_path, pd_value=0.9, steps=1, outbase="runs", dump=True):
    scn = load_cm_scenario(mat_path, pd_value=pd_value)
    incol = InCol()
    state = initial_state(incol)

    pre_cluster_threshold = compute_pre_cluster_threshold(scn.system, scn.params, scn.pD)

    config = {
        "pD": scn.pD,
        "gamma_gate": GAMMA_GATE,
        "do_mechi": DO_MECHI,
        "do_semi_two_point": DO_SEMI_TWO_POINT,
        "adole_thres": ADOLE_THRES,
        "nHypoMax": NHYP_MAX,
        "nHypoTotalMax": NHYP_TOTAL_MAX,
        "pre_cluster_threshold": pre_cluster_threshold,
        "label_gen": state.label_gen,
        "sig": SIG,
        "splitting_threshold": SPLITTING_THRESHOLD,
        "maha_cs_thres": MAHA_CS_THRES,
        "non_split_lag": NON_SPLIT_LAG,
    }

    run_dir = make_run_dir(outbase) if dump else None
    if run_dir:
        print(f"Output directory: {run_dir}")
    print(f"preClusterThreshold = {pre_cluster_threshold:.4f}")

    n = min(steps, scn.Nsteps)

    for k in range(1, n + 1):
        dump_bundle, extra = run_prior_likelihood(state, scn, k, config)
        m = dump_bundle["measurements"].shape[1]
        print(f"  step {k}: m={m} measurements, "
              f"nTracksTenta={dump_bundle['meaHistColNew'].shape[1]}, "
              f"nTracks(in)={dump_bundle['trackFile'].shape[1]}")
        if dump:
            path = save_priorlikelihood(dump_bundle, k, run_dir)
            print(f"    wrote {path}")

        if k < n:
            try:
                carry_over(state, dump_bundle, extra, config)
            except NotImplementedError as exc:
                print(f"  carry-over stopped after step {k}: {exc}")
                break

    if run_dir:
        print(f"Output directory: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="CM PMBM prior/likelihood pipeline")
    parser.add_argument("--mat", default="../scenario1MC.mat")
    parser.add_argument("--pd", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--outbase", default="runs")
    parser.add_argument("--no-dump", action="store_true")
    args = parser.parse_args()
    run(args.mat, pd_value=args.pd, steps=args.steps, outbase=args.outbase,
        dump=not args.no_dump)


if __name__ == "__main__":
    main()
