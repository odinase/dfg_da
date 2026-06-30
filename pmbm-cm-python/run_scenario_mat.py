"""Run the ported PMBM filter on the ``scenario1MC.mat`` benchmark dataset.

This mirrors the "Angel" (reference-filter) branch of
``pmbm-cm-matlab/script_pmbm91.m``: at each of the 1397 time steps it predicts
with an ownship-centred PPP birth, updates with the Cartesian measurements,
estimates with estimator 3, scores GOSPA against the ground truth, and (unless
disabled) dumps the per-step filter state to ``priorLikelihood{k}.mat`` — the
analog of ``script_pmbm91.m:1236``.

Each invocation writes all of its output (per-step dumps, summary plot,
consolidated ``results_mat.mat``) into a fresh ``runs/run_<timestamp>/`` folder.

Usage:
    python run_scenario_mat.py [--mat PATH] [--pd 0.9] [--steps N]
                               [--estimator {1,2,3}] [--outbase runs]
                               [--no-dump] [--no-plot]
"""

import argparse
import time

import numpy as np
import scipy.io as sio

from pmbm import compute_gospa_error, estimate1, estimate2, estimate3, predict, prune, update
from pmbm.matio import make_run_dir, save_priorlikelihood
from pmbm.scenario_mat import load_mat_scenario


# Filter parameters (Table I of Brekke & Hem 2023 / the Angel branch).
NHYP_MAX = 400
GATING_THRESHOLD = 20.0
T_PRUNING = 0.0
T_PRUNING_POIS = 1e-5
EXISTENCE_THRESHOLD = 1e-5
EXISTENCE_EST_THRESHOLD1 = 0.4


def run(mat_path, pd_value=0.9, steps=None, type_estimator=3,
        outbase="runs", dump=True, plot=True):
    scn = load_mat_scenario(mat_path, pd_value=pd_value)
    Nsteps = scn.Nsteps if steps is None else min(steps, scn.Nsteps)

    run_dir = make_run_dir(outbase) if (dump or plot) else None
    if run_dir:
        print(f"Output directory: {run_dir}")
    print(f"Running {Nsteps} of {scn.Nsteps} steps, p_d={scn.p_d}, estimator={type_estimator}")

    sq_gospa = np.zeros(Nsteps)
    sq_loc = np.zeros(Nsteps)
    sq_false = np.zeros(Nsteps)
    sq_mis = np.zeros(Nsteps)
    n_hyps = np.zeros(Nsteps, dtype=int)
    n_est = np.zeros(Nsteps, dtype=int)

    flt_upd = scn.initial_filter()

    t0 = time.time()
    for k in range(1, Nsteps + 1):
        weights_b, means_b, covs_b = scn.birth_at(k)
        flt_pred = predict(flt_upd, scn.F, scn.Q, scn.p_s, weights_b, means_b, covs_b)

        z = scn.measurements_at(k)
        flt_upd = update(flt_pred, z, scn.H, scn.R, scn.p_d, k,
                         GATING_THRESHOLD, scn.intensity_clutter, NHYP_MAX)

        if type_estimator == 1:
            X_est = estimate1(flt_upd, EXISTENCE_EST_THRESHOLD1)
        elif type_estimator == 2:
            X_est = estimate2(flt_upd)
        else:
            X_est = estimate3(flt_upd)

        d2, loc, mis, fal = compute_gospa_error(
            X_est, scn.X_truth, scn.t_birth, scn.t_death, scn.c_gospa, k,
            pos_idx=scn.pos_idx,
        )
        sq_gospa[k - 1] = d2
        sq_loc[k - 1] = loc
        sq_false[k - 1] = fal
        sq_mis[k - 1] = mis
        n_hyps[k - 1] = len(flt_upd.get("globHypWeight", []))
        n_est[k - 1] = X_est.size // 4

        if dump:
            save_priorlikelihood(flt_upd, k, run_dir, z, X_est, scn.p_d,
                                 ownship_k=scn.ownship[:, k - 1])

        flt_upd = prune(flt_upd, T_PRUNING, T_PRUNING_POIS, NHYP_MAX, EXISTENCE_THRESHOLD)

        if k % 50 == 0 or k == Nsteps:
            elapsed = time.time() - t0
            print(f"  step {k}/{Nsteps}  "
                  f"hyps={n_hyps[k - 1]} est={n_est[k - 1]}  "
                  f"({elapsed:.1f}s, {elapsed / k * 1000:.0f} ms/step)")

    rms_gospa_tot = np.sqrt(np.sum(sq_gospa) / Nsteps)
    rms_loc_tot = np.sqrt(np.sum(sq_loc) / Nsteps)
    rms_false_tot = np.sqrt(np.sum(sq_false) / Nsteps)
    rms_mis_tot = np.sqrt(np.sum(sq_mis) / Nsteps)

    print()
    print(f"Mean GOSPA^2 per step      : {np.mean(sq_gospa):.4f}")
    print(f"RMS GOSPA total            : {rms_gospa_tot:.4f}")
    print(f"RMS GOSPA localisation tot : {rms_loc_tot:.4f}")
    print(f"RMS GOSPA false target tot : {rms_false_tot:.4f}")
    print(f"RMS GOSPA missed target tot: {rms_mis_tot:.4f}")

    if run_dir:
        steps_axis = np.arange(1, Nsteps + 1)
        results = {
            "sq_gospa": sq_gospa, "sq_loc": sq_loc,
            "sq_false": sq_false, "sq_mis": sq_mis,
            "n_hyps": n_hyps, "n_est": n_est,
            "pD": scn.p_d, "Nsteps": Nsteps,
            "rms_gospa_tot": rms_gospa_tot,
        }
        sio.savemat(f"{run_dir}/results_mat.mat", results, do_compression=True)

        if plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            for ax, data, title in zip(
                axes.ravel(),
                [sq_gospa, sq_loc, sq_false, sq_mis],
                ["GOSPA^2", "Localisation^2", "False target^2", "Missed target^2"],
            ):
                ax.plot(steps_axis, data, "b", linewidth=0.8)
                ax.set_xlabel("Time step")
                ax.set_ylabel(title)
                ax.grid(True)
            fig.suptitle(f"scenario1MC.mat — p_d={scn.p_d}, estimator {type_estimator}")
            fig.tight_layout()
            fig.savefig(f"{run_dir}/gospa_results_mat.png", dpi=120)
            print(f"\nSaved plot and results to {run_dir}/")

    if run_dir:
        print(f"Output directory: {run_dir}")

    return {
        "sq_gospa": sq_gospa, "rms_gospa_tot": rms_gospa_tot, "run_dir": run_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="PMBM filter on scenario1MC.mat")
    parser.add_argument("--mat", default="../scenario1MC.mat", help="path to scenario1MC.mat")
    parser.add_argument("--pd", type=float, default=0.9, help="detection probability")
    parser.add_argument("--steps", type=int, default=None, help="limit number of time steps")
    parser.add_argument("--estimator", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--outbase", default="runs", help="base dir for per-run output folders")
    parser.add_argument("--no-dump", action="store_true", help="do not write priorLikelihood{k}.mat")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    run(args.mat, pd_value=args.pd, steps=args.steps, type_estimator=args.estimator,
        outbase=args.outbase, dump=not args.no_dump, plot=not args.no_plot)


if __name__ == "__main__":
    main()
