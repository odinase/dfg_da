"""Demo of the PMBM filter on the Williams-15 scenario.

Port of ``PoissonMBMtarget_filter.m``. Runs a number of Monte Carlo trials and
reports the RMS GOSPA error (alpha = 2) and its localisation / missed / false
decomposition, then plots the per-time-step RMS GOSPA curves.

Usage:
    python run_demo.py [--mc N] [--estimator {1,2,3}] [--seed S] [--no-plot]
"""

import argparse
import time

import numpy as np

from pmbm import (
    build_scenario,
    compute_gospa_error,
    create_measurement,
    estimate1,
    estimate2,
    estimate3,
    predict,
    prune,
    update,
)


def run(n_mc=10, type_estimator=1, seed=9, plot=True):
    rng = np.random.default_rng(seed)
    s = build_scenario(rng=rng, seed=seed)

    Nx = s.Nx
    Nsteps = s.Nsteps

    # Filter parameters (as in the MATLAB demo).
    T_pruning = 0.0
    T_pruningPois = 1e-5
    Nhyp_max = 200
    gating_threshold = 20.0
    existence_threshold = 1e-5
    existence_estimation_threshold1 = 0.4

    sq_gospa = np.zeros(Nsteps)
    sq_loc = np.zeros(Nsteps)
    sq_false = np.zeros(Nsteps)
    sq_mis = np.zeros(Nsteps)

    for it in range(n_mc):
        t0 = time.time()

        flt_pred = {
            "weightPois": [float(s.lambda0)],
            "meanPois": [s.means_b[:, 0].astype(float).copy()],
            "covPois": [s.covs_b[0].astype(float).copy()],
            "tracks": [],
            "globHyp": np.zeros((0, 0), dtype=int),
            "globHypWeight": np.zeros(0),
        }

        # Simulate the measurements for all time steps.
        z_t = []
        for k in range(1, Nsteps + 1):
            z = create_measurement(
                s.X_truth[:, k - 1], s.t_birth, s.t_death, s.p_d, s.l_clutter,
                s.Area, k, s.H, s.chol_R, Nx, rng,
            )
            z_t.append(z)

        for k in range(1, Nsteps + 1):
            z = z_t[k - 1]

            flt_upd = update(
                flt_pred, z, s.H, s.R, s.p_d, k, gating_threshold,
                s.intensity_clutter, Nhyp_max,
            )

            if type_estimator == 1:
                X_est = estimate1(flt_upd, existence_estimation_threshold1)
            elif type_estimator == 2:
                X_est = estimate2(flt_upd)
            else:
                X_est = estimate3(flt_upd)

            d2, loc, mis, fal = compute_gospa_error(
                X_est, s.X_truth, s.t_birth, s.t_death, s.c_gospa, k,
            )
            sq_gospa[k - 1] += d2
            sq_loc[k - 1] += loc
            sq_false[k - 1] += fal
            sq_mis[k - 1] += mis

            flt_upd = prune(
                flt_upd, T_pruning, T_pruningPois, Nhyp_max, existence_threshold,
            )

            flt_pred = predict(
                flt_upd, s.F, s.Q, s.p_s, s.weights_b, s.means_b, s.covs_b,
            )

        print(f"Completed iteration {it + 1} in {time.time() - t0:.2f} s")

    rms_gospa_t = np.sqrt(sq_gospa / n_mc)
    rms_loc_t = np.sqrt(sq_loc / n_mc)
    rms_false_t = np.sqrt(sq_false / n_mc)
    rms_mis_t = np.sqrt(sq_mis / n_mc)

    rms_gospa_tot = np.sqrt(np.sum(sq_gospa) / (n_mc * Nsteps))
    rms_loc_tot = np.sqrt(np.sum(sq_loc) / (n_mc * Nsteps))
    rms_false_tot = np.sqrt(np.sum(sq_false) / (n_mc * Nsteps))
    rms_mis_tot = np.sqrt(np.sum(sq_mis) / (n_mc * Nsteps))

    print()
    print(f"RMS GOSPA total            : {rms_gospa_tot:.4f}")
    print(f"RMS GOSPA localisation tot : {rms_loc_tot:.4f}")
    print(f"RMS GOSPA false target tot : {rms_false_tot:.4f}")
    print(f"RMS GOSPA missed target tot: {rms_mis_tot:.4f}")

    if plot:
        import matplotlib.pyplot as plt

        steps = np.arange(1, Nsteps + 1)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, data, title in zip(
            axes.ravel(),
            [rms_gospa_t, rms_loc_t, rms_false_t, rms_mis_t],
            ["RMS GOSPA error", "RMS GOSPA localisation error",
             "RMS GOSPA false target error", "RMS GOSPA missed target error"],
        ):
            ax.plot(steps, data, "b", linewidth=1.3)
            ax.set_xlabel("Time step")
            ax.set_ylabel(title)
            ax.grid(True)
        fig.tight_layout()
        fig.savefig("gospa_results.png", dpi=120)
        print("\nSaved plot to gospa_results.png")

    return {
        "rms_gospa_t": rms_gospa_t,
        "rms_loc_t": rms_loc_t,
        "rms_false_t": rms_false_t,
        "rms_mis_t": rms_mis_t,
        "rms_gospa_tot": rms_gospa_tot,
    }


def main():
    parser = argparse.ArgumentParser(description="PMBM filter demo")
    parser.add_argument("--mc", type=int, default=10, help="number of Monte Carlo runs")
    parser.add_argument("--estimator", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    run(n_mc=args.mc, type_estimator=args.estimator, seed=args.seed, plot=not args.no_plot)


if __name__ == "__main__":
    main()
