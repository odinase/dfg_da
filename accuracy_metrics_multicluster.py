"""Accuracy of the approximate multi-cluster data-association solvers.

For every tractable scan we compute the **exact** (true) track-association
marginal probabilities and normalization constant with ``MulticlusterExactEHM2``
and compare against the two approximate methods:

* **LBP** -- the multi-cluster loopy belief propagation + Bethe pseudodual
  (``py_dfg_da.lbp.lbp_multicluster``), and
* **IE LBP+Bethe** -- the inclusion-exclusion (overlapping event space) solver
  from ``cluster_partition`` with the compiled single-hypothesis LBP+Bethe inner
  solver (the method added to ``ravens_parser_parallell_multicluster.py``).

It reuses the solvers, the ``MAX_ENUM_FOR_EXACT`` skip cap and the topology
helpers from ``ravens_parser_parallell_multicluster`` (single source of truth),
and the ``Marginals`` / ``MarginalsErrors`` accuracy primitives from
``dfg_da.stats_logger`` (same convention as ``plotting_multicluster.py``).

Outputs (under ``ravens_output_multicluster/accuracy/``):

* ``accuracy.csv``                   -- per-scan accuracy summary
* ``marginal_abs_error_hist.{png,pdf}``  -- distribution of |p_est - p_true|
* ``marginal_signed_error_hist.{png,pdf}`` -- distribution of (p_true - p_est)
* ``marginal_max_error_per_scan.{png,pdf}`` -- worst marginal error per scan
* ``normconst_scatter.{png,pdf}``    -- estimated vs. true Z (log-log, y=x)
* ``normconst_relerror_hist.{png,pdf}`` -- |Z_est - Z_true| / Z_true

Run from the repo root::

    EVAL_WORKERS=8 pmbm-cm-python/.venv/bin/python -W ignore accuracy_metrics_multicluster.py
"""

import os
import csv
import time
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool

import py_dfg_da
import dfg_da.marginals_computers as mc
import dfg_da.stats_logger as sl
from dfg_da.stats_logger import Marginals, MarginalsErrors
from dfg_da.marginal_association_Odin import ExplicitHypothesisEnumerationError
from cluster_partition import MulticlusterPartitionedMarginals

# Reuse the exact same solvers / config as the main eval so accuracy is measured
# on the identical method wiring (no duplicated solver code).
import ravens_parser_parallell_multicluster as ev

PMBM_DATA_PATH = ev.PMBM_DATA_PATH
MAX_ENUM_FOR_EXACT = ev.MAX_ENUM_FOR_EXACT
ACCURACY_PATH = f"{ev.OUTPUT_PATH_BASE}/accuracy"

# Methods compared against the exact (true) values. Keyed by the short id used
# in the per-scan record; value is (display label, colour).
METHODS = [
    ("lbp", "LBP", "C0"),
    ("ie", "IE LBP+Bethe", "C3"),
]


# ---------------------------------------------------------------------------
# Per-scan accuracy computation
# ---------------------------------------------------------------------------

def loop_func(pmbm_file):
    """Return a per-scan accuracy record, or ``None`` if the scan has no
    clusters or is too heavy for the exact (truth) solver."""
    mat_data = sl.MatFileParser(pmbm_file, use_cpp=True)
    R = np.asfortranarray(mat_data.reward_matrix_edmund)
    R_LC = np.asfortranarray(mat_data.reward_matrix_lc)
    prior_hypotheses_per_cluster = mat_data.prior_hypotheses_per_cluster

    if len(prior_hypotheses_per_cluster) == 0:
        return None

    pmbm_filename = Path(pmbm_file).stem
    assocLocal = mat_data.ws["assocLocal"].copy()

    enum_product, _ = ev.enumeration_product(
        R_LC, prior_hypotheses_per_cluster, assocLocal)
    # No exact truth available for scans above the skip cap -> no accuracy to
    # measure, so leave them out entirely.
    if enum_product > MAX_ENUM_FOR_EXACT:
        return None

    # Snapshot hypotheses for the IE solver BEFORE the exact solver reindexes
    # (mutates) them in place.
    cp_prior_hypotheses = ev._to_cp_hypotheses(prior_hypotheses_per_cluster)

    # --- Approximate method 1: multi-cluster LBP + Bethe --------------------
    lbp = py_dfg_da.lbp.lbp_multicluster(R, prior_hypotheses_per_cluster)
    M_lbp = np.asarray(lbp.track_association_marginals()).T           # (n, m+2)
    Z_lbp = float(lbp.bethe_pseudodual_normalization_constant())

    # --- Approximate method 2: inclusion-exclusion LBP+Bethe ----------------
    ie = MulticlusterPartitionedMarginals(
        R_LC, cp_prior_hypotheses, ev.ie_lbp_bethe_solver,
        mode="overlap_ie", assocLocal=assocLocal)
    M_ie, _ie_theta, Z_ie = ie.compute_marginals_likelihood()
    M_ie = np.asarray(M_ie)
    Z_ie = float(Z_ie)

    # --- Exact (true) marginals + normalization constant --------------------
    try:
        exact = ev.exact_computer(R_LC, prior_hypotheses_per_cluster,
                                  assocLocal=assocLocal)
    except ExplicitHypothesisEnumerationError:
        return None
    M_ex = np.asarray(exact.exact_marginals)                         # (n, m+2)
    Z_ex = float(exact.exact_normalization_constant)

    if M_lbp.shape != M_ex.shape or M_ie.shape != M_ex.shape:
        # Shapes should always align (verified); guard rather than corrupt stats.
        return None

    exact_marg = Marginals(M_ex)
    err_lbp = MarginalsErrors(exact_marg, Marginals(M_lbp))
    err_ie = MarginalsErrors(exact_marg, Marginals(M_ie))

    def relerr(z_est):
        return abs(z_est - Z_ex) / Z_ex if Z_ex > 0 else np.nan

    return {
        "file": pmbm_filename,
        "enum_product": enum_product,
        "cyclomatic": ev.cyclomatic_number(R_LC),
        "n_tracks": int(M_ex.shape[0]),
        "n_marg_entries": int(M_ex.size),
        # normalization constants
        "Z_exact": Z_ex,
        "Z_lbp": Z_lbp,
        "Z_ie": Z_ie,
        "relerr_Z_lbp": relerr(Z_lbp),
        "relerr_Z_ie": relerr(Z_ie),
        # per-scan marginal error summaries
        "max_marg_err_lbp": float(err_lbp.max_errors.max()),
        "max_marg_err_ie": float(err_ie.max_errors.max()),
        "mean_marg_err_lbp": float(err_lbp.abs_errors.mean()),
        "mean_marg_err_ie": float(err_ie.abs_errors.mean()),
        # raw per-entry errors (float32 to keep pickling cheap) for histograms
        "_abs_lbp": err_lbp.abs_errors.astype(np.float32),
        "_abs_ie": err_ie.abs_errors.astype(np.float32),
        "_signed_lbp": err_lbp.raw_errors.astype(np.float32),
        "_signed_ie": err_ie.raw_errors.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Plotting (matplotlib + numpy only; saves png + pdf like plotting_multicluster)
# ---------------------------------------------------------------------------

def save_fig(fig, name, out_dir):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = Path(out_dir) / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {Path(out_dir) / name}.{{png,pdf}}")


def _concat(records, key):
    arrs = [r[key] for r in records if r[key].size]
    return np.concatenate(arrs) if arrs else np.array([], dtype=np.float32)


def plot_marginal_abs_error(records, out_dir):
    """Histogram of |p_est - p_true| over every marginal entry, per method."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = {mid: _concat(records, f"_abs_{mid}") for mid, _, _ in METHODS}
    all_pos = np.concatenate([d[d > 0] for d in data.values()])
    if all_pos.size == 0:
        plt.close(fig)
        return
    lo = max(all_pos.min(), 1e-16)
    bins = np.logspace(np.log10(lo), 0, 60)
    for mid, label, color in METHODS:
        d = data[mid]
        dpos = d[d > 0]
        ax.hist(dpos, bins=bins, alpha=0.55, color=color,
                label=f"{label}  (mean {d.mean():.2e}, max {d.max():.2e})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"absolute marginal error $|p_\mathrm{est}-p_\mathrm{true}|$")
    ax.set_ylabel("count (marginal entries)")
    ax.set_title("Marginal-probability accuracy vs. exact")
    ax.legend()
    save_fig(fig, "marginal_abs_error_hist", out_dir)


def plot_marginal_signed_error(records, out_dir):
    """Histogram of the signed error (p_true - p_est), per method."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = {mid: _concat(records, f"_signed_{mid}") for mid, _, _ in METHODS}
    all_s = np.concatenate(list(data.values()))
    if all_s.size == 0:
        plt.close(fig)
        return
    m = np.abs(all_s).max() or 1.0
    bins = np.linspace(-m, m, 80)
    for mid, label, color in METHODS:
        d = data[mid]
        ax.hist(d, bins=bins, alpha=0.55, color=color,
                label=f"{label}  (std {d.std():.2e})")
    ax.set_yscale("log")
    ax.set_xlabel(r"signed marginal error $p_\mathrm{true}-p_\mathrm{est}$")
    ax.set_ylabel("count (marginal entries)")
    ax.set_title("Signed marginal error vs. exact")
    ax.legend()
    save_fig(fig, "marginal_signed_error_hist", out_dir)


def plot_max_error_per_scan(records, out_dir):
    """Histogram of the worst marginal error in each scan, per method."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vals = {mid: np.array([r[f"max_marg_err_{mid}"] for r in records])
            for mid, _, _ in METHODS}
    allv = np.concatenate([v[v > 0] for v in vals.values()])
    if allv.size == 0:
        plt.close(fig)
        return
    bins = np.logspace(np.log10(max(allv.min(), 1e-16)), 0, 50)
    for mid, label, color in METHODS:
        v = vals[mid]
        vpos = v[v > 0]
        ax.hist(vpos, bins=bins, alpha=0.55, color=color,
                label=f"{label}  (median {np.median(v):.2e})")
    ax.set_xscale("log")
    ax.set_xlabel("max absolute marginal error per scan")
    ax.set_ylabel("number of scans")
    ax.set_title("Worst-case marginal error per scan vs. exact")
    ax.legend()
    save_fig(fig, "marginal_max_error_per_scan", out_dir)


def plot_normconst_scatter(records, out_dir):
    """Estimated vs. true normalization constant (log-log, y=x reference)."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    Zex = np.array([r["Z_exact"] for r in records])
    for mid, label, color in METHODS:
        Zest = np.array([r[f"Z_{mid}"] for r in records])
        ok = (Zex > 0) & (Zest > 0)
        ax.scatter(Zex[ok], Zest[ok], s=10, alpha=0.4, color=color, label=label)
    both = Zex[Zex > 0]
    if both.size:
        lim = [both.min() * 0.5, both.max() * 2]
        ax.plot(lim, lim, "k--", lw=1, label="exact ($y=x$)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true normalization constant $Z_\\mathrm{exact}$")
    ax.set_ylabel("estimated normalization constant")
    ax.set_title("Normalization-constant accuracy vs. exact")
    ax.legend()
    save_fig(fig, "normconst_scatter", out_dir)


def plot_normconst_relerror(records, out_dir):
    """Histogram of the relative normalization-constant error, per method."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vals = {mid: np.array([r[f"relerr_Z_{mid}"] for r in records])
            for mid, _, _ in METHODS}
    allv = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in vals.values()])
    if allv.size == 0:
        plt.close(fig)
        return
    bins = np.logspace(np.log10(max(allv.min(), 1e-16)),
                       np.log10(allv.max()), 50)
    for mid, label, color in METHODS:
        v = vals[mid]
        v = v[np.isfinite(v)]
        vpos = v[v > 0]
        ax.hist(vpos, bins=bins, alpha=0.55, color=color,
                label=f"{label}  (median {np.median(v):.2e})")
    ax.set_xscale("log")
    ax.set_xlabel(r"relative $Z$ error $|Z_\mathrm{est}-Z_\mathrm{true}|/Z_\mathrm{true}$")
    ax.set_ylabel("number of scans")
    ax.set_title("Normalization-constant relative error vs. exact")
    ax.legend()
    save_fig(fig, "normconst_relerror_hist", out_dir)


# ---------------------------------------------------------------------------
# CSV + summary
# ---------------------------------------------------------------------------

def _binned_stats(x, y, nbins=12):
    """Median and IQR of ``y`` in ``nbins`` equal-count bins of ``x``.
    Returns (bin_centers, median, q25, q75)."""
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    edges = np.linspace(0, x.size, nbins + 1).astype(int)
    cx, med, q25, q75 = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        cx.append(np.median(x[a:b]))
        med.append(np.median(y[a:b]))
        q25.append(np.percentile(y[a:b], 25))
        q75.append(np.percentile(y[a:b], 75))
    return np.array(cx), np.array(med), np.array(q25), np.array(q75)


def plot_accuracy_vs_cyclomatic(records, out_dir):
    """Marginal error and normalization-constant error vs. the cyclomatic number
    of the association graph LBP runs on (per-scan scatter + binned median/IQR)."""
    mu = np.array([r["cyclomatic"] for r in records], dtype=float)
    panels = [
        ("max marginal error", "max_marg_err_{}", False),
        (r"relative $Z$ error $|Z_\mathrm{est}-Z_\mathrm{true}|/Z_\mathrm{true}$",
         "relerr_Z_{}", True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (ylabel, keyfmt, is_rel) in zip(axes, panels):
        for mid, label, color in METHODS:
            y = np.array([r[keyfmt.format(mid)] for r in records], dtype=float)
            ok = np.isfinite(y) & np.isfinite(mu)
            ax.scatter(mu[ok], y[ok], s=8, alpha=0.15, color=color)
            cx, med, q25, q75 = _binned_stats(mu[ok], y[ok])
            ax.plot(cx, med, "-o", color=color, ms=4, label=f"{label} (binned median)")
            ax.fill_between(cx, q25, q75, color=color, alpha=0.18)
        ax.set_xlabel(r"cyclomatic number $\mu = E - V + C$ of the LBP graph")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Approximation accuracy vs. cyclomatic number (loopiness) of the LBP graph")
    save_fig(fig, "accuracy_vs_cyclomatic", out_dir)


def save_csv(records, out_dir):
    fields = ["file", "enum_product", "cyclomatic", "n_tracks", "n_marg_entries",
              "Z_exact", "Z_lbp", "Z_ie", "relerr_Z_lbp", "relerr_Z_ie",
              "max_marg_err_lbp", "max_marg_err_ie",
              "mean_marg_err_lbp", "mean_marg_err_ie"]
    p = Path(out_dir) / "accuracy.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"Wrote {p}")


def summarize(records, total_s):
    print("\n" + "=" * 68)
    print(f"Accuracy over {len(records)} scans with exact truth "
          f"(computed in {total_s:.1f} s = {total_s/60:.2f} min)")
    for mid, label, _ in METHODS:
        abs_all = _concat(records, f"_abs_{mid}")
        relZ = np.array([r[f"relerr_Z_{mid}"] for r in records])
        relZ = relZ[np.isfinite(relZ)]
        maxe = np.array([r[f"max_marg_err_{mid}"] for r in records])
        print(f"  {label:14s}: marg |err| mean={abs_all.mean():.3e} "
              f"max={abs_all.max():.3e} | per-scan max median={np.median(maxe):.3e} "
              f"| relZ median={np.median(relZ):.3e} max={relZ.max():.3e}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    pmbm_files = sorted(
        glob(PMBM_DATA_PATH + "/*.mat"),
        key=lambda p: int("".join(c for c in Path(p).stem if c.isdigit()) or 0))
    if not pmbm_files:
        raise ValueError(f"No .mat files found under {PMBM_DATA_PATH}")

    out_dir = Path(ACCURACY_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_workers = int(os.environ.get("EVAL_WORKERS", "8"))
    print(f"Computing accuracy for {len(pmbm_files)} files with {n_workers} workers...")
    start = time.time()
    records = []
    with Pool(processes=n_workers) as pool:
        for r in tqdm(pool.imap_unordered(loop_func, pmbm_files, chunksize=1),
                      total=len(pmbm_files)):
            if r is not None:
                records.append(r)
    total_s = time.time() - start

    if not records:
        raise SystemExit("No scans with exact truth were computed.")

    summarize(records, total_s)
    save_csv(records, out_dir)
    plot_marginal_abs_error(records, out_dir)
    plot_marginal_signed_error(records, out_dir)
    plot_max_error_per_scan(records, out_dir)
    plot_normconst_scatter(records, out_dir)
    plot_normconst_relerror(records, out_dir)
    plot_accuracy_vs_cyclomatic(records, out_dir)
    print(f"Done in {total_s:.1f} s. Outputs in {out_dir}/")
