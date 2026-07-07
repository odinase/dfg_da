"""Plot marginal-accuracy + normalization-constant (Z) accuracy from the
``ravens_output_multicluster_convergence/`` stat files.

Each ``*_stats`` file is a pickled ``dfg_da.stats_logger.MulticlusterData`` with 5
approximate methods + the exact reference (see the module docstring of
``ravens_parser_parallell_multicluster2.py``). For every scan we compare each
approximate method's per-track marginals and its normalization constant against
the exact EHM2 truth, then aggregate across scans.

    .venv/bin/python -W ignore plot_convergence_stats.py [--limit N] [--out DIR]

Reuses ``sl.Marginals`` / ``sl.MarginalsErrors`` (dfg_da/stats_logger.py) for the
error primitives; borrows the survival/heatmap styling from
``plotting_multicluster_memory_efficient_convergence.py`` + ``chapter10_plots.py``.
"""
from __future__ import annotations

import argparse
import pickle
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import dfg_da.stats_logger as sl

STATS_DIR = "./ravens_output_multicluster_convergence"

# One shared method list + colors so every figure overlays the same 5 methods.
# (label, MulticlusterData attribute, marginals attr, Z attr, color)
METHODS = [
    ("MCMH-LBP", "mcmhlbp_output", "approx_marginals", "approx_normalization_constant", "C0"),
    ("Bethe",    "mc_bethe_output", "marginals", "likelihood", "C1"),
    ("MHLBP",    "mc_mhlbp_output", "marginals", "likelihood", "C2"),
    ("PHD",      "mc_phd_output",   "marginals", "likelihood", "C3"),
    ("IE",       "mc_lbp_ie_output", "marginals", "likelihood", "C4"),
]


# --------------------------------------------------------------------------- #
# Load + reduce the stat files into flat per-method arrays
# --------------------------------------------------------------------------- #
class Collected:
    """Aggregated errors/values across all scans, keyed by method label."""

    def __init__(self):
        # marginal error arrays (concatenated over tracks x scans)
        self.abs_err: Dict[str, List[np.ndarray]] = {m[0]: [] for m in METHODS}
        self.raw_err: Dict[str, List[np.ndarray]] = {m[0]: [] for m in METHODS}
        self.misdet_err: Dict[str, List[np.ndarray]] = {m[0]: [] for m in METHODS}
        self.det_err: Dict[str, List[np.ndarray]] = {m[0]: [] for m in METHODS}
        self.nonexist_err: Dict[str, List[np.ndarray]] = {m[0]: [] for m in METHODS}
        # exact vs approx marginal values (for the correlation heatmap)
        self.marg_exact: Dict[str, List[np.ndarray]] = {m[0]: [] for m in METHODS}
        self.marg_approx: Dict[str, List[np.ndarray]] = {m[0]: [] for m in METHODS}
        # per-scan max marginal error
        self.max_err_per_scan: Dict[str, List[float]] = {m[0]: [] for m in METHODS}
        # normalization constants (per scan)
        self.z_exact: List[float] = []
        self.z_approx: Dict[str, List[float]] = {m[0]: [] for m in METHODS}
        self.n_scans = 0

    def finalize(self):
        cat = lambda d: {k: (np.concatenate(v) if v else np.array([])) for k, v in d.items()}
        self.abs_err = cat(self.abs_err)
        self.raw_err = cat(self.raw_err)
        self.misdet_err = cat(self.misdet_err)
        self.det_err = cat(self.det_err)
        self.nonexist_err = cat(self.nonexist_err)
        self.marg_exact = cat(self.marg_exact)
        self.marg_approx = cat(self.marg_approx)
        self.max_err_per_scan = {k: np.asarray(v) for k, v in self.max_err_per_scan.items()}
        self.z_exact = np.asarray(self.z_exact)
        self.z_approx = {k: np.asarray(v) for k, v in self.z_approx.items()}
        return self


def collect(files: List[str]) -> Collected:
    c = Collected()
    skipped = 0
    for f in files:
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            skipped += 1
            continue
        if getattr(d, "explicit_hypothesis_enumeration_error", False):
            skipped += 1
            continue
        exact_out = getattr(d, "exact_output", None)
        if exact_out is None:  # exact was skipped for this scan -> no truth
            skipped += 1
            continue
        exact_marg_arr = exact_out.exact_marginals
        exact_M = sl.Marginals(exact_marg_arr)
        z_ex = float(exact_out.exact_normalization_constant)
        self_used = False
        for label, attr, marg_attr, z_attr, _color in METHODS:
            out = getattr(d, attr, None)
            if out is None:
                continue
            approx_arr = getattr(out, marg_attr)
            if approx_arr.shape != exact_marg_arr.shape:
                continue
            approx_M = sl.Marginals(approx_arr)
            err = sl.MarginalsErrors(exact_M, approx_M)
            c.abs_err[label].append(err.abs_errors)
            c.raw_err[label].append(err.raw_errors)
            c.misdet_err[label].append(err.misdetection_errors)
            c.det_err[label].append(err.detection_errors)
            c.nonexist_err[label].append(np.abs(exact_M.nonexistence_marginals - approx_M.nonexistence_marginals))
            c.marg_exact[label].append(exact_marg_arr.ravel())
            c.marg_approx[label].append(approx_arr.ravel())
            c.max_err_per_scan[label].append(float(err.max_errors.max()) if err.max_errors.size else 0.0)
            c.z_approx[label].append(float(getattr(out, z_attr)))
            self_used = True
        if self_used:
            c.z_exact.append(z_ex)
            c.n_scans += 1
    print(f"Loaded {c.n_scans} scans ({skipped} skipped: enum-error / no-exact / unreadable)")
    return c.finalize()


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _survival(ax, values: np.ndarray, label: str, color: str):
    """P(err > x) vs x on a symlog-x / log-y axis."""
    v = np.sort(values[np.isfinite(values)])
    if v.size == 0:
        return
    surv = 1.0 - np.arange(v.size) / v.size
    ax.plot(v, surv, label=label, color=color, lw=1.6)


def _save(fig, out: Path, name: str):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


# --- A. Marginal accuracy --------------------------------------------------- #
def a1_survival(c: Collected, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, *_ , color in METHODS:
        _survival(ax, c.abs_err[label], label, color)
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_yscale("log")
    ax.set_xlabel("absolute marginal error")
    ax.set_ylabel(r"survival  $P(\mathrm{err} > x)$")
    ax.set_title("Marginal-error survival function (all methods vs exact)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, out, "A1_marginal_error_survival")


def a2_histograms(c: Collected, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    lo, hi = -0.5, 0.5
    sbins = np.linspace(lo, hi, 80)
    abins = np.linspace(0, 0.5, 60)
    for label, *_ , color in METHODS:
        r = c.raw_err[label]; r = r[np.isfinite(r)]
        a = c.abs_err[label]; a = a[np.isfinite(a)]
        if r.size:
            axes[0].hist(np.clip(r, lo, hi), bins=sbins, histtype="step", label=label, color=color)
        if a.size:
            axes[1].hist(np.clip(a, 0, 0.5), bins=abins, histtype="step", label=label, color=color)
    axes[0].set_title("Signed marginal error (exact - approx)")
    axes[0].set_xlabel("signed error")
    axes[1].set_title("Absolute marginal error")
    axes[1].set_xlabel("|error|")
    for ax in axes:
        ax.set_yscale("log"); ax.set_ylabel("count"); ax.legend()
    _save(fig, out, "A2_marginal_error_histograms")


def a3_components(c: Collected, out: Path):
    comps = [("misdetection", c.misdet_err), ("detection", c.det_err), ("nonexistence", c.nonexist_err)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (name, data) in zip(axes, comps):
        for label, *_ , color in METHODS:
            _survival(ax, data[label], label, color)
        ax.set_xscale("symlog", linthresh=1e-6)
        ax.set_yscale("log")
        ax.set_xlabel("|error|")
        ax.set_title(name)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"$P(\mathrm{err} > x)$")
    axes[-1].legend()
    fig.suptitle("Per-component marginal-error survival (misdetection / detection / nonexistence)")
    _save(fig, out, "A3_marginal_error_by_component")


def a4_heatmaps(c: Collected, out: Path):
    n = len(METHODS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2))
    bins = np.linspace(0, 1, 60)
    for ax, (label, *_ , _color) in zip(axes, METHODS):
        ex, ap = c.marg_exact[label], c.marg_approx[label]
        m = np.isfinite(ex) & np.isfinite(ap)
        h = ax.hist2d(ap[m], ex[m], bins=bins, norm=LogNorm(), cmap="Reds")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(label)
        ax.set_xlabel("approx marginal")
    axes[0].set_ylabel("exact marginal")
    fig.suptitle("Exact vs approximate marginal (2D histogram, LogNorm)")
    fig.colorbar(h[3], ax=axes.ravel().tolist(), shrink=0.8, label="count")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"A4_marginal_correlation_heatmaps.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved A4_marginal_correlation_heatmaps")


def a5_max_error(c: Collected, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [m[0] for m in METHODS]
    data = [c.max_err_per_scan[l][np.isfinite(c.max_err_per_scan[l])] for l in labels]
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, (*_ , color) in zip(bp["boxes"], METHODS):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    ax.set_ylabel("max per-track marginal error per scan")
    ax.set_title("Worst-case marginal error per scan")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out, "A5_max_marginal_error_per_scan")


# --- B. Normalization constant Z -------------------------------------------- #
def b1_scatter(c: Collected, out: Path):
    fig, ax = plt.subplots(figsize=(7, 7))
    zex = c.z_exact
    for label, *_ , color in METHODS:
        za = c.z_approx[label]
        m = np.isfinite(za) & np.isfinite(zex) & (za > 0) & (zex > 0)
        ax.scatter(zex[m], za[m], s=8, alpha=0.35, color=color, label=label)
    both = np.concatenate([zex[zex > 0]] + [c.z_approx[l][c.z_approx[l] > 0] for l, *_ in METHODS])
    lim = [both.min(), both.max()]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("exact Z"); ax.set_ylabel("approx Z")
    ax.set_title("Normalization constant: exact vs approx")
    ax.legend()
    _save(fig, out, "B1_normconst_scatter")


def _rel_z_err(c: Collected, signed: bool):
    out = {}
    zex = c.z_exact
    for label, *_ in METHODS:
        za = c.z_approx[label]
        m = np.isfinite(za) & np.isfinite(zex) & (zex > 0)
        e = (za[m] - zex[m]) / zex[m]
        out[label] = e if signed else np.abs(e)
    return out


def b2_relerror_box(c: Collected, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    err = _rel_z_err(c, signed=False)
    labels = [m[0] for m in METHODS]
    data = [err[l][err[l] > 0] for l in labels]
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, (*_ , color) in zip(bp["boxes"], METHODS):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    ax.set_yscale("log")
    ax.set_ylabel(r"relative $Z$ error  $|Z_a - Z_e| / Z_e$")
    ax.set_title("Normalization-constant relative error per method")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    _save(fig, out, "B2_normconst_relerror_box")


def b3_signed_hist(c: Collected, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    err = _rel_z_err(c, signed=True)
    bins = np.linspace(-1, 1, 80)
    for label, *_ , color in METHODS:
        e = err[label][np.isfinite(err[label])]
        if e.size:
            ax.hist(np.clip(e, -1, 1), bins=bins, histtype="step", label=label, color=color)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(r"signed relative $Z$ error  $(Z_a - Z_e)/Z_e$")
    ax.set_ylabel("count")
    ax.set_title("Normalization-constant over/under-estimation")
    ax.legend()
    _save(fig, out, "B3_normconst_signed_relerror_hist")


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stats-dir", default=STATS_DIR)
    p.add_argument("--out", default=None, help="figure output dir (default: <stats-dir>/figures)")
    p.add_argument("--limit", type=int, default=None, help="only load the first N stat files")
    args = p.parse_args(argv)

    files = sorted(glob(f"{args.stats_dir}/*_stats"))
    if not files:
        raise SystemExit(f"No *_stats files under {args.stats_dir}")
    if args.limit:
        files = files[: args.limit]
    out = Path(args.out or f"{args.stats_dir}/figures")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Reading {len(files)} stat files -> figures in {out}")

    c = collect(files)
    if c.n_scans == 0:
        raise SystemExit("No usable scans (all skipped).")

    # A. marginal accuracy
    a1_survival(c, out)
    a2_histograms(c, out)
    a3_components(c, out)
    a4_heatmaps(c, out)
    a5_max_error(c, out)
    # B. normalization constant Z
    b1_scatter(c, out)
    b2_relerror_box(c, out)
    b3_signed_hist(c, out)

    # quick numeric summary (matches the plots). Marginals are sparse (mostly 0/0),
    # so report mean + 99th-percentile |error| rather than a near-zero median.
    print("\nper method: median|relZ err|  mean|marg err|  p99|marg err|")
    relz = _rel_z_err(c, signed=False)
    for label, *_ in METHODS:
        rz = relz[label][np.isfinite(relz[label])]
        ae = c.abs_err[label][np.isfinite(c.abs_err[label])]
        mz = np.median(rz) if rz.size else np.nan
        print(f"  {label:9s}  relZ={mz:8.4g}   marg_mean={ae.mean():.3g}   marg_p99={np.percentile(ae,99):.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
