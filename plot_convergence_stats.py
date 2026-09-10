"""Plot marginal-accuracy + normalization-constant (Z) accuracy from the
``ravens_output_multicluster_convergence/`` stat files.

Each ``*_stats`` file is a pickled ``dfg_da.stats_logger.MulticlusterData`` with the
approximate methods + the exact reference (see the module docstring of
``ravens_parser_parallell_multicluster2.py``). For every scan we compare each
approximate method's per-track marginals and its normalization constant against
the exact EHM2 truth, then aggregate across scans.

Two families are overlaid on every figure: the five LBP-style methods (solid, ``C0``-``C4``)
and the Murty branch-and-bound baseline, which is stored as a ``{nHypoTotalMax: output}``
dict and is plotted once per truncation depth in ``murty_marginals.K_SWEEP`` (dashed, a
light-to-dark viridis ramp) so the sweep reads as one method converging with ``K``.

Figure families: ``A*`` marginal accuracy, ``B*`` normalization constant, ``C1`` Murty
truncation depth, and ``C2``-``C5`` cost -- accuracy bought per second, the per-scan runtime
distribution, how runtime scales with problem size, and cost relative to the exact solve.
Timing comes from the ``runtime`` stamped on each output plus the ``MulticlusterTimings``
record; scans written before those existed simply drop out of the ``C2``-``C5`` figures.

    .venv/bin/python -W ignore plot_convergence_stats.py [--limit N] [--out DIR] [--timed-only]

Reuses ``sl.Marginals`` / ``sl.MarginalsErrors`` (dfg_da/stats_logger.py) for the
error primitives; borrows the survival/heatmap styling from
``plotting_multicluster_memory_efficient_convergence.py`` + ``chapter10_plots.py``.
"""
from __future__ import annotations

import argparse
import pickle
from glob import glob
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import dfg_da.stats_logger as sl
from dfg_da.murty_marginals import K_SWEEP

STATS_DIR = "./ravens_output_multicluster_convergence"

# Costs recorded on MulticlusterTimings that belong to the harness rather than to any
# single method. Reported next to the per-method medians so the method timings can be
# read against the fixed overhead they sit on top of.
HARNESS_COSTS = ("parse", "cluster_links", "exact_theta_posteriors")


class Method(NamedTuple):
    """One plotted series. ``key`` is ``None`` for the methods stored as a single output
    object, and the ``nHypoTotalMax`` truncation depth for Murty, whose ``MulticlusterData``
    attribute holds a ``{K: output}`` dict."""

    label: str
    attr: str            # MulticlusterData attribute
    marg_attr: str       # marginals attribute on that output
    z_attr: str          # normalization-constant attribute on that output
    color: object        # matplotlib color ("C0" or an RGBA tuple)
    key: Optional[int] = None
    ls: str = "-"
    t_attr: str = "runtime"   # seconds; uniform across every output dataclass


# One shared method list + styles so every figure overlays the same methods in the same order.
_LBP_METHODS = [
    Method("MCMH-LBP", "mcmhlbp_output", "approx_marginals", "approx_normalization_constant", "C0"),
    Method("Bethe",    "mc_bethe_output", "marginals", "likelihood", "C1"),
    Method("MHLBP",    "mc_mhlbp_output", "marginals", "likelihood", "C2"),
    Method("PHD",      "mc_phd_output",   "marginals", "likelihood", "C3"),
    Method("IE",       "mc_lbp_ie_output", "marginals", "likelihood", "C4"),
]

# Murty depths come from the sweep constant rather than being hardcoded again, so the plots
# follow ``murty_sweep`` automatically if the sweep is ever changed.
MURTY_KS = tuple(K_SWEEP)
_MURTY_SHADES = matplotlib.colormaps["viridis"](np.linspace(0.15, 0.85, len(MURTY_KS)))

METHODS: List[Method] = _LBP_METHODS + [
    Method(f"Murty-{k}", "mc_murty_outputs", "marginals", "likelihood", color, int(k), "--")
    for k, color in zip(MURTY_KS, _MURTY_SHADES)
]


# --------------------------------------------------------------------------- #
# Load + reduce the stat files into flat per-method arrays
# --------------------------------------------------------------------------- #
class Collected:
    """Aggregated errors/values across all scans, keyed by method label."""

    def __init__(self):
        # marginal error arrays (concatenated over tracks x scans)
        self.abs_err: Dict[str, List[np.ndarray]] = {m.label: [] for m in METHODS}
        self.raw_err: Dict[str, List[np.ndarray]] = {m.label: [] for m in METHODS}
        self.misdet_err: Dict[str, List[np.ndarray]] = {m.label: [] for m in METHODS}
        self.det_err: Dict[str, List[np.ndarray]] = {m.label: [] for m in METHODS}
        self.nonexist_err: Dict[str, List[np.ndarray]] = {m.label: [] for m in METHODS}
        # exact vs approx marginal values (for the correlation heatmap)
        self.marg_exact: Dict[str, List[np.ndarray]] = {m.label: [] for m in METHODS}
        self.marg_approx: Dict[str, List[np.ndarray]] = {m.label: [] for m in METHODS}
        # per-scan max marginal error
        self.max_err_per_scan: Dict[str, List[float]] = {m.label: [] for m in METHODS}
        # normalization constants (per scan)
        self.z_exact: List[float] = []
        self.z_approx: Dict[str, List[float]] = {m.label: [] for m in METHODS}
        # Murty-only: global hypotheses actually retained, per truncation depth (per scan)
        self.murty_n_hypos: Dict[int, List[float]] = {int(k): [] for k in MURTY_KS}
        # wall-clock seconds per scan, per method, plus exact as the cost ceiling
        self.runtime_per_scan: Dict[str, List[float]] = {m.label: [] for m in METHODS}
        # solve-only seconds, self-timed inside compute_marginals_likelihood(); the gap to
        # runtime_per_scan is the method's construction cost
        self.solve_runtime_per_scan: Dict[str, List[float]] = {m.label: [] for m in METHODS}
        self.exact_runtime: List[float] = []
        # per-scan problem size, for the runtime-scaling figure
        self.n_tracks: List[float] = []
        self.n_superclusters: List[float] = []
        # harness costs charged to no method
        self.harness: Dict[str, List[float]] = {k: [] for k in HARNESS_COSTS}
        self.n_scans = 0
        self.n_timed = 0

    def finalize(self):
        cat = lambda d: {k: (np.concatenate(v) if v else np.array([])) for k, v in d.items()}
        self.abs_err = cat(self.abs_err)
        self.raw_err = cat(self.raw_err)
        self.misdet_err = cat(self.misdet_err)
        self.det_err = cat(self.det_err)
        self.nonexist_err = cat(self.nonexist_err)
        self.marg_exact = cat(self.marg_exact)
        self.marg_approx = cat(self.marg_approx)
        self.max_err_per_scan = {k: np.asarray(v, float) for k, v in self.max_err_per_scan.items()}
        self.z_exact = np.asarray(self.z_exact, float)
        self.z_approx = {k: np.asarray(v, float) for k, v in self.z_approx.items()}
        self.murty_n_hypos = {k: np.asarray(v, float) for k, v in self.murty_n_hypos.items()}
        self.runtime_per_scan = {k: np.asarray(v, float) for k, v in self.runtime_per_scan.items()}
        self.solve_runtime_per_scan = {k: np.asarray(v, float)
                                       for k, v in self.solve_runtime_per_scan.items()}
        self.exact_runtime = np.asarray(self.exact_runtime, float)
        self.n_tracks = np.asarray(self.n_tracks, float)
        self.n_superclusters = np.asarray(self.n_superclusters, float)
        self.harness = {k: np.asarray(v, float) for k, v in self.harness.items()}
        return self


def collect(files: List[str], timed_only: bool = False) -> Collected:
    """Reduce the stat files to flat per-method arrays.

    ``timed_only`` keeps only scans that carry a ``MulticlusterTimings`` record. The stats
    directory is written in place, one file per scan, so an interrupted run leaves a mix:
    scans it reached carry per-method timings, the rest are the previous run's output and
    carry none. Accuracy is unaffected by that mix, which is why the default is to use
    every scan; pass the flag to hold the accuracy figures to the same scans the timing
    figures can use.
    """
    c = Collected()
    skipped = 0
    for f in files:
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            # includes the half-written file an interrupted run leaves behind
            skipped += 1
            continue
        if getattr(d, "explicit_hypothesis_enumeration_error", False):
            skipped += 1
            continue
        timings = getattr(d, "timings", None)
        if timed_only and timings is None:
            skipped += 1
            continue
        exact_out = getattr(d, "exact_output", None)
        if exact_out is None:  # exact was skipped for this scan -> no truth
            skipped += 1
            continue
        exact_marg_arr = exact_out.exact_marginals
        exact_M = sl.Marginals(exact_marg_arr)
        z_ex = float(exact_out.exact_normalization_constant)

        # Per-scan values are buffered and only committed once we know the scan counts, so
        # that z_approx/max_err_per_scan stay index-aligned with z_exact even when a method
        # is missing for this scan (murty_sweep is guarded by try/except in the parser, so
        # mc_murty_outputs can legitimately be None).
        scan_max_err: Dict[str, float] = {}
        scan_z: Dict[str, float] = {}
        scan_n_hypos: Dict[int, float] = {}
        scan_runtime: Dict[str, float] = {}
        scan_solve: Dict[str, float] = {}
        self_used = False
        for m in METHODS:
            out = getattr(d, m.attr, None)
            if m.key is not None:
                out = out.get(m.key) if out else None
            if out is None:
                continue
            approx_arr = getattr(out, m.marg_attr)
            if approx_arr.shape != exact_marg_arr.shape:
                continue
            approx_M = sl.Marginals(approx_arr)
            err = sl.MarginalsErrors(exact_M, approx_M)
            c.abs_err[m.label].append(err.abs_errors)
            c.raw_err[m.label].append(err.raw_errors)
            c.misdet_err[m.label].append(err.misdetection_errors)
            c.det_err[m.label].append(err.detection_errors)
            c.nonexist_err[m.label].append(np.abs(exact_M.nonexistence_marginals - approx_M.nonexistence_marginals))
            c.marg_exact[m.label].append(exact_marg_arr.ravel())
            c.marg_approx[m.label].append(approx_arr.ravel())
            scan_max_err[m.label] = float(err.max_errors.max()) if err.max_errors.size else 0.0
            scan_z[m.label] = float(getattr(out, m.z_attr))
            # nan on stat files written before per-method timing existed
            scan_runtime[m.label] = float(getattr(out, m.t_attr, np.nan))
            scan_solve[m.label] = float(getattr(out, "solve_runtime", np.nan))
            if m.key is not None:
                scan_n_hypos[m.key] = float(getattr(out, "n_hypotheses", np.nan))
            self_used = True
        if self_used:
            # An interrupted run leaves the directory mixing two runs: the scans it reached
            # carry a MulticlusterTimings record, the rest are the previous run's output.
            # Those older scans are not simply untimed -- they still carry a runtime for
            # MCMH-LBP, Murty and exact while the conditioning-LBP methods are nan, and they
            # were measured before the BLAS pinning, under whatever thread oversubscription
            # the previous run had. Mixing them in would compare some methods over 1394 scans
            # and the rest over 734, under two different sets of conditions. So timing is
            # taken only from scans carrying the record; accuracy still uses every scan.
            timed = timings is not None
            for m in METHODS:
                c.max_err_per_scan[m.label].append(scan_max_err.get(m.label, np.nan))
                c.z_approx[m.label].append(scan_z.get(m.label, np.nan))
                c.runtime_per_scan[m.label].append(scan_runtime.get(m.label, np.nan) if timed else np.nan)
                c.solve_runtime_per_scan[m.label].append(scan_solve.get(m.label, np.nan) if timed else np.nan)
            for k in c.murty_n_hypos:
                c.murty_n_hypos[k].append(scan_n_hypos.get(k, np.nan))
            c.z_exact.append(z_ex)
            c.exact_runtime.append(float(getattr(exact_out, "runtime", np.nan)) if timed else np.nan)
            # Problem size, for the scaling figure. Tracks come from the truth marginals;
            # superclusters are what is left after assocLocal merges the prior clusters, which
            # is the count the conditioning methods and exact actually pay for.
            c.n_tracks.append(float(exact_marg_arr.shape[0]))
            c.n_superclusters.append(float(np.size(exact_out.normalization_constant_per_cluster)))
            for k in c.harness:
                c.harness[k].append(float(getattr(timings, k, np.nan)) if timed else np.nan)
            c.n_timed += int(timed)
            c.n_scans += 1
    print(f"Loaded {c.n_scans} scans, {c.n_timed} of them with per-method timings "
          f"({skipped} skipped: enum-error / no-exact / unreadable)")
    return c.finalize()


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _survival(ax, values: np.ndarray, m: Method):
    """P(err > x) vs x on a symlog-x / log-y axis."""
    v = np.sort(values[np.isfinite(values)])
    if v.size == 0:
        return
    surv = 1.0 - np.arange(v.size) / v.size
    ax.plot(v, surv, label=m.label, color=m.color, ls=m.ls, lw=1.6)


def _legend(ax):
    """Two columns: 10 overlaid series make a single-column legend eat the axes."""
    ax.legend(ncol=2, fontsize="small")


def _boxplot(ax, data, labels, colors):
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.tick_params(axis="x", rotation=30)
    return bp


def _save(fig, out: Path, name: str):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


# --- A. Marginal accuracy --------------------------------------------------- #
def a1_survival(c: Collected, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in METHODS:
        _survival(ax, c.abs_err[m.label], m)
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_yscale("log")
    ax.set_xlabel("absolute marginal error")
    ax.set_ylabel(r"survival  $P(\mathrm{err} > x)$")
    ax.set_title("Marginal-error survival function (all methods vs exact)")
    ax.grid(True, which="both", alpha=0.3)
    _legend(ax)
    _save(fig, out, "A1_marginal_error_survival")


def a2_histograms(c: Collected, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    lo, hi = -0.5, 0.5
    sbins = np.linspace(lo, hi, 80)
    abins = np.linspace(0, 0.5, 60)
    for m in METHODS:
        r = c.raw_err[m.label]; r = r[np.isfinite(r)]
        a = c.abs_err[m.label]; a = a[np.isfinite(a)]
        if r.size:
            axes[0].hist(np.clip(r, lo, hi), bins=sbins, histtype="step", label=m.label,
                         color=m.color, ls=m.ls)
        if a.size:
            axes[1].hist(np.clip(a, 0, 0.5), bins=abins, histtype="step", label=m.label,
                         color=m.color, ls=m.ls)
    axes[0].set_title("Signed marginal error (exact - approx)")
    axes[0].set_xlabel("signed error")
    axes[1].set_title("Absolute marginal error")
    axes[1].set_xlabel("|error|")
    for ax in axes:
        ax.set_yscale("log"); ax.set_ylabel("count"); _legend(ax)
    _save(fig, out, "A2_marginal_error_histograms")


def a3_components(c: Collected, out: Path):
    comps = [("misdetection", c.misdet_err), ("detection", c.det_err), ("nonexistence", c.nonexist_err)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (name, data) in zip(axes, comps):
        for m in METHODS:
            _survival(ax, data[m.label], m)
        ax.set_xscale("symlog", linthresh=1e-6)
        ax.set_yscale("log")
        ax.set_xlabel("|error|")
        ax.set_title(name)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"$P(\mathrm{err} > x)$")
    _legend(axes[-1])
    fig.suptitle("Per-component marginal-error survival (misdetection / detection / nonexistence)")
    _save(fig, out, "A3_marginal_error_by_component")


def a4_heatmaps(c: Collected, out: Path):
    # One panel per method: wrap into a grid rather than a single strip, which would be
    # 4 in x len(METHODS) = 40 in wide once the Murty sweep is included.
    n = len(METHODS)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.4 * nrows), squeeze=False)
    flat = axes.ravel()
    bins = np.linspace(0, 1, 60)
    h = None
    for ax, m in zip(flat, METHODS):
        ex, ap = c.marg_exact[m.label], c.marg_approx[m.label]
        mask = np.isfinite(ex) & np.isfinite(ap)
        h = ax.hist2d(ap[mask], ex[mask], bins=bins, norm=LogNorm(), cmap="Reds")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(m.label)
        ax.set_xlabel("approx marginal")
    for ax in flat[n:]:
        ax.set_axis_off()
    for r in range(nrows):
        axes[r, 0].set_ylabel("exact marginal")
    fig.suptitle("Exact vs approximate marginal (2D histogram, LogNorm)")
    fig.colorbar(h[3], ax=axes.ravel().tolist(), shrink=0.8, label="count")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"A4_marginal_correlation_heatmaps.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved A4_marginal_correlation_heatmaps")


def a5_max_error(c: Collected, out: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [m.label for m in METHODS]
    data = [c.max_err_per_scan[l][np.isfinite(c.max_err_per_scan[l])] for l in labels]
    _boxplot(ax, data, labels, [m.color for m in METHODS])
    ax.set_ylabel("max per-track marginal error per scan")
    ax.set_title("Worst-case marginal error per scan")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out, "A5_max_marginal_error_per_scan")


# --- B. Normalization constant Z -------------------------------------------- #
def b1_scatter(c: Collected, out: Path):
    fig, ax = plt.subplots(figsize=(7, 7))
    zex = c.z_exact
    for m in METHODS:
        za = c.z_approx[m.label]
        mask = np.isfinite(za) & np.isfinite(zex) & (za > 0) & (zex > 0)
        ax.scatter(zex[mask], za[mask], s=6, alpha=0.25, color=m.color, label=m.label)
    both = np.concatenate([zex[zex > 0]] + [c.z_approx[m.label][c.z_approx[m.label] > 0] for m in METHODS])
    lim = [both.min(), both.max()]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("exact Z"); ax.set_ylabel("approx Z")
    ax.set_title("Normalization constant: exact vs approx")
    _legend(ax)
    _save(fig, out, "B1_normconst_scatter")


def _rel_z_err(c: Collected, signed: bool):
    out = {}
    zex = c.z_exact
    for m in METHODS:
        za = c.z_approx[m.label]
        mask = np.isfinite(za) & np.isfinite(zex) & (zex > 0)
        e = (za[mask] - zex[mask]) / zex[mask]
        out[m.label] = e if signed else np.abs(e)
    return out


def b2_relerror_box(c: Collected, out: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    err = _rel_z_err(c, signed=False)
    labels = [m.label for m in METHODS]
    data = [err[l][err[l] > 0] for l in labels]
    _boxplot(ax, data, labels, [m.color for m in METHODS])
    ax.set_yscale("log")
    ax.set_ylabel(r"relative $Z$ error  $|Z_a - Z_e| / Z_e$")
    ax.set_title("Normalization-constant relative error per method")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    _save(fig, out, "B2_normconst_relerror_box")


def b3_signed_hist(c: Collected, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    err = _rel_z_err(c, signed=True)
    bins = np.linspace(-1, 1, 80)
    for m in METHODS:
        e = err[m.label][np.isfinite(err[m.label])]
        if e.size:
            ax.hist(np.clip(e, -1, 1), bins=bins, histtype="step", label=m.label,
                    color=m.color, ls=m.ls)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(r"signed relative $Z$ error  $(Z_a - Z_e)/Z_e$")
    ax.set_ylabel("count")
    ax.set_title("Normalization-constant over/under-estimation")
    _legend(ax)
    _save(fig, out, "B3_normconst_signed_relerror_hist")


# --- C. Murty-only diagnostics ---------------------------------------------- #
def c1_murty_hypotheses(c: Collected, out: Path):
    """How much of the hypothesis cloud each truncation depth actually retains.

    ``n_hypotheses`` is summed over superclusters, so it always exceeds the per-cluster cap
    ``K``; the informative quantities are how it grows with ``K`` (left) and the multiplier
    over the cap (right), which falls toward 1 once one supercluster dominates the cloud.
    """
    ks = sorted(c.murty_n_hypos)
    counts = [c.murty_n_hypos[k][np.isfinite(c.murty_n_hypos[k])] for k in ks]
    if not any(v.size for v in counts):
        print("  skipped C1_murty_hypotheses_kept (no Murty data)")
        return
    # Key the colors off the method rows so the ramp stays matched to K even if K_SWEEP
    # is ever given out of order.
    by_k = {m.key: m.color for m in METHODS if m.key is not None}
    colors = [by_k[k] for k in ks]
    labels = [str(k) for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _boxplot(axes[0], counts, labels, colors)
    axes[0].plot(np.arange(1, len(ks) + 1), ks, "k--o", lw=1, ms=4, label=r"cap $K$ (per cluster)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("global hypotheses kept (summed over superclusters)")
    axes[0].set_title("Hypotheses retained vs truncation depth")
    axes[0].legend(fontsize="small")

    ratios = [c.murty_n_hypos[k][np.isfinite(c.murty_n_hypos[k])] / k for k in ks]
    _boxplot(axes[1], ratios, labels, colors)
    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"hypotheses kept $/\ K$")
    axes[1].set_title("Multiplier over the per-cluster cap")

    for ax in axes:
        ax.set_xlabel(r"nHypoTotalMax  $K$")
        ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.suptitle("Murty: hypotheses actually kept per truncation depth")
    _save(fig, out, "C1_murty_hypotheses_kept")


def _median_runtimes(c: Collected):
    """(labels, colors, median seconds) for every method that has timing, exact last."""
    labels, colors, meds = [], [], []
    for m in METHODS:
        v = c.runtime_per_scan[m.label]
        v = v[np.isfinite(v)]
        if v.size:
            labels.append(m.label); colors.append(m.color); meds.append(float(np.median(v)))
    ex = c.exact_runtime[np.isfinite(c.exact_runtime)]
    if ex.size:
        labels.append("exact"); colors.append("0.25"); meds.append(float(np.median(ex)))
    return labels, colors, np.asarray(meds)


def c2_accuracy_vs_runtime(c: Collected, out: Path):
    """Cost against accuracy: what each extra second of compute actually buys."""
    labels, colors, meds = _median_runtimes(c)
    if meds.size == 0:
        print("  skipped C2_accuracy_vs_runtime (no timing data -- re-run the pipeline)")
        return
    err = [float(np.median(c.max_err_per_scan[l][np.isfinite(c.max_err_per_scan[l])]))
           if l != "exact" else 0.0 for l in labels]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for label, color, x, y in zip(labels, colors, meds, err):
        if label == "exact":  # the reference, not a competitor: a vertical cost ceiling
            ax.axvline(x, color=color, ls=":", lw=1.2)
            # axes-fraction y so the label does not depend on the data limits mid-draw
            ax.annotate("exact", xy=(x, 0.98), xycoords=("data", "axes fraction"),
                        color=color, fontsize=8, rotation=90, va="top", ha="right",
                        xytext=(-3, 0), textcoords="offset points")
            continue
        ax.scatter(x, y, s=70, color=color, zorder=3, edgecolor="white", linewidth=0.8)
        ax.annotate(label, (x, y), fontsize=8, color="0.25",
                    xytext=(7, 4), textcoords="offset points")
    # the Murty sweep is a trajectory, not five unrelated points -- join it
    mk = [(x, y) for l, x, y in zip(labels, meds, err) if l.startswith("Murty-")]
    if len(mk) > 1:
        ax.plot([p[0] for p in mk], [p[1] for p in mk], color=METHODS[-1].color,
                ls="--", lw=1.2, zorder=2, alpha=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    # headroom on the right so the trailing direct label is not clipped
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi * 1.6)
    ax.set_xlabel("median runtime per scan [s]")
    ax.set_ylabel("median worst-case marginal error per scan")
    ax.set_title("What a second of compute buys")
    ax.grid(True, which="both", alpha=0.25)
    _save(fig, out, "C2_accuracy_vs_runtime")


def c3_runtime_distribution(c: Collected, out: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    data, labels, colors = [], [], []
    for m in METHODS:
        v = c.runtime_per_scan[m.label]; v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            data.append(v); labels.append(m.label); colors.append(m.color)
    ex = c.exact_runtime[np.isfinite(c.exact_runtime) & (c.exact_runtime > 0)]
    if ex.size:
        data.append(ex); labels.append("exact"); colors.append("0.25")
    if not data:
        print("  skipped C3_runtime_distribution (no timing data -- re-run the pipeline)")
        plt.close(fig)
        return
    _boxplot(ax, data, labels, colors)
    ax.set_yscale("log")
    ax.set_ylabel("runtime per scan [s]")
    ax.set_title("Per-scan runtime by method")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    _save(fig, out, "C3_runtime_distribution")


def _binned_median(x: np.ndarray, y: np.ndarray, n_bins: int = 8):
    """Median of ``y`` within quantile bins of ``x``.

    Quantile edges rather than equal-width ones: both size measures are heavily
    right-skewed, so equal-width bins would put almost every scan in the first bin and
    leave the tail bins with too few scans to have a meaningful median.
    """
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[mask], y[mask]
    if x.size < 2 * n_bins:
        return np.array([]), np.array([])
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        return np.array([]), np.array([])
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, edges.size - 2)
    cx, cy = [], []
    for b in range(edges.size - 1):
        sel = idx == b
        if sel.sum() >= 3:  # a median over one or two scans is noise, not a trend
            cx.append(float(np.median(x[sel])))
            cy.append(float(np.median(y[sel])))
    return np.asarray(cx), np.asarray(cy)


def c4_runtime_scaling(c: Collected, out: Path):
    """Median runtime against the two things that drive it, on a log runtime axis.

    Per-scan runtime spans four orders of magnitude, so the flat distributions in C3 hide
    which methods stay cheap because the problems are small and which stay cheap because
    they scale well. Exact is drawn as the ceiling, not as a competitor.
    """
    # log x on the track panel so the slope reads as an exponent; linear on the
    # supercluster panel, whose integer range is too narrow for log ticks to land usefully.
    panels = [(c.n_tracks, "tracks in scan", "log"),
              (c.n_superclusters, "superclusters after merge", "linear")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    drawn = False
    for ax, (size, xlabel, xscale) in zip(axes, panels):
        for m in METHODS:
            bx, by = _binned_median(size, c.runtime_per_scan[m.label])
            if bx.size:
                ax.plot(bx, by, label=m.label, color=m.color, ls=m.ls, lw=1.6,
                        marker="o", ms=3.5)
                drawn = True
        bx, by = _binned_median(size, c.exact_runtime)
        if bx.size:
            ax.plot(bx, by, label="exact", color="0.25", ls=":", lw=2.2, marker="s", ms=4)
        ax.set_xscale(xscale)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(True, which="both", alpha=0.3)
    if not drawn:
        print("  skipped C4_runtime_scaling (no timing data -- re-run the pipeline)")
        plt.close(fig)
        return
    axes[0].set_ylabel("median runtime per scan [s]")
    _legend(axes[-1])
    fig.suptitle("How each method's runtime grows with the problem")
    _save(fig, out, "C4_runtime_scaling")


def c5_speedup_vs_exact(c: Collected, out: Path):
    """Per-scan cost relative to the exact solve on the same scan.

    A ratio per scan rather than a ratio of medians: the scans differ in size by orders of
    magnitude, so dividing the aggregates would let the largest scans decide the number.
    Above 1 the approximation is the cheaper option; below 1 it costs more than the truth
    it approximates.
    """
    ex = c.exact_runtime
    data, labels, colors = [], [], []
    for m in METHODS:
        v = c.runtime_per_scan[m.label]
        mask = np.isfinite(v) & np.isfinite(ex) & (v > 0) & (ex > 0)
        if mask.any():
            data.append(ex[mask] / v[mask])
            labels.append(m.label)
            colors.append(m.color)
    if not data:
        print("  skipped C5_speedup_vs_exact (no timing data -- re-run the pipeline)")
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    _boxplot(ax, data, labels, colors)
    ax.axhline(1.0, color="k", ls="--", lw=0.9)
    ax.set_yscale("log")
    ax.set_ylabel(r"speedup over exact  $t_\mathrm{exact} / t_\mathrm{method}$")
    ax.set_title("Cost relative to exact EHM2, per scan (above 1 = cheaper than exact)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    _save(fig, out, "C5_speedup_vs_exact")


def print_timing_summary(c: Collected):
    """Numeric companion to C3-C5, plus the harness costs the method timings sit on top of."""
    if c.n_timed == 0:
        print("\nno per-method timings in this set -- re-run the pipeline to record them")
        return
    ex = c.exact_runtime
    print(f"\nper method ({c.n_timed} timed scans): median runtime, "
          f"median speedup vs exact, median construction share")
    for m in METHODS:
        v = c.runtime_per_scan[m.label]
        fin = np.isfinite(v) & (v > 0)
        if not fin.any():
            continue
        sp = np.isfinite(ex) & (ex > 0) & fin
        sv = c.solve_runtime_per_scan[m.label]
        setup = np.isfinite(sv) & fin
        share = np.median((v[setup] - sv[setup]) / v[setup]) if setup.any() else np.nan
        speed = np.median(ex[sp] / v[sp]) if sp.any() else np.nan
        # MCMH-LBP and Murty are not self-timed, so they have no construction split
        share_s = f"{share:7.2%}" if np.isfinite(share) else f"{'-':>7s}"
        print(f"  {m.label:10s}  t={np.median(v[fin]):9.4g} s   "
              f"speedup={speed:8.3g}x   construction={share_s}")
    exf = ex[np.isfinite(ex) & (ex > 0)]
    if exf.size:
        print(f"  {'exact':10s}  t={np.median(exf):9.4g} s")
    parts = []
    for k in HARNESS_COSTS:
        h = c.harness[k][np.isfinite(c.harness[k])]
        if h.size:
            parts.append(f"{k}={np.median(h):.4g} s")
    if parts:
        print("  harness (charged to no method): " + ", ".join(parts))


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stats-dir", default=STATS_DIR)
    p.add_argument("--out", default=None, help="figure output dir (default: <stats-dir>/figures)")
    p.add_argument("--limit", type=int, default=None, help="only load the first N stat files")
    p.add_argument("--timed-only", action="store_true",
                   help="drop scans with no per-method timings, so every figure covers the "
                        "same scans (an interrupted run leaves the directory a mix of runs)")
    args = p.parse_args(argv)

    files = sorted(glob(f"{args.stats_dir}/*_stats"))
    if not files:
        raise SystemExit(f"No *_stats files under {args.stats_dir}")
    if args.limit:
        files = files[: args.limit]
    out = Path(args.out or f"{args.stats_dir}/figures")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Reading {len(files)} stat files -> figures in {out}")

    c = collect(files, timed_only=args.timed_only)
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
    # C. Murty-only diagnostics + cost
    c1_murty_hypotheses(c, out)
    c2_accuracy_vs_runtime(c, out)
    c3_runtime_distribution(c, out)
    c4_runtime_scaling(c, out)
    c5_speedup_vs_exact(c, out)

    # quick numeric summary (matches the plots). Marginals are sparse (mostly 0/0),
    # so report mean + 99th-percentile |error| rather than a near-zero median.
    print("\nper method: median|relZ err|  mean|marg err|  p99|marg err|")
    relz = _rel_z_err(c, signed=False)
    for m in METHODS:
        rz = relz[m.label][np.isfinite(relz[m.label])]
        ae = c.abs_err[m.label][np.isfinite(c.abs_err[m.label])]
        mz = np.median(rz) if rz.size else np.nan
        print(f"  {m.label:10s}  relZ={mz:8.4g}   marg_mean={ae.mean():.3g}   marg_p99={np.percentile(ae,99):.3g}")
    print_timing_summary(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
