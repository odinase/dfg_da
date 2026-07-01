"""Reproduce the Chapter 10 (Results) figures of the thesis on our data.

The thesis ("no.ntnu_inspera_140443607_46178819.pdf", Ch. 10) evaluates several
approximate marginal-association methods against the exact solution. We reproduce
the *plots* (not necessarily the same set of methods) using the approximate
methods we have data for:

  * ``MCMH-LBP``            -- multi-cluster multi-hypothesis LBP + Bethe
                              (``py_dfg_da.lbp.lbp_multicluster``); this is the
                              thesis's "MCMH-LBP" (no cluster conditioning).
  * ``Approx Efficient Bethe`` -- the hypothesis-conditioned, single-cluster
                              LBP+Bethe method run in the overlapping event space
                              with inclusion-exclusion (``cluster_partition``,
                              ``mode="overlap_ie"``); this is the thesis's
                              "Approximate Efficient Bethe".

Figures produced (under ``ravens_output_multicluster/chapter10/``), matching the
thesis figure numbers and styles:

  * ``fig10_1_normconst_boxplot`` + ``fig10_1_normconst_scatter``
        Fig 10.1 -- normalization-constant estimation accuracy.
        Boxplot of relative error (Z - Zhat)/Z on a symlog y-axis, and a
        log-log scatter of exact vs. approximate Z with a "perfect correlation"
        line.
  * ``fig10_2_marginal_heatmap``
        Fig 10.2 -- 2D-histogram correlation heatmap of approximate vs. exact
        track-association marginals (LogNorm colour, one panel per method).
  * ``fig10_3_survival_marginal_errors``
        Fig 10.3 -- survival function (1 - empirical CDF) of the absolute
        marginal errors, split into Max / Abs / Misdetection / Detection /
        Nonexistence, symlog x (linthresh 1e-6), log y.
  * ``fig10_4_theta_posterior_heatmap`` + ``fig10_5_theta_posterior_signed_error``
        Fig 10.4 / 10.5 -- prior-hypothesis-posterior Pr{theta | Z} accuracy for
        MCMH-LBP (correlation heatmap; signed-error histogram + boxplot). The IE
        method does not expose per-hypothesis posteriors, so only MCMH-LBP is
        shown here (which is the method the thesis discusses for this figure).

Run from the repo root::

    EVAL_WORKERS=8 pmbm-cm-python/.venv/bin/python -W ignore chapter10_plots.py
"""

import os
import time
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from multiprocessing import Pool

import py_dfg_da
import dfg_da.stats_logger as sl
from dfg_da.stats_logger import Marginals, MarginalsErrors
from dfg_da.marginal_association_Odin import ExplicitHypothesisEnumerationError
from cluster_partition import MulticlusterPartitionedMarginals

import ravens_parser_parallell_multicluster as ev

PMBM_DATA_PATH = ev.PMBM_DATA_PATH
MAX_ENUM_FOR_EXACT = ev.MAX_ENUM_FOR_EXACT
OUT_PATH = f"{ev.OUTPUT_PATH_BASE}/chapter10"

# (short id, thesis label, colour). Order controls plotting/legend order.
METHODS = [
    ("lbp", "MCMH-LBP", "C0"),
    ("ie", "Approx Efficient Bethe", "C3"),
]
# Methods for which we also have an aligned per-hypothesis posterior estimate.
THETA_METHODS = [("lbp", "MCMH-LBP", "C0")]


# ---------------------------------------------------------------------------
# Per-scan collection (parallel worker)
# ---------------------------------------------------------------------------

def loop_func(pmbm_file):
    mat_data = sl.MatFileParser(pmbm_file, use_cpp=True)
    R = np.asfortranarray(mat_data.reward_matrix_edmund)
    R_LC = np.asfortranarray(mat_data.reward_matrix_lc)
    ph = mat_data.prior_hypotheses_per_cluster
    if len(ph) == 0:
        return None
    assocLocal = mat_data.ws["assocLocal"].copy()

    enum_product, _ = ev.enumeration_product(R_LC, ph, assocLocal)
    if enum_product > MAX_ENUM_FOR_EXACT:
        return None  # no exact truth -> nothing to compare against

    cp = ev._to_cp_hypotheses(ph)

    # MCMH-LBP
    lbp = py_dfg_da.lbp.lbp_multicluster(R, ph)
    M_lbp = np.asarray(lbp.track_association_marginals()).T
    Z_lbp = float(lbp.bethe_pseudodual_normalization_constant())
    theta_lbp = np.concatenate([np.asarray(t, dtype=float).ravel()
                                for t in lbp.hypotheses_marginals()])

    # Approx Efficient Bethe (inclusion-exclusion)
    ie = MulticlusterPartitionedMarginals(
        R_LC, cp, ev.ie_lbp_bethe_solver, mode="overlap_ie", assocLocal=assocLocal)
    M_ie, _ie_theta, Z_ie = ie.compute_marginals_likelihood()
    M_ie = np.asarray(M_ie)
    Z_ie = float(Z_ie)

    # Exact truth
    try:
        exact = ev.exact_computer(R_LC, ph, assocLocal=assocLocal)
    except ExplicitHypothesisEnumerationError:
        return None
    M_ex = np.asarray(exact.exact_marginals)
    Z_ex = float(exact.exact_normalization_constant)
    theta_ex = np.concatenate([np.asarray(t, dtype=float).ravel()
                               for t in exact.compute_theta_posteriors()])

    if M_lbp.shape != M_ex.shape or M_ie.shape != M_ex.shape:
        return None

    exact_marg = Marginals(M_ex)
    rec = {
        "Z_exact": Z_ex, "Z_lbp": Z_lbp, "Z_ie": Z_ie,
        "cyclomatic": ev.cyclomatic_number(R_LC),
        # marginal pairs for the correlation heatmap (float32 to keep pickling cheap)
        "ex_marg": M_ex.ravel().astype(np.float32),
        "lbp_marg": M_lbp.ravel().astype(np.float32),
        "ie_marg": M_ie.ravel().astype(np.float32),
        # hypothesis posterior (MCMH-LBP only; aligned with exact)
        "ex_theta": theta_ex.astype(np.float32),
        "lbp_theta": theta_lbp.astype(np.float32),
    }
    # component marginal errors per method (MarginalsErrors: same convention as
    # the existing plotting scripts).
    for mid, M in (("lbp", M_lbp), ("ie", M_ie)):
        e = MarginalsErrors(exact_marg, Marginals(M))
        rec[f"{mid}_max"] = e.max_errors.astype(np.float32)
        rec[f"{mid}_abs"] = e.abs_errors.astype(np.float32)
        rec[f"{mid}_misdet"] = e.misdetection_errors.astype(np.float32)
        rec[f"{mid}_det"] = e.detection_errors.astype(np.float32)
        rec[f"{mid}_nonexist"] = e.nonexistence_errors.astype(np.float32)
    return rec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_fig(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(OUT_PATH) / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {Path(OUT_PATH) / name}.{{png,pdf}}")


def _cat(records, key):
    return np.concatenate([r[key] for r in records if r[key].size])


def _subsample_sorted(sorted_arr, n=20000):
    """Even index subsample of a sorted array, preserving endpoints/shape."""
    if sorted_arr.size <= n:
        return sorted_arr
    idx = np.unique(np.linspace(0, sorted_arr.size - 1, n).astype(int))
    return sorted_arr[idx]


# ---------------------------------------------------------------------------
# Fig 10.1 -- normalization constant estimation accuracy
# ---------------------------------------------------------------------------

def fig10_1(records):
    Zex = np.array([r["Z_exact"] for r in records])
    rel = {mid: (Zex - np.array([r[f"Z_{mid}"] for r in records])) / Zex
           for mid, _, _ in METHODS}

    # (a) boxplot of relative error, symlog y-axis
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([rel[mid] for mid, _, _ in METHODS], showfliers=True)
    ax.set_xticklabels([lbl for _, lbl, _ in METHODS], rotation=10)
    ax.set_ylabel(r"Relative error $(Z-\hat{Z})/Z$")
    ax.set_yscale("symlog")
    ax.grid(True, alpha=0.3)
    ax.set_title("Fig 10.1 (a): normalization-constant relative error")
    save_fig(fig, "fig10_1_normconst_boxplot")

    # (b) scatter of exact vs approximate Z, log-log, perfect-correlation line
    fig, ax = plt.subplots(figsize=(8, 6))
    for mid, lbl, color in METHODS:
        Zest = np.array([r[f"Z_{mid}"] for r in records])
        ok = (Zex > 0) & (Zest > 0)
        ax.plot(Zest[ok], Zex[ok], "o", ms=3, alpha=0.15, color=color, label=lbl)
    lim = [Zex[Zex > 0].min(), Zex.max()]
    ax.plot(lim, lim, "k--", lw=1, label="Perfect correlation")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Approximate normalization constant")
    ax.set_ylabel("Exact normalization constant")
    ax.grid(True, alpha=0.3)
    leg = ax.legend(loc="upper left")
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    ax.set_title("Fig 10.1 (b): exact vs. approximate $Z$")
    save_fig(fig, "fig10_1_normconst_scatter")


# ---------------------------------------------------------------------------
# Fig 10.2 -- marginal correlation heatmaps
# ---------------------------------------------------------------------------

def _corr_heatmap(ax, approx_flat, exact_flat, xlabel, nbins=200):
    edges = np.linspace(0, 1, nbins + 1)
    H, _, _ = np.histogram2d(approx_flat, exact_flat, bins=(edges, edges))
    H = np.ma.masked_where(H.T == 0, H.T)  # rows=exact, cols=approx
    vmax = H.max()
    mesh = ax.pcolormesh(edges, edges, H, norm=LogNorm(vmin=1, vmax=vmax),
                         cmap="Reds", shading="flat")
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Exact marginals")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    return mesh


def fig10_2(records):
    ex = _cat(records, "ex_marg")
    fig, axes = plt.subplots(1, len(METHODS), figsize=(6.5 * len(METHODS), 6))
    axes = np.atleast_1d(axes)
    for ax, (mid, lbl, _) in zip(axes.ravel(), METHODS):
        mesh = _corr_heatmap(ax, _cat(records, f"{mid}_marg"), ex, lbl)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Fig 10.2: approximate vs. exact track-association marginals")
    save_fig(fig, "fig10_2_marginal_heatmap")


# ---------------------------------------------------------------------------
# Fig 10.3 -- survival function of the marginal errors
# ---------------------------------------------------------------------------

def _plot_survival(ax, errors, label, color):
    e = np.sort(errors)
    if e.size == 0:
        return
    steps = np.linspace(1.0, 0.0, e.size)
    es, ss = _subsample_sorted(e), None
    # subsample steps with the same indices as the error array
    idx = np.unique(np.linspace(0, e.size - 1, min(e.size, 20000)).astype(int))
    ax.step(e[idx], steps[idx], where="post", label=label, color=color)


def fig10_3(records):
    comps = [("max", "Max errors"), ("abs", "Abs errors"),
             ("misdet", "Misdetection errors"), ("det", "Detection errors"),
             ("nonexist", "Nonexistence errors")]
    fig, axes = plt.subplots(len(comps), 1, figsize=(8, 13), sharex=True)
    linthresh = 1e-6
    for ax, (ckey, title) in zip(axes, comps):
        for mid, lbl, color in METHODS:
            _plot_survival(ax, _cat(records, f"{mid}_{ckey}"), lbl, color)
        ax.set_title(title)
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.2)
        ax.set_ylabel("survival")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("absolute marginal error (symlog, linthresh $10^{-6}$)")
    fig.suptitle("Fig 10.3: survival function of marginal errors")
    save_fig(fig, "fig10_3_survival_marginal_errors")


# ---------------------------------------------------------------------------
# Fig 10.4 / 10.5 -- prior hypothesis posterior Pr{theta | Z} (MCMH-LBP)
# ---------------------------------------------------------------------------

def fig10_4(records):
    ex = _cat(records, "ex_theta")
    fig, axes = plt.subplots(1, len(THETA_METHODS),
                             figsize=(6.5 * len(THETA_METHODS), 6), squeeze=False)
    for ax, (mid, lbl, _) in zip(axes.ravel(), THETA_METHODS):
        mesh = _corr_heatmap(ax, _cat(records, f"{mid}_theta"), ex, lbl)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(r"Fig 10.4: prior hypothesis posterior $\Pr\{\theta\,|\,Z\}$")
    save_fig(fig, "fig10_4_theta_posterior_heatmap")


def fig10_5(records):
    ex = _cat(records, "ex_theta")
    fig, (axh, axb) = plt.subplots(1, 2, figsize=(13, 5))
    signed = {}
    for mid, lbl, color in THETA_METHODS:
        est = _cat(records, f"{mid}_theta")
        s = ex - est  # signed error (true - estimate)
        signed[lbl] = s
        m = np.abs(s).max() or 1.0
        axh.hist(s, bins=np.linspace(-m, m, 80), alpha=0.6, color=color,
                 label=f"{lbl} (mean {s.mean():+.2e})")
    axh.set_yscale("log")
    axh.set_xlabel(r"signed error $\Pr_\mathrm{true}-\Pr_\mathrm{est}$")
    axh.set_ylabel("count")
    axh.set_title("Fig 10.5 (a): hypothesis-posterior signed error")
    axh.legend()
    axb.boxplot(list(signed.values()), showfliers=True)
    axb.set_xticklabels(list(signed.keys()), rotation=10)
    axb.set_ylabel("signed error")
    axb.set_yscale("symlog", linthresh=1e-3)
    axb.grid(True, alpha=0.3)
    axb.set_title("Fig 10.5 (b): signed-error distribution")
    save_fig(fig, "fig10_5_theta_posterior_signed_error")


# ---------------------------------------------------------------------------

def _binned_median(x, y, nbins=12):
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
    return map(np.array, (cx, med, q25, q75))


def fig_cyclomatic(records):
    """Accuracy vs. the cyclomatic number (loopiness) of the LBP graph.

    LBP is exact on a forest (mu = 0); this shows how the marginal error and the
    normalization-constant error grow with the number of independent cycles mu.
    """
    mu = np.array([r["cyclomatic"] for r in records], dtype=float)
    Zex = np.array([r["Z_exact"] for r in records])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # (a) per-scan max marginal error vs mu
    for mid, lbl, color in METHODS:
        ymax = np.array([r[f"{mid}_max"].max() if r[f"{mid}_max"].size else np.nan
                         for r in records])
        ok = np.isfinite(ymax)
        axes[0].scatter(mu[ok], ymax[ok], s=8, alpha=0.15, color=color)
        cx, med, q25, q75 = _binned_median(mu[ok], ymax[ok])
        axes[0].plot(cx, med, "-o", ms=4, color=color, label=f"{lbl} (median)")
        axes[0].fill_between(cx, q25, q75, color=color, alpha=0.18)
    axes[0].set_ylabel("max marginal error per scan")
    # (b) relative Z error vs mu
    for mid, lbl, color in METHODS:
        rel = np.abs(Zex - np.array([r[f"Z_{mid}"] for r in records])) / Zex
        ok = np.isfinite(rel) & (rel > 0)
        axes[1].scatter(mu[ok], rel[ok], s=8, alpha=0.15, color=color)
        cx, med, q25, q75 = _binned_median(mu[ok], rel[ok])
        axes[1].plot(cx, med, "-o", ms=4, color=color, label=f"{lbl} (median)")
        axes[1].fill_between(cx, q25, q75, color=color, alpha=0.18)
    axes[1].set_ylabel(r"relative $Z$ error $|Z-\hat Z|/Z$")
    for ax in axes:
        ax.set_xlabel(r"cyclomatic number $\mu = E - V + C$ of the LBP graph")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Accuracy vs. cyclomatic number (loopiness) of the LBP association graph")
    save_fig(fig, "fig_accuracy_vs_cyclomatic")


def summarize(records, total_s):
    Zex = np.array([r["Z_exact"] for r in records])
    print("\n" + "=" * 70)
    print(f"Chapter-10 stats over {len(records)} scans (exact truth) "
          f"in {total_s:.1f} s ({total_s/60:.2f} min)")
    for mid, lbl, _ in METHODS:
        rel = (Zex - np.array([r[f"Z_{mid}"] for r in records])) / Zex
        absm = _cat(records, f"{mid}_abs")
        print(f"  {lbl:24s}: relZ median={np.median(rel):+.2e} "
              f"IQR=[{np.percentile(rel,25):+.2e},{np.percentile(rel,75):+.2e}] "
              f"| |marg err| mean={absm.mean():.2e} max={absm.max():.2e}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    Path(OUT_PATH).mkdir(parents=True, exist_ok=True)
    pmbm_files = sorted(
        glob(PMBM_DATA_PATH + "/*.mat"),
        key=lambda p: int("".join(c for c in Path(p).stem if c.isdigit()) or 0))
    if not pmbm_files:
        raise ValueError(f"No .mat files found under {PMBM_DATA_PATH}")

    n_workers = int(os.environ.get("EVAL_WORKERS", "8"))
    print(f"Collecting Chapter-10 data for {len(pmbm_files)} files "
          f"with {n_workers} workers...")
    start = time.time()
    records = []
    with Pool(processes=n_workers) as pool:
        for r in tqdm(pool.imap_unordered(loop_func, pmbm_files, chunksize=1),
                      total=len(pmbm_files)):
            if r is not None:
                records.append(r)
    total_s = time.time() - start
    if not records:
        raise SystemExit("No scans with exact truth.")

    summarize(records, total_s)
    fig10_1(records)
    fig10_2(records)
    fig10_3(records)
    fig10_4(records)
    fig10_5(records)
    fig_cyclomatic(records)
    print(f"Done in {total_s:.1f} s. Figures in {OUT_PATH}/")
