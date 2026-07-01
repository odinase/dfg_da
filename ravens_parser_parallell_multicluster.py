import dfg_da.marginals_computers as mc
import dfg_da.prior_hypothesis as phs
import dfg_da.cluster_bayes_tree as cbt
import dfg_da.stats_logger as sl
from dfg_da.marginal_association_Odin import ExplicitHypothesisEnumerationError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glob import glob
from typing import List
import numpy as np
from tqdm import tqdm

from collections import Counter
from multiprocessing import Pool, Lock
import csv
import time
from pathlib import Path

import networkx as nx
import py_dfg_da

# Inclusion-exclusion (overlapping event space, thesis Sec. 7.5.1) marginalization
# with a single-cluster LBP+Bethe inner solver: the "efficient / linking-measurement"
# method run with LBP+Bethe and de-biased by the signed inclusion-exclusion weights.
from cluster_partition import (
    MulticlusterPartitionedMarginals,
    Hypothesis as CPHypothesis,
    Hypotheses as CPHypotheses,
    HypothesesList as CPHypothesesList,
)
from cluster_partition.solvers import _assemble_multihypothesis, _single_hyp_exact

OUTPUT_PATH_BASE = "./ravens_output_multicluster"
PMBM_DATA_PATH = "./data/pmbm_output_files"
METRICS_PATH = f"{OUTPUT_PATH_BASE}/metrics"

# Number of files we expect for the "9 ravens" dataset. Used only as a soft
# sanity check (the old code hard-required exactly 10_000 and crashed otherwise).
EXPECTED_NUM_FILES = 1397

# Safety cap: the exact/efficient conditioning enumeration is
# ``prod(1 + #clusters per linking measurement)``. With the supercluster
# balancing in place this is normally <= ~1e7, but to guarantee the full run
# finishes we skip the exact/efficient solvers for any file whose enumeration
# exceeds this (LBP is still timed). Such files are reported at the end.
MAX_ENUM_FOR_EXACT = 5e7

exact_computer = mc.MulticlusterExactEHM2()


class CppLBPBetheSolver:
    """``cluster_partition`` inner-solver contract backed by the compiled
    ``py_dfg_da`` LBP + Bethe pseudodual, applied **per single hypothesis**.

    Call contract (identical to ``cluster_partition.LBPBetheSolver``)::

        solver(R_cluster, prior_hypotheses, enforce_meas=(), reindex=True)
            -> (marginals (n, m+2), theta_posterior, likelihood)

    Structure mirrors the pure-Python ``LBPBetheSolver``: for each prior
    hypothesis we run a *single-hypothesis, single-cluster* LBP+Bethe on that
    hypothesis's existing-track sub-matrix and combine the per-hypothesis
    constants by total probability (thesis Eq. (7.16)) -- assembled by the
    shared ``_assemble_multihypothesis``.  Only the single-hypothesis kernel is
    swapped for the compiled ``lbp_single_cluster`` (invoked with a one-element
    hypothesis list so it reduces to the single-hypothesis JPDA LBP).

    ``R_cluster`` is the track-oriented "lc" layout (col 0 misdetection, cols
    1..m measurements); the compiled LBP consumes the "edmund" layout
    ``[meas (m) | n x n existence block]`` (per-track misdetection on the
    existence-block diagonal, ``-inf`` off it -- verified bit-exact against
    ``MatFileParser.reward_matrix_edmund``).
    """

    def __init__(self, max_num_iters: int = 300):
        self.max_num_iters = max_num_iters

    def _single_hyp(self, R_sub, enforce):
        # R_sub: (n_e, m+1) lc sub-matrix for ONE hypothesis's existing tracks.
        if enforce:
            # The overlapping event space never enforces a detection; defer to
            # the exact primitive for correctness if a caller ever does.
            return _single_hyp_exact(R_sub, enforce)
        n_e, mp1 = R_sub.shape
        if n_e == 0:
            return np.zeros((0, mp1)), 1.0
        exist = np.full((n_e, n_e), -np.inf)
        np.fill_diagonal(exist, R_sub[:, 0])
        R_edmund = np.asfortranarray(np.hstack((R_sub[:, 1:], exist)))
        one_hyp = py_dfg_da.hypothesis.Hypotheses([
            py_dfg_da.hypothesis.Hypothesis(list(range(1, n_e + 1)), 0.0)])
        out = py_dfg_da.lbp.lbp_single_cluster(R_edmund, one_hyp, self.max_num_iters)
        Z = float(out.bethe_pseudodual_normalization_constant())
        # C++ marginals are (m+2, n_e) [misdetect, meas_1..m, nonexist]; keep the
        # [misdetect, meas] block, renormalize per track, return unnormalized
        # (marg * Z) as ``_assemble_multihypothesis`` expects.
        tam = np.asarray(out.track_association_marginals()).T  # (n_e, m+2)
        marg = tam[:, :mp1].copy()
        s = marg.sum(axis=1, keepdims=True)
        marg = np.divide(marg, s, out=np.zeros_like(marg), where=s > 0)
        return marg * Z, Z

    def __call__(self, R_cluster, prior_hypotheses, enforce_meas=(), reindex=True):
        return _assemble_multihypothesis(
            R_cluster, prior_hypotheses, self._single_hyp, enforce_meas, reindex)


# Stateless single-cluster solver reused across scans (mirrors how
# ``exact_computer`` is a module-level singleton).
ie_lbp_bethe_solver = CppLBPBetheSolver()


def _to_cp_hypotheses(prior_hypotheses_per_cluster):
    """Convert ``py_dfg_da`` prior hypotheses to ``cluster_partition`` ones.

    Must be called *before* any upstream solver runs: the upstream solvers call
    ``reindex_tracks()`` which mutates the (compiled) hypotheses in place,
    relabelling global track ids to per-cluster locals and corrupting the
    global-id bookkeeping the inclusion-exclusion solver relies on.
    """
    out = CPHypothesesList()
    for cluster in prior_hypotheses_per_cluster:
        out.append(CPHypotheses([
            CPHypothesis(list(map(int, h.tracks())), float(h.log_prob()))
            for h in cluster
        ]))
    return out


NUM_DONE = 0


def merge_clusters(assocLocal, prior_hypotheses_per_cluster):
    num_posterior_clusters = np.sum(assocLocal[1])
    # First build master array
    prior_hypotheses_per_cluster_posterior: py_dfg_da.hypothesis.HypothesesList = py_dfg_da.hypothesis.HypothesesList([
        h for k, h in enumerate(prior_hypotheses_per_cluster) if assocLocal[1, k]
    ])
    master_idxs = np.cumsum(assocLocal[1]) - 1

    assert len(prior_hypotheses_per_cluster_posterior) == num_posterior_clusters

    for c, (master, is_master) in enumerate(assocLocal.T):
        if is_master:
            continue

        # We already have the masters, merge clusters
        hs = prior_hypotheses_per_cluster[c]
        prior_hypotheses_per_cluster_posterior[master_idxs[master]] = prior_hypotheses_per_cluster_posterior[master_idxs[master]].combine(hs)

    return prior_hypotheses_per_cluster_posterior


def cyclomatic_number(R_LC):
    """Cyclomatic number (circuit rank / first Betti number) ``mu = E - V + C``
    of the bipartite track<->measurement association graph that LBP runs its
    message passing on.

    Vertices are the tracks and the measurements; there is an edge
    ``(track i, measurement j)`` whenever ``R_LC[i, j]`` (for a measurement
    column ``j >= 1``) is finite, i.e. that association is feasible. ``mu`` is
    the number of independent cycles (https://en.wikipedia.org/wiki/Cyclomatic_number):
    ``mu = 0`` iff the graph is a forest, for which LBP is exact; larger ``mu``
    means more loops and is expected to correlate with larger LBP error.

    Isolated vertices (tracks whose only feasible event is misdetection) do not
    change ``mu`` (they add one to both ``V`` and ``C``), so the graph is built
    from the feasible track-measurement edges only.
    """
    meas = np.asarray(R_LC)[:, 1:]
    track_idx, meas_idx = np.where(np.isfinite(meas))
    if track_idx.size == 0:
        return 0
    G = nx.Graph()
    G.add_edges_from(zip((f"t{i}" for i in track_idx),
                         (f"m{j}" for j in meas_idx)))
    return int(G.number_of_edges() - G.number_of_nodes()
               + nx.number_connected_components(G))


def enumeration_product(R_LC, prior_hypotheses_per_cluster, assocLocal):
    """Total exact-conditioning enumeration size for this scan:
    ``prod over superclusters of prod(1 + #clusters per linking measurement)``."""
    cluster_links = cbt.ClusterLinks(
        R_LC=R_LC, prior_hypotheses_per_cluster=prior_hypotheses_per_cluster,
        assocLocal=assocLocal)
    total = 1
    n_linking = 0
    for lmaps in cluster_links.linking_mappings_per_merging_clusters():
        for _meas, clusters in lmaps.linking_measurements_to_clusters.items():
            total *= (1 + len(clusters))
            n_linking += 1
    return total, n_linking


def loop_func(pmbm_file):
    """Run LBP + (when tractable) exact EHM2 + efficient marginals + the
    inclusion-exclusion LBP+Bethe method on one scan, returning a metrics dict
    (or ``None`` if the scan has no clusters)."""
    mat_data: sl.MatFileParser = sl.MatFileParser(pmbm_file, use_cpp=True)

    R = np.asfortranarray(mat_data.reward_matrix_edmund)
    R_LC = np.asfortranarray(mat_data.reward_matrix_lc)
    prior_hypotheses_per_cluster = mat_data.prior_hypotheses_per_cluster

    pmbm_file_path = Path(pmbm_file)
    pmbm_filename = pmbm_file_path.name[:-len(pmbm_file_path.suffix)]

    num_clusters = len(prior_hypotheses_per_cluster)
    if num_clusters == 0:
        return None

    # Snapshot the prior hypotheses for the inclusion-exclusion solver BEFORE any
    # upstream solver reindexes (and thereby mutates) them in place.
    cp_prior_hypotheses = _to_cp_hypotheses(prior_hypotheses_per_cluster)

    assocLocal = mat_data.ws["assocLocal"].copy()

    # --- Supercluster topology metrics ---------------------------------------
    n_superclusters = int(np.sum(assocLocal[1]))
    group_sizes = Counter(assocLocal[0].astype(int).tolist())
    max_clusters_per_supercluster = max(group_sizes.values()) if group_sizes else 0
    n_tracks = int(R_LC.shape[0])
    n_measurements = int(R_LC.shape[1] - 1)
    enum_product, n_linking_measurements = enumeration_product(
        R_LC, prior_hypotheses_per_cluster, assocLocal)
    cyclomatic = cyclomatic_number(R_LC)

    metrics = {
        "file": pmbm_filename,
        "num_clusters": num_clusters,
        "n_superclusters": n_superclusters,
        "max_clusters_per_supercluster": max_clusters_per_supercluster,
        "n_tracks": n_tracks,
        "n_measurements": n_measurements,
        "n_linking_measurements": n_linking_measurements,
        "enum_product": enum_product,
        "cyclomatic": cyclomatic,
        "t_lbp": np.nan,
        "t_exact": np.nan,
        "t_efficient": np.nan,
        "t_ie_lbp_bethe": np.nan,
        "like_ie_lbp_bethe": np.nan,
        "exact_enum_error": False,
        "skipped_heavy": False,
        "marg_match": None,
        "like_match": None,
    }

    # --- LBP (always run) ----------------------------------------------------
    start = time.time()
    mcmhlbp = py_dfg_da.lbp.lbp_multicluster(R, prior_hypotheses_per_cluster)
    metrics["t_lbp"] = time.time() - start

    # --- Skip the heavy solvers if the enumeration is too large --------------
    if enum_product > MAX_ENUM_FOR_EXACT:
        metrics["skipped_heavy"] = True
        return metrics

    # --- Exact EHM2 ----------------------------------------------------------
    exact_output = None
    try:
        start = time.time()
        exact_output: mc.MulticlusterExactOutput = exact_computer(
            R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal)
        metrics["t_exact"] = time.time() - start
    except ExplicitHypothesisEnumerationError:
        metrics["exact_enum_error"] = True

    # --- Efficient (cluster-Bayes-tree) marginals ----------------------------
    start = time.time()
    exact_efficient: cbt.MulticlusterEfficientMarginals = cbt.MulticlusterEfficientMarginals(
        R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal)
    efficient_marginals, _efficient_theta_posteriors, efficient_likelihood = \
        exact_efficient.compute_marginals_likelihood()
    metrics["t_efficient"] = time.time() - start

    # --- Inclusion-exclusion (overlapping event space) + LBP+Bethe -----------
    # Same linking-measurement conditioning as the efficient solver, but with the
    # overlapping event space (Eq. (7.29)) de-biased by the signed inclusion-
    # exclusion weights, and an LBP+Bethe single-cluster inner solver.
    start = time.time()
    ie_lbp = MulticlusterPartitionedMarginals(
        R_LC, cp_prior_hypotheses, ie_lbp_bethe_solver,
        mode="overlap_ie", assocLocal=assocLocal)
    _ie_marg, _ie_theta, ie_like = ie_lbp.compute_marginals_likelihood()
    metrics["t_ie_lbp_bethe"] = time.time() - start
    metrics["like_ie_lbp_bethe"] = float(ie_like)

    if exact_output is not None:
        metrics["marg_match"] = bool(
            np.allclose(exact_output.exact_marginals, efficient_marginals))
        metrics["like_match"] = bool(
            np.isclose(exact_output.exact_normalization_constant, efficient_likelihood))
        if not metrics["marg_match"]:
            raise ValueError(f"Incorrect marginals at {pmbm_file_path}!")
        if not metrics["like_match"]:
            raise ValueError(f"Incorrect likelihood at {pmbm_file_path}!")

    return metrics


# ---------------------------------------------------------------------------
# Metrics aggregation + plotting (matplotlib + numpy only)
# ---------------------------------------------------------------------------

def _save_metrics_csv(records, out_dir):
    csv_path = Path(out_dir) / "metrics.csv"
    fields = ["file", "num_clusters", "n_superclusters",
              "max_clusters_per_supercluster", "n_tracks", "n_measurements",
              "n_linking_measurements", "enum_product", "cyclomatic", "t_lbp",
              "t_exact", "t_efficient", "t_ie_lbp_bethe", "like_ie_lbp_bethe",
              "exact_enum_error", "skipped_heavy", "marg_match",
              "like_match"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"Wrote {csv_path}")


def _runtime_histogram(records, out_dir):
    methods = [("t_lbp", "LBP"), ("t_exact", "Exact EHM2"),
               ("t_efficient", "Efficient marginals"),
               ("t_ie_lbp_bethe", "IE LBP+Bethe")]
    fig, ax = plt.subplots(figsize=(8, 5))
    # Common log-spaced bins across all finite, positive timings.
    all_t = np.concatenate([
        np.array([r[k] for r in records], dtype=float) for k, _ in methods])
    all_t = all_t[np.isfinite(all_t) & (all_t > 0)]
    if all_t.size == 0:
        plt.close(fig)
        return
    bins = np.logspace(np.log10(all_t.min()), np.log10(all_t.max()), 40)
    for key, label in methods:
        t = np.array([r[key] for r in records], dtype=float)
        t = t[np.isfinite(t) & (t > 0)]
        if t.size:
            ax.hist(t, bins=bins, alpha=0.55,
                    label=f"{label} (median {np.median(t)*1e3:.1f} ms, max {t.max():.2f} s)")
    ax.set_xscale("log")
    ax.set_xlabel("runtime per scan [s]")
    ax.set_ylabel("number of scans")
    ax.set_title("Per-method runtime distribution")
    ax.legend()
    fig.tight_layout()
    p = Path(out_dir) / "runtime_histogram.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")


def _metric_histograms(records, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    enum = np.array([r["enum_product"] for r in records], dtype=float)
    enum = enum[enum > 0]
    if enum.size:
        bins = np.logspace(0, np.log10(enum.max()), 40)
        axes[0, 0].hist(enum, bins=bins, color="C3")
        axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel(r"enumeration size  $\prod(1+\#$clusters per linking meas$)$")
    axes[0, 0].set_ylabel("number of scans")
    axes[0, 0].set_title("Exact-conditioning enumeration size")

    msz = np.array([r["max_clusters_per_supercluster"] for r in records], dtype=int)
    if msz.size:
        axes[0, 1].hist(msz, bins=np.arange(0.5, msz.max() + 1.5, 1.0), color="C0")
    axes[0, 1].set_xlabel("max clusters in any supercluster")
    axes[0, 1].set_ylabel("number of scans")
    axes[0, 1].set_title("Largest supercluster per scan")

    nsuper = np.array([r["n_superclusters"] for r in records], dtype=int)
    if nsuper.size:
        axes[1, 0].hist(nsuper, bins=np.arange(0.5, nsuper.max() + 1.5, 1.0), color="C2")
    axes[1, 0].set_xlabel("number of superclusters")
    axes[1, 0].set_ylabel("number of scans")
    axes[1, 0].set_title("Superclusters per scan")

    nlink = np.array([r["n_linking_measurements"] for r in records], dtype=int)
    if nlink.size:
        axes[1, 1].hist(nlink, bins=np.arange(-0.5, nlink.max() + 1.5, 1.0), color="C4")
    axes[1, 1].set_xlabel("number of linking measurements")
    axes[1, 1].set_ylabel("number of scans")
    axes[1, 1].set_title("Linking measurements per scan")

    fig.tight_layout()
    p = Path(out_dir) / "metric_histograms.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")


def _runtime_vs_enum(records, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    enum = np.array([r["enum_product"] for r in records], dtype=float)
    for key, label, color in [("t_exact", "Exact EHM2", "C1"),
                              ("t_efficient", "Efficient marginals", "C2"),
                              ("t_lbp", "LBP", "C0"),
                              ("t_ie_lbp_bethe", "IE LBP+Bethe", "C5")]:
        t = np.array([r[key] for r in records], dtype=float)
        ok = np.isfinite(t) & (t > 0) & (enum > 0)
        if ok.any():
            ax.scatter(enum[ok], t[ok], s=8, alpha=0.4, label=label, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("enumeration size")
    ax.set_ylabel("runtime per scan [s]")
    ax.set_title("Runtime vs. enumeration size")
    ax.legend()
    fig.tight_layout()
    p = Path(out_dir) / "runtime_vs_enum.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")


def summarize_and_plot(records, total_runtime_s, out_dir=METRICS_PATH):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(records)
    print("\n" + "=" * 64)
    print(f"Processed {n} non-empty scans in {total_runtime_s:.1f} s "
          f"({total_runtime_s/60:.2f} min)")

    def stat(key):
        a = np.array([r[key] for r in records], dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return "n/a"
        return (f"n={a.size:4d}  median={np.median(a)*1e3:8.2f} ms  "
                f"mean={a.mean()*1e3:8.2f} ms  max={a.max():7.3f} s  "
                f"total={a.sum():7.2f} s")

    print(f"  LBP                 : {stat('t_lbp')}")
    print(f"  Exact EHM2          : {stat('t_exact')}")
    print(f"  Efficient marginals : {stat('t_efficient')}")
    print(f"  IE LBP+Bethe        : {stat('t_ie_lbp_bethe')}")

    n_skipped = sum(1 for r in records if r["skipped_heavy"])
    n_enum_err = sum(1 for r in records if r["exact_enum_error"])
    matched = [r for r in records if r["marg_match"] is not None]
    n_marg_ok = sum(1 for r in matched if r["marg_match"] and r["like_match"])
    print(f"  Heavy-solver skips (enum > {MAX_ENUM_FOR_EXACT:.0e}): {n_skipped}")
    print(f"  Exact enumeration errors                : {n_enum_err}")
    print(f"  Exact/efficient agreement               : {n_marg_ok}/{len(matched)} scans")
    print("=" * 64 + "\n")

    _save_metrics_csv(records, out_dir)
    _runtime_histogram(records, out_dir)
    _metric_histograms(records, out_dir)
    _runtime_vs_enum(records, out_dir)


if __name__ == "__main__":
    pmbm_files = sorted(
        glob(PMBM_DATA_PATH + "/*.mat"),
        key=lambda p: int("".join(c for c in Path(p).stem if c.isdigit()) or 0))
    num_files = len(pmbm_files)

    if num_files == 0:
        raise ValueError(f"No .mat files found under {PMBM_DATA_PATH}")
    if num_files != EXPECTED_NUM_FILES:
        print(f"WARNING: found {num_files} files (expected {EXPECTED_NUM_FILES}).")

    import os
    n_workers = int(os.environ.get("EVAL_WORKERS", "8"))
    print(f"Computing {num_files} files with {n_workers} workers...")
    start = time.time()
    records = []
    with Pool(processes=n_workers) as pool:
        for m in tqdm(pool.imap_unordered(loop_func, pmbm_files, chunksize=1),
                      total=num_files):
            if m is not None:
                records.append(m)
    total_runtime_s = time.time() - start
    print("Done")

    summarize_and_plot(records, total_runtime_s)
    print(f"Spent {total_runtime_s:.3f} s = {total_runtime_s/60:.3f} min "
          f"= {total_runtime_s/3600:.3f} h")
