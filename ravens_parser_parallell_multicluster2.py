# Pin the BLAS/OpenMP thread pools to one thread each, before anything imports numpy.
# numpy here links scipy-openblas, which would otherwise start a thread pool per worker on
# top of Pool(cpu_count()) -- oversubscribing the machine and making the per-method timings
# below both noisy and unattributable. setdefault, so an explicit override still wins.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import dfg_da.marginals_computers as mc
import dfg_da.prior_hypothesis as phs
import dfg_da.cluster_bayes_tree as cbt
import dfg_da.stats_logger as sl
from dfg_da.marginal_association_Odin import ExplicitHypothesisEnumerationError
from dfg_da.cluster_conditioning_lbp import MulticlusterEfficientMarginalsLBP, MulticlusterConditionendLBPOutput
from dfg_da.cluster_conditioning_lbp_ie import MulticlusterEfficientMarginalsLBPInclusionExclusion
from dfg_da.murty_marginals import murty_sweep

import matplotlib.pyplot as plt
from glob import glob
from typing import List
import numpy as np
from tqdm import tqdm


from copy import deepcopy

from collections import Counter
from multiprocessing import Pool
import multiprocessing
import sys
import time
import traceback
from pathlib import Path

import py_dfg_da

def _n_workers():
    """Worker count, resolved the same way in the pool and in each worker's timing record."""
    return int(os.environ.get("EVAL_WORKERS", multiprocessing.cpu_count()))


def warning_handler(pmbm_file, function_flags):
    Path('./warnings').mkdir(parents=True, exist_ok=True)
    with open(f'./warnings/{pmbm_file}.log', 'a') as f:
        for function_name, flag in function_flags:
            f.write(f"{function_name}: {flag}\n")


OUTPUT_PATH_BASE = "./ravens_output_multicluster"
PMBM_DATA_PATH = "./data/pmbm_output_files"

# Progress reporting when stderr is not a terminal (run redirected to a log file):
# a tqdm bar would fill the log with carriage returns, so print a discrete line
# every PROGRESS_EVERY files or every PROGRESS_INTERVAL_S seconds, whichever
# comes first.
PROGRESS_EVERY = 25
PROGRESS_INTERVAL_S = 60.0


class _ProgressPrinter:
    """Line-based progress for non-tty output. Call ``update(n_done)`` per file."""

    def __init__(self, total, every=PROGRESS_EVERY, interval_s=PROGRESS_INTERVAL_S):
        self.total = total
        self.every = every
        self.interval_s = interval_s
        self.start = time.time()
        self.last_print = self.start
        self.last_n = 0

    def update(self, n_done, **extra):
        now = time.time()
        due = (n_done - self.last_n >= self.every) or (now - self.last_print >= self.interval_s)
        if not due and n_done != self.total:
            return
        self.last_print = now
        self.last_n = n_done
        self._emit(n_done, now, extra)

    def _emit(self, n_done, now, extra):
        elapsed = now - self.start
        rate = n_done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - n_done) / rate if rate > 0 else float("nan")
        pct = 100.0 * n_done / self.total if self.total else 100.0
        suffix = "".join(f", {k}={v}" for k, v in extra.items() if v)
        print(f"[{time.strftime('%H:%M:%S')}] {n_done}/{self.total} ({pct:5.1f}%) "
              f"elapsed {elapsed/60:.1f} min, {rate:.2f} files/s, "
              f"eta {eta/60:.1f} min{suffix}", flush=True)


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


def _timed(factory):
    """Construct and solve one conditioning-LBP method under a single timer.

    The measured boundary is the same one every other method is held to: inputs in hand ->
    marginals + Z in hand. The caller's ``deepcopy`` is already done by the time we start,
    so that harness cost is not charged to the method.
    """
    start = time.perf_counter()
    computer = factory()
    out = computer.compute_marginals_likelihood()
    out.runtime = time.perf_counter() - start
    return computer, out


def _process_file(pmbm_file):
    start = time.perf_counter()
    mat_data: sl.MatFileParser = sl.MatFileParser(pmbm_file, use_cpp=True)
    dur_parse = time.perf_counter() - start

    R = np.asfortranarray(mat_data.reward_matrix_edmund)
    R_LC = np.asfortranarray(mat_data.reward_matrix_lc)
    prior_hypotheses_per_cluster = mat_data.prior_hypotheses_per_cluster

    pmbm_file_path = Path(pmbm_file)
    pmbm_filename = pmbm_file_path.name[:-len(pmbm_file_path.suffix)]

    path = Path(f"{OUTPUT_PATH_BASE}_convergence")
    path.mkdir(parents=True, exist_ok=True)

    save_path = f"{path}/{pmbm_filename}_stats"

    num_clusters = len(prior_hypotheses_per_cluster)
    if num_clusters == 0:
        return "empty"

    # Marginal and Z extraction is inside the timer: it is part of "inputs in hand -> marginals
    # + Z in hand", the boundary every method here is measured on. hypotheses_marginals() is
    # left outside, matching the exact path, since theta posteriors are not part of that.
    start = time.perf_counter()
    mcmhlbp = py_dfg_da.lbp.lbp_multicluster(R, prior_hypotheses_per_cluster)
    mcmhlbp_marginals = mcmhlbp.track_association_marginals().T
    mcmhlbp_normalization_constant = mcmhlbp.bethe_pseudodual_normalization_constant()
    dur_mcmhlbp = time.perf_counter() - start
    mcmhlbp_theta_posteriors = mcmhlbp.hypotheses_marginals()

    assocLocal = mat_data.ws["assocLocal"].copy()
    explicit_hypothesis_enumeration_error = False
    exact_output = None
    dur_exact_thetas = float("nan")
    exact_computer = mc.MulticlusterExactEHM2()

    try:
        start = time.perf_counter()
        exact_output: mc.MulticlusterExactOutput = exact_computer(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal.copy())
        exact_output.runtime = time.perf_counter() - start

        start = time.perf_counter()
        exact_output.theta_posteriors = exact_output.compute_theta_posteriors()
        dur_exact_thetas = time.perf_counter() - start
    except ExplicitHypothesisEnumerationError:
        explicit_hypothesis_enumeration_error = True

    # Cluster links are shared topology, not any one method's work. Bethe used to build them
    # implicitly and hand them to PHD and IE for free, which made those two look cheaper than
    # they are and left MHLBP rebuilding its own; hoisting the build charges it to nobody and
    # drops the redundant rebuild. Only read downstream, and no reference to the hypotheses is
    # retained, so one instance is safe to share across all four.
    links_hypotheses = deepcopy(prior_hypotheses_per_cluster)
    start = time.perf_counter()
    cluster_links = cbt.ClusterLinks(R_LC=R_LC, prior_hypotheses_per_cluster=links_hypotheses, assocLocal=assocLocal.copy())
    dur_cluster_links = time.perf_counter() - start

    # The deepcopy is required, not defensive: ConditionedCluster.__init__ calls
    # prior_hypotheses.reindex_tracks(), which mutates. It is taken outside each timer.
    bethe_hypotheses = deepcopy(prior_hypotheses_per_cluster)
    mc_bethe, mc_bethe_output = _timed(lambda: MulticlusterEfficientMarginalsLBP(R_LC=R_LC, prior_hypotheses_per_cluster=bethe_hypotheses, assocLocal=assocLocal.copy(), lbp_solver=mc.LBPMarginalsByTotalProbBethe(), cluster_links=cluster_links))

    mhlbp_hypotheses = deepcopy(prior_hypotheses_per_cluster)
    mc_mhlbp, mc_mhlbp_output = _timed(lambda: MulticlusterEfficientMarginalsLBP(R_LC=R_LC, prior_hypotheses_per_cluster=mhlbp_hypotheses, assocLocal=assocLocal.copy(), lbp_solver=mc.LBPMarginalsFullAssociationCPP(), cluster_links=cluster_links))

    phd_hypotheses = deepcopy(prior_hypotheses_per_cluster)
    mc_phd, mc_phd_output = _timed(lambda: MulticlusterEfficientMarginalsLBP(R_LC=R_LC, prior_hypotheses_per_cluster=phd_hypotheses, assocLocal=assocLocal.copy(), lbp_solver=mc.LBPMarginalsByTotalProbPHD(), cluster_links=cluster_links))

    ie_hypotheses = deepcopy(prior_hypotheses_per_cluster)
    mc_lbp_ie, mc_lbp_ie_output = _timed(lambda: MulticlusterEfficientMarginalsLBPInclusionExclusion(R_LC=R_LC, prior_hypotheses_per_cluster=ie_hypotheses, assocLocal=assocLocal.copy(), lbp_solver=mc.LBPMarginalsByTotalProbBethe(), cluster_links=cluster_links))

    # Murty baseline over the nHypoTotalMax sweep. Guarded: this runs under Pool.map over
    # all 1397 scans and the branch-and-bound carries iteration caps, so one bad scan must
    # not take the pool down with it.
    try:
        mc_murty_outputs = murty_sweep(mat_data.ws, mat_data.num_tracks,
                                       mat_data.num_measurements)
    except Exception as e:
        mc_murty_outputs = None
        warning_handler(pmbm_filename, [("murty_sweep", repr(e))])


    cluster_data = sl.MulticlusterData(
        mcmhlbp_output=sl.MulticlusterApproximateOutput(
            approx_marginals=mcmhlbp_marginals,
            approx_normalization_constant=mcmhlbp_normalization_constant,
            approx_theta_posteriors=mcmhlbp_theta_posteriors,
            full_output=mcmhlbp,
            runtime=dur_mcmhlbp
        ),
        mc_phd_output=mc_phd_output,
        mc_bethe_output=mc_bethe_output,
        mc_mhlbp_output=mc_mhlbp_output,
        mc_lbp_ie_output=mc_lbp_ie_output,
        exact_output=exact_output,
        mc_murty_outputs=mc_murty_outputs,
        timings=sl.MulticlusterTimings(
            parse=dur_parse,
            cluster_links=dur_cluster_links,
            exact_theta_posteriors=dur_exact_thetas,
            n_workers=_n_workers(),
            blas_threads=int(os.environ["OMP_NUM_THREADS"]),
        ),
        explicit_hypothesis_enumeration_error=explicit_hypothesis_enumeration_error
    )

    cluster_data.save_data(save_path)

    return "ok"


def loop_func(pmbm_file):
    """Worker entry point. Returns ``(kind, filename)`` with kind in ok/empty/failed.

    A single unreadable or pathological scan must not take the whole pool down and
    lose the hours of work already queued behind it, so every non-exit exception is
    logged under ./warnings/ and reported back to the driver instead of propagating.
    """
    pmbm_file_path = Path(pmbm_file)
    pmbm_filename = pmbm_file_path.name[:-len(pmbm_file_path.suffix)]
    try:
        kind = _process_file(pmbm_file)
    except Exception as e:
        warning_handler(pmbm_filename, [("loop_func", repr(e)),
                                        ("traceback", traceback.format_exc())])
        return "failed", pmbm_filename
    return kind, pmbm_filename


if __name__ == "__main__":
    pmbm_files = glob(PMBM_DATA_PATH + "/*.mat")

    pmbm_files = sorted(pmbm_files)
    num_files = len(pmbm_files)

    if num_files == 0:
        raise ValueError(f"No .mat files found under {PMBM_DATA_PATH}")

    n_workers = _n_workers()
    print(f"Computing {num_files} files with {n_workers} workers, "
          f"{os.environ['OMP_NUM_THREADS']} BLAS thread(s) each...", flush=True)

    counts = Counter()
    failed_files = []
    is_tty = sys.stderr.isatty()
    printer = None if is_tty else _ProgressPrinter(num_files)

    start = time.perf_counter()
    with Pool(processes=n_workers) as pool:
        # chunksize=1: per-file runtime spans milliseconds to minutes, so the default
        # chunking both stalls the bar and unbalances the workers.
        results = pool.imap_unordered(loop_func, pmbm_files, chunksize=1)
        with tqdm(total=num_files, disable=not is_tty) as pbar:
            for n_done, (kind, name) in enumerate(results, start=1):
                counts[kind] += 1
                if kind == "failed":
                    failed_files.append(name)
                pbar.update(1)
                pbar.set_postfix(empty=counts["empty"], failed=counts["failed"])
                if printer is not None:
                    printer.update(n_done, empty=counts["empty"], failed=counts["failed"])
    stop = time.perf_counter()

    print("Pools done")
    print(f"{counts['ok']} ok, {counts['empty']} empty, {counts['failed']} failed")
    if failed_files:
        for name in failed_files[:20]:
            print(f"  failed: {name}")
        if len(failed_files) > 20:
            print(f"  ... and {len(failed_files) - 20} more; see ./warnings/ for details")
        else:
            print("  see ./warnings/ for details")

    duration_s = stop - start
    duration_min = duration_s / 60.0
    duration_h = duration_min / 60.0
    print(f"Spent {duration_s:.3f} s = {duration_min:.3f} min = {duration_h:.3f} h")
