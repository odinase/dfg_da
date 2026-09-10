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
import dfg_da.graph_stats as gs
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

import argparse
from collections import Counter
from contextlib import contextmanager
import functools
from multiprocessing import Pool
import multiprocessing
import signal
import sys
import time
import traceback
from pathlib import Path

import py_dfg_da

def _n_workers():
    """Default worker count, from EVAL_WORKERS or the machine. Overridable with --workers.

    The resolved value is handed to each worker so it can stamp the count into its
    MulticlusterTimings: the per-method runtimes are only comparable across runs at the
    same level of oversubscription.
    """
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

# Wall-clock budget for one scan, overridable with --timeout. Off by default: the budget
# abandons exactly the slowest scans, which are the ones the per-method timings most need,
# so leaving it armed would censor the tail of every runtime distribution this script feeds.
# 120 s is the useful value when throughput matters more than the tail; see TIMINGS.md.
DEFAULT_TIMEOUT_S = 0.0

# Once the budget expires the alarm repeats at this interval until the step is left,
# so a swallowed ScanTimeout cannot let a scan run past its budget unbounded.
TIMER_RETRY_INTERVAL_S = 5.0

# Cap on the merged-cluster hypotheses whose conditioned graph we are willing to build when
# the exact solver did not run and its merge is therefore not available to reuse. One
# conditioned graph costs on the order of 80 us, so this bounds the fallback at a few seconds
# on a scan that has already proved pathological. Above it, per_merged_cluster is left empty
# with merged_skipped_reason = "enumeration_cap"; the prior-cluster numbers are unaffected.
MAX_MERGED_HYPOTHESES = 100_000


class ScanTimeout(BaseException):
    """Raised in a worker when a scan exceeds its wall-clock budget.

    Derives from ``BaseException`` rather than ``Exception`` on purpose: the broad
    ``except Exception`` around murty_sweep below, and any such guard inside the
    solvers, must not be able to swallow it.
    """

    def __init__(self, label, budget_s, elapsed_s):
        super().__init__(f"{label} was running when the {budget_s:g} s scan budget "
                         f"expired after {elapsed_s:.1f} s")
        self.label = label
        self.budget_s = budget_s
        self.elapsed_s = elapsed_s


class _ScanDeadline:
    """One wall-clock budget shared by every solver of a single scan.

    Enforced with SIGALRM. Every heavy path in these solvers is a Python loop calling
    into C++ per hypothesis (the worst scan measured is 14086 EHM2 calls of <= 0.15 s
    each), so the interpreter regains control every few milliseconds and the handler
    fires promptly. A budget of None or <= 0 disables the mechanism entirely.
    """

    def __init__(self, budget_s):
        self.budget_s = budget_s if budget_s and budget_s > 0 else None
        self.start = time.monotonic()
        self.deadline = None if self.budget_s is None else self.start + self.budget_s

    def elapsed(self):
        return time.monotonic() - self.start

    def remaining(self):
        return None if self.deadline is None else self.deadline - time.monotonic()

    @contextmanager
    def step(self, label):
        """Run one solver under the remaining budget, raising ScanTimeout if it runs out."""
        if self.deadline is None:
            yield
            return

        remaining = self.remaining()
        if remaining <= 0.0:
            raise ScanTimeout(label, self.budget_s, self.elapsed())

        def _fire(signum, frame):
            raise ScanTimeout(label, self.budget_s, self.elapsed())

        previous = signal.signal(signal.SIGALRM, _fire)
        signal.setitimer(signal.ITIMER_REAL, remaining, TIMER_RETRY_INTERVAL_S)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous)


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


def merged_enumeration_size(prior_hypotheses_per_cluster, assocLocal):
    """Total hypotheses over the merged clusters, i.e. how many conditioned graphs the
    merged pass would have to build. Merging is a Cartesian product over the member
    clusters, so this is the number that explodes."""
    total = 0
    for members in gs.merged_cluster_members(assocLocal):
        size = 1
        for cluster in members:
            size *= len(prior_hypotheses_per_cluster[cluster])
        total += size

    return total


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


def _process_file(pmbm_file, timeout_s=None, n_workers=0):
    """Run every solver on one scan under a shared wall-clock budget.

    Returns ``(kind, detail)``. When the budget expires the solver that held the clock
    and every later one are left as None and the scan is still saved: every field of
    MulticlusterData is Optional and plot_convergence_stats skips missing methods per
    scan, so the methods that did finish still count.
    """
    deadline = _ScanDeadline(timeout_s)

    pmbm_file_path = Path(pmbm_file)
    pmbm_filename = pmbm_file_path.name[:-len(pmbm_file_path.suffix)]

    path = Path(f"{OUTPUT_PATH_BASE}_convergence")
    path.mkdir(parents=True, exist_ok=True)

    save_path = f"{path}/{pmbm_filename}_stats"

    mat_data = None
    mcmhlbp_output = None
    exact_output = None
    cluster_links = None
    mc_bethe = None
    mc_bethe_output = None
    mc_mhlbp_output = None
    mc_phd_output = None
    mc_lbp_ie_output = None
    mc_murty_outputs = None
    graph_stats = None
    explicit_hypothesis_enumeration_error = False
    timed_out_label = None

    # Harness durations, pre-set to nan: a scan whose budget expires part-way still saves a
    # MulticlusterTimings, carrying real numbers for the phases that ran and nan for the rest.
    dur_parse = float("nan")
    dur_cluster_links = float("nan")
    dur_exact_thetas = float("nan")
    dur_graph_stats = float("nan")

    try:
        with deadline.step("parse"):
            start = time.perf_counter()
            mat_data = sl.MatFileParser(pmbm_file, use_cpp=True)
            dur_parse = time.perf_counter() - start

            R = np.asfortranarray(mat_data.reward_matrix_edmund)
            R_LC = np.asfortranarray(mat_data.reward_matrix_lc)
            prior_hypotheses_per_cluster = mat_data.prior_hypotheses_per_cluster

            num_clusters = len(prior_hypotheses_per_cluster)
            assocLocal = mat_data.ws["assocLocal"].copy()

        if num_clusters == 0:
            return "empty", None

        # Cyclomatic numbers of the association graphs the solvers are about to run on.
        # Taken before any of them, because the scans a budget cuts short are exactly the
        # loopy ones whose topology we most want recorded. The merged-cluster half is left
        # for the end of the scan; until it runs the record says so.
        with deadline.step("graph_stats"):
            start = time.perf_counter()
            graph_stats = gs.multicluster_graph_stats(
                R_LC, prior_hypotheses_per_cluster, merged_skipped_reason="timeout")
            dur_graph_stats = time.perf_counter() - start

        # Marginal and Z extraction is inside the timer: it is part of "inputs in hand ->
        # marginals + Z in hand", the boundary every method here is measured on.
        # hypotheses_marginals() is left outside, matching the exact path, since theta
        # posteriors are not part of that.
        with deadline.step("mcmhlbp"):
            start = time.perf_counter()
            mcmhlbp = py_dfg_da.lbp.lbp_multicluster(R, prior_hypotheses_per_cluster)
            mcmhlbp_marginals = mcmhlbp.track_association_marginals().T
            mcmhlbp_normalization_constant = mcmhlbp.bethe_pseudodual_normalization_constant()
            dur_mcmhlbp = time.perf_counter() - start

            mcmhlbp_output = sl.MulticlusterApproximateOutput(
                approx_marginals=mcmhlbp_marginals,
                approx_normalization_constant=mcmhlbp_normalization_constant,
                approx_theta_posteriors=mcmhlbp.hypotheses_marginals(),
                full_output=mcmhlbp,
                runtime=dur_mcmhlbp
            )

        with deadline.step("exact"):
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
        # retained, so one instance is safe to share across all four. Hoisting also decouples PHD
        # and IE from Bethe having finished, which matters once the budget can cut a scan short.
        with deadline.step("cluster_links"):
            links_hypotheses = deepcopy(prior_hypotheses_per_cluster)
            start = time.perf_counter()
            cluster_links = cbt.ClusterLinks(R_LC=R_LC, prior_hypotheses_per_cluster=links_hypotheses, assocLocal=assocLocal.copy())
            dur_cluster_links = time.perf_counter() - start

        # The deepcopy is required, not defensive: ConditionedCluster.__init__ calls
        # prior_hypotheses.reindex_tracks(), which mutates. It is taken outside each timer.
        with deadline.step("mc_bethe"):
            bethe_hypotheses = deepcopy(prior_hypotheses_per_cluster)
            mc_bethe, mc_bethe_output = _timed(lambda: MulticlusterEfficientMarginalsLBP(R_LC=R_LC, prior_hypotheses_per_cluster=bethe_hypotheses, assocLocal=assocLocal.copy(), lbp_solver=mc.LBPMarginalsByTotalProbBethe(), cluster_links=cluster_links))

        with deadline.step("mc_mhlbp"):
            mhlbp_hypotheses = deepcopy(prior_hypotheses_per_cluster)
            mc_mhlbp, mc_mhlbp_output = _timed(lambda: MulticlusterEfficientMarginalsLBP(R_LC=R_LC, prior_hypotheses_per_cluster=mhlbp_hypotheses, assocLocal=assocLocal.copy(), lbp_solver=mc.LBPMarginalsFullAssociationCPP(), cluster_links=cluster_links))

        with deadline.step("mc_phd"):
            phd_hypotheses = deepcopy(prior_hypotheses_per_cluster)
            mc_phd, mc_phd_output = _timed(lambda: MulticlusterEfficientMarginalsLBP(R_LC=R_LC, prior_hypotheses_per_cluster=phd_hypotheses, assocLocal=assocLocal.copy(), lbp_solver=mc.LBPMarginalsByTotalProbPHD(), cluster_links=cluster_links))

        with deadline.step("mc_lbp_ie"):
            ie_hypotheses = deepcopy(prior_hypotheses_per_cluster)
            mc_lbp_ie, mc_lbp_ie_output = _timed(lambda: MulticlusterEfficientMarginalsLBPInclusionExclusion(R_LC=R_LC, prior_hypotheses_per_cluster=ie_hypotheses, assocLocal=assocLocal.copy(), lbp_solver=mc.LBPMarginalsByTotalProbBethe(), cluster_links=cluster_links))

        # Murty baseline over the nHypoTotalMax sweep. Guarded: this runs under Pool.map over
        # all 1397 scans and the branch-and-bound carries iteration caps, so one bad scan must
        # not take the pool down with it. ScanTimeout is a BaseException and passes through.
        with deadline.step("murty_sweep"):
            try:
                mc_murty_outputs = murty_sweep(mat_data.ws, mat_data.num_tracks,
                                               mat_data.num_measurements)
            except Exception as e:
                mc_murty_outputs = None
                warning_handler(pmbm_filename, [("murty_sweep", repr(e))])

        # Merged clusters are what MulticlusterExactEHM2 conditions on hypothesis for
        # hypothesis. Done last so it can never take budget from a method, and reusing the
        # exact solver's own merge when there is one: that is free, and it is exactly the
        # partition the exact numbers were computed over.
        with deadline.step("graph_stats_merged"):
            start = time.perf_counter()
            merged_clusters = None
            if exact_output is not None:
                merged_clusters = exact_output.cluster_hypotheses_posterior.prior_hypotheses_per_cluster_posterior
            elif merged_enumeration_size(prior_hypotheses_per_cluster, assocLocal) <= MAX_MERGED_HYPOTHESES:
                merged_hypotheses = deepcopy(prior_hypotheses_per_cluster)
                merged_clusters = mc.ClusterHypothesesPosterior(
                    assocLocal=assocLocal.copy(),
                    prior_hypotheses_per_cluster=merged_hypotheses,
                ).prior_hypotheses_per_cluster_posterior

            if merged_clusters is None:
                graph_stats.merged_skipped_reason = "enumeration_cap"
            else:
                graph_stats.per_merged_cluster = [
                    gs.cluster_graph_stats(R_LC, hypotheses, members)
                    for hypotheses, members in zip(merged_clusters,
                                                   gs.merged_cluster_members(assocLocal))
                ]
                graph_stats.merged_skipped_reason = ""
            dur_graph_stats += time.perf_counter() - start

    except ScanTimeout as e:
        timed_out_label = e.label
        warning_handler(pmbm_filename, [("timeout", str(e))])

    if mat_data is None:
        # The budget expired while reading the .mat file, so there is nothing to save.
        return "timeout", timed_out_label

    cluster_data = sl.MulticlusterData(
        mcmhlbp_output=mcmhlbp_output,
        mc_phd_output=mc_phd_output,
        mc_bethe_output=mc_bethe_output,
        mc_mhlbp_output=mc_mhlbp_output,
        mc_lbp_ie_output=mc_lbp_ie_output,
        exact_output=exact_output,
        mc_murty_outputs=mc_murty_outputs,
        graph_stats=graph_stats,
        timings=sl.MulticlusterTimings(
            parse=dur_parse,
            cluster_links=dur_cluster_links,
            exact_theta_posteriors=dur_exact_thetas,
            graph_stats=dur_graph_stats,
            n_workers=n_workers,
            blas_threads=int(os.environ["OMP_NUM_THREADS"]),
        ),
        explicit_hypothesis_enumeration_error=explicit_hypothesis_enumeration_error
    )

    cluster_data.save_data(save_path)

    if timed_out_label is not None:
        return "timeout", timed_out_label
    return "ok", None


def loop_func(pmbm_file, timeout_s=None, n_workers=0):
    """Worker entry point. Returns ``(kind, filename, detail)``.

    ``kind`` is ok/empty/timeout/failed. A single unreadable or pathological scan must
    not take the whole pool down and lose the hours of work already queued behind it, so
    every non-exit exception is logged under ./warnings/ and reported back to the driver
    instead of propagating.
    """
    pmbm_file_path = Path(pmbm_file)
    pmbm_filename = pmbm_file_path.name[:-len(pmbm_file_path.suffix)]
    try:
        kind, detail = _process_file(pmbm_file, timeout_s, n_workers)
    except ScanTimeout as e:
        # Safety net for an alarm landing outside any step, e.g. while saving.
        warning_handler(pmbm_filename, [("timeout", str(e))])
        return "timeout", pmbm_filename, e.label
    except Exception as e:
        warning_handler(pmbm_filename, [("loop_func", repr(e)),
                                        ("traceback", traceback.format_exc())])
        return "failed", pmbm_filename, repr(e)
    return kind, pmbm_filename, detail


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run every multicluster marginals solver over the ravens PMBM scans.")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S, metavar="SECONDS",
        help="wall-clock budget for one scan, shared by every solver "
             "(default: %(default)s, 0 disables). When it expires the scan stops there "
             "and the methods that already finished are still saved. Note that it cuts "
             "the slowest scans, so an armed budget censors the recorded runtime tail.")
    parser.add_argument(
        "--workers", type=int,
        default=_n_workers(),
        metavar="N", help="number of worker processes (default: %(default)s)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    pmbm_files = glob(PMBM_DATA_PATH + "/*.mat")

    pmbm_files = sorted(pmbm_files)
    num_files = len(pmbm_files)

    if num_files == 0:
        raise ValueError(f"No .mat files found under {PMBM_DATA_PATH}")

    n_workers = args.workers
    budget = f"{args.timeout:g} s timeout per scan" if args.timeout > 0 else "no timeout"
    print(f"Computing {num_files} files with {n_workers} workers, "
          f"{os.environ['OMP_NUM_THREADS']} BLAS thread(s) each, {budget}...", flush=True)

    counts = Counter()
    failed_files = []
    timed_out_files = []
    is_tty = sys.stderr.isatty()
    printer = None if is_tty else _ProgressPrinter(num_files)

    # n_workers is resolved here and handed to the workers: each one stamps it into its
    # MulticlusterTimings, and a worker cannot see args.
    worker = functools.partial(loop_func, timeout_s=args.timeout, n_workers=n_workers)

    start = time.perf_counter()
    with Pool(processes=n_workers) as pool:
        # chunksize=1: per-file runtime spans milliseconds to minutes, so the default
        # chunking both stalls the bar and unbalances the workers.
        results = pool.imap_unordered(worker, pmbm_files, chunksize=1)
        with tqdm(total=num_files, disable=not is_tty) as pbar:
            for n_done, (kind, name, detail) in enumerate(results, start=1):
                counts[kind] += 1
                if kind == "failed":
                    failed_files.append(name)
                elif kind == "timeout":
                    timed_out_files.append((name, detail))
                pbar.update(1)
                pbar.set_postfix(empty=counts["empty"], timeout=counts["timeout"],
                                 failed=counts["failed"])
                if printer is not None:
                    printer.update(n_done, empty=counts["empty"],
                                   timeout=counts["timeout"], failed=counts["failed"])
    stop = time.perf_counter()

    print("Pools done")
    print(f"{counts['ok']} ok, {counts['empty']} empty, "
          f"{counts['timeout']} timed out, {counts['failed']} failed")
    if timed_out_files:
        print(f"Timed out (partial stats saved, see ./warnings/):")
        for name, label in timed_out_files[:20]:
            print(f"  {name}: {label} held the clock")
        if len(timed_out_files) > 20:
            print(f"  ... and {len(timed_out_files) - 20} more")
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
