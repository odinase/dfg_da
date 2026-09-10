"""Add ``graph_stats`` to ``*_stats`` files that were written before the field existed.

The convergence sweep in ``ravens_parser_parallell_multicluster2.py`` takes hours, almost all
of it in the solvers. The association-graph topology depends on nothing the solvers produce --
only on the scan's reward matrix and its prior hypotheses -- so it can be filled in afterwards
from the .mat file, leaving every other field of the saved record untouched.

Where the record already holds an exact solve, the merged (posterior) clusters are taken from
its own ``cluster_hypotheses_posterior``: that is free, and it is exactly the partition the
exact numbers were computed over.

Run from the repo root::

    .venv/bin/python backfill_graph_stats.py --workers 8
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import functools
import multiprocessing
import pickle
import sys
import time
import traceback
from collections import Counter
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

import dfg_da.graph_stats as gs
import dfg_da.stats_logger as sl
from dfg_da.marginals_computers import ClusterHypothesesPosterior

from ravens_parser_parallell_multicluster2 import (
    MAX_MERGED_HYPOTHESES,
    OUTPUT_PATH_BASE,
    PMBM_DATA_PATH,
    _ProgressPrinter,
    merged_enumeration_size,
    warning_handler,
)

STATS_PATH = f"{OUTPUT_PATH_BASE}_convergence"


def compute_graph_stats(mat_file, data=None):
    """Graph stats for one scan, reusing ``data``'s merge when it carries an exact solve."""
    mat_data = sl.MatFileParser(mat_file, use_cpp=True)
    prior_hypotheses_per_cluster = mat_data.prior_hypotheses_per_cluster
    if len(prior_hypotheses_per_cluster) == 0:
        return None

    R_LC = np.asfortranarray(mat_data.reward_matrix_lc)
    assocLocal = mat_data.ws["assocLocal"]

    exact_output = getattr(data, "exact_output", None)
    merged_clusters = None
    if exact_output is not None:
        merged_clusters = exact_output.cluster_hypotheses_posterior.prior_hypotheses_per_cluster_posterior
    elif merged_enumeration_size(prior_hypotheses_per_cluster, assocLocal) <= MAX_MERGED_HYPOTHESES:
        merged_clusters = ClusterHypothesesPosterior(
            assocLocal=assocLocal.copy(),
            prior_hypotheses_per_cluster=prior_hypotheses_per_cluster,
        ).prior_hypotheses_per_cluster_posterior

    return gs.multicluster_graph_stats(
        R_LC,
        prior_hypotheses_per_cluster,
        merged_clusters=merged_clusters,
        merged_members=None if merged_clusters is None else gs.merged_cluster_members(assocLocal),
        merged_skipped_reason="" if merged_clusters is not None else "enumeration_cap",
    )


def backfill_one(stats_file, data_dir, force=False, dry_run=False):
    """Returns ``(kind, name, detail)`` with kind ok/skipped/missing/empty/failed."""
    stats_path = Path(stats_file)
    name = stats_path.name[:-len("_stats")] if stats_path.name.endswith("_stats") else stats_path.name

    try:
        data = sl.MulticlusterData.from_data(stats_path)

        if getattr(data, "graph_stats", None) is not None and not force:
            return "skipped", name, None

        mat_file = Path(data_dir) / f"{name}.mat"
        if not mat_file.exists():
            return "missing", name, str(mat_file)

        graph_stats = compute_graph_stats(mat_file, data)
        if graph_stats is None:
            return "empty", name, None

        if dry_run:
            return "ok", name, None

        # Write beside the target and rename over it, so an interrupted run leaves the
        # original pickle intact rather than a truncated one.
        tmp_path = stats_path.with_name(f"{stats_path.name}.backfill.{os.getpid()}")
        data.graph_stats = graph_stats
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f)
        os.replace(tmp_path, stats_path)
    except Exception as e:
        warning_handler(name, [("backfill_graph_stats", repr(e)),
                               ("traceback", traceback.format_exc())])
        return "failed", name, repr(e)

    return "ok", name, None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Add the association-graph cyclomatic numbers to existing *_stats files.")
    parser.add_argument("--stats-dir", default=STATS_PATH, metavar="DIR",
                        help="directory of *_stats pickles to update (default: %(default)s)")
    parser.add_argument("--data-dir", default=PMBM_DATA_PATH, metavar="DIR",
                        help="directory of the source .mat scans (default: %(default)s)")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("EVAL_WORKERS", multiprocessing.cpu_count())),
                        metavar="N", help="number of worker processes (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="recompute even for files that already carry graph_stats")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute everything but write nothing back")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    stats_files = sorted(glob(f"{args.stats_dir}/*_stats"))
    num_files = len(stats_files)
    if num_files == 0:
        raise ValueError(f"No *_stats files found under {args.stats_dir}")

    print(f"Backfilling {num_files} stats files with {args.workers} workers"
          f"{' (dry run)' if args.dry_run else ''}...", flush=True)

    counts = Counter()
    failed_files = []
    is_tty = sys.stderr.isatty()
    printer = None if is_tty else _ProgressPrinter(num_files)

    worker = functools.partial(backfill_one, data_dir=args.data_dir,
                               force=args.force, dry_run=args.dry_run)

    start = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        results = pool.imap_unordered(worker, stats_files, chunksize=1)
        with tqdm(total=num_files, disable=not is_tty) as pbar:
            for n_done, (kind, name, detail) in enumerate(results, start=1):
                counts[kind] += 1
                if kind in ("failed", "missing"):
                    failed_files.append((kind, name, detail))
                pbar.update(1)
                pbar.set_postfix(skipped=counts["skipped"], missing=counts["missing"],
                                 failed=counts["failed"])
                if printer is not None:
                    printer.update(n_done, skipped=counts["skipped"],
                                   missing=counts["missing"], failed=counts["failed"])
    duration_s = time.perf_counter() - start

    print(f"{counts['ok']} updated, {counts['skipped']} already had graph_stats, "
          f"{counts['empty']} empty, {counts['missing']} without a .mat, "
          f"{counts['failed']} failed")
    for kind, name, detail in failed_files[:20]:
        print(f"  {kind}: {name}: {detail}")
    if len(failed_files) > 20:
        print(f"  ... and {len(failed_files) - 20} more; see ./warnings/ for details")

    print(f"Spent {duration_s:.3f} s = {duration_s/60:.3f} min")
