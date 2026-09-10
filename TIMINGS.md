# Timings in `ravens_parser_parallell_multicluster2.py`

Three separate clocks run during an evaluation sweep, and they answer different
questions. This document describes what each one measures, where its numbers end up,
and how accurate they are.

| Clock | Question it answers | Where the number lives |
|---|---|---|
| Per-method runtime | How long did this solver take on this scan? | Inside the saved `*_stats` pickle |
| Scan budget | Has this scan run long enough that we should give up? | Nowhere; it aborts the scan |
| Run wall clock | How long is the whole sweep taking? | Printed by the driver |

---

## 1. Per-method runtimes

Every solver is timed with `time.perf_counter()` around its call, and every method now
persists what it measured. The boundary is the same for all of them: inputs in hand ->
marginals and normalization constant in hand. The `deepcopy` of the prior hypotheses each
method needs is taken outside its timer, so that harness cost is charged to nobody.

| Method | Timed field | Notes |
|---|---|---|
| Multicluster LBP (C++) | `MulticlusterApproximateOutput.runtime` | Declared field. The dataclass is frozen, so the duration is passed at construction. Marginal and Z extraction sit inside the timer; `hypotheses_marginals()` is taken outside it, matching the exact path. |
| Exact EHM2 | `MulticlusterExactOutput.runtime` | Declared field, defaulting to nan. Covers the merge plus the EHM2 solves. |
| Bethe, MHLBP, PHD, inclusion-exclusion | `MulticlusterConditionendLBPOutput.runtime` and `.solve_runtime` | `runtime` is the fair total stamped by the driver, construction plus solve. `solve_runtime` is self-timed inside `compute_marginals_likelihood()`. The difference between the two is the method's setup cost, which `plot_convergence_stats.py` reports as a construction share. |
| Murty sweep | `MulticlusterMurtyOutput.runtime` | Declared field. One value **per truncation depth**, each covering that depth's full branch-and-bound plus marginal accumulation. |

Both new fields carry plain scalar nan defaults rather than `field(default_factory=...)`.
Pickles written before the fields existed restore `__dict__` without calling `__init__`,
and only a plain default installs the class attribute they then fall through to. Older
stat files therefore load, and read as untimed rather than crashing.

### Harness costs

Work that belongs to no single method is recorded separately, in
`MulticlusterData.timings`, a `MulticlusterTimings` record:

| Field | What it covers |
|---|---|
| `parse` | `MatFileParser` on the `.mat` file |
| `cluster_links` | the shared `ClusterLinks` build, hoisted out of all four conditioning-LBP methods |
| `exact_theta_posteriors` | `compute_theta_posteriors()`, excluded from `exact_output.runtime` |
| `n_workers` | pool size the scan was measured under |
| `blas_threads` | value of `OMP_NUM_THREADS` in the worker |
| `clock` | `"perf_counter"` |

`cluster_links` is hoisted rather than built inside Bethe. Bethe used to build it
implicitly and hand it to PHD and inclusion-exclusion for free, which made those two read
cheaper than they are while MHLBP rebuilt its own. Hoisting charges it to nobody and drops
the redundant rebuild.

The BLAS and OpenMP thread pools are pinned to one thread each at the top of the script,
before anything imports numpy. Without that, numpy's scipy-openblas backend starts a thread
pool per worker on top of `Pool(cpu_count())`, and the resulting oversubscription makes the
per-method numbers both noisy and unattributable. `n_workers` and `blas_threads` are stored
so two runs are only ever compared under the same conditions.

One documented asymmetry remains. The conditioning-LBP methods build and normalize their
theta posteriors inside `compute_marginals_likelihood()`, so that work sits inside their
`runtime`, while the exact and MCMH-LBP equivalents are measured outside theirs. The
LBP-side cost is a normalization loop over an already-built dict and is negligible next to
the solves.

### Runtimes make output files non-reproducible byte for byte

Two runs over the same scan produce pickles of identical size that differ in a few
dozen bytes. Those bytes are the recorded durations. This is expected. Compare outputs by
loading and comparing arrays, never with `cmp`.

---

## 2. The scan budget

```
python ravens_parser_parallell_multicluster2.py --timeout 120
```

`--timeout` is a wall-clock budget in seconds for **one scan, shared by every solver**.

**It is off by default.** The budget abandons exactly the slowest scans, which are the ones
the runtime distributions in section 1 most need, so an armed budget censors the tail of
every per-method measurement the run produces. Arm it when throughput matters more than the
tail, for instance on an accuracy-only sweep, and read section 4 before choosing a value.

**What it covers.** The clock starts when a worker picks up a scan, including the
`.mat` parse, and runs through these steps in order:

```
parse -> mcmhlbp -> exact -> cluster_links
      -> mc_bethe -> mc_mhlbp -> mc_phd -> mc_lbp_ie -> murty_sweep
```

**How it is enforced.** A monotonic deadline is set once per scan. Before each step the
remaining budget is armed with `signal.setitimer(ITIMER_REAL, remaining, 5.0)`, and the
handler raises `ScanTimeout` naming the step that was running. The 5 second repeat
interval is a safety net: if any layer ever swallowed the first exception, the next
alarm still lands. The timer is disarmed before the output is written, so saving is
never interrupted.

`ScanTimeout` derives from `BaseException`, not `Exception`, on purpose. The parser
wraps `murty_sweep` in a broad `except Exception` so one bad scan cannot kill the pool,
and a plain exception would be swallowed there.

**What happens on expiry.** The step holding the clock and every later step are left as
`None`, and the scan is still saved. Its `MulticlusterTimings` is saved too, carrying real
numbers for the phases that ran and nan for the rest. This is useful rather than wasteful: every field of
`MulticlusterData` is `Optional`, and `plot_convergence_stats.py` skips missing methods
per scan, so the methods that did finish still contribute. A scan whose exact solver
timed out contributes nothing either way, since the exact result is the reference the
others are compared against.

The scan is reported with a `timeout` outcome, distinct from `failed`, and the solver
that held the clock is named in the run summary and in `./warnings/<scan>.log`:

```
timeout: mc_lbp_ie was running when the 120 s scan budget expired after 120.0 s
```

**Why a signal works here, and how precise it is.** Overshoot is bounded by the longest
uninterruptible C++ call, because a signal handler only runs when the interpreter is
executing Python. Every heavy path in these solvers is a Python loop calling into C++
per hypothesis. The worst scan on this dataset, `priorLikelihood116` at 435 s, is 14086
separate `EHM2.run_and_likelihood` calls with the slowest single call at 0.15 s. In
practice four heavy scans run at `--timeout 20` finished in 20.025 s of wall clock.

This is also why no process kill or watchdog is needed. If a future solver ever spends
minutes inside a single C++ call, the alarm would not land until that call returned, and
only then would a hard mechanism become necessary.

---

## 3. Run wall clock and progress

The driver times the whole pool and prints the total in seconds, minutes and hours. Live
progress comes from the same counter: a `tqdm` bar on a terminal, or, when stderr is
redirected to a log file, one timestamped line every 25 scans or 60 seconds carrying
elapsed time, rate and an ETA extrapolated from the mean rate. Both carry running
`empty`, `timeout` and `failed` counts.

Because scans vary from milliseconds to minutes, the pool uses `chunksize=1`. The mean
rate is therefore meaningful, but the ETA is optimistic while the tail of slow scans is
still queued.

---

## 4. Measured distribution, and choosing a budget

This is the historical measurement the candidate budgets below were chosen from, taken
before this script recorded its own per-method timings. From
`ravens_output_multicluster/metrics/metrics.csv`, 1396 scans, seconds:

| Solver | p50 | p90 | p99 | max |
|---|---|---|---|---|
| LBP | 0.00 | 0.01 | 0.03 | 0.04 |
| Exact EHM2 | 0.16 | 4.31 | 48.09 | 435.0 |
| Efficient marginals | 0.09 | 4.23 | 44.83 | 295.5 |
| Inclusion-exclusion | 0.21 | 11.57 | 139.64 | 582.8 |
| **Whole scan** | 0.71 | 24.88 | 226.90 | 984.7 |

Effect of the budget on that same population:

| `--timeout` | Scans hit | Share | Wall clock saved |
|---|---|---|---|
| 60 | 60 / 1396 | 4.3 % | ~2.0 h |
| 120 | 28 / 1396 | 2.0 % | ~1.3 h |
| 180 | 18 / 1396 | 1.3 % | ~0.9 h |
| 300 | 9 / 1396 | 0.6 % | ~0.5 h |

At 120 s, the inclusion-exclusion method holds the clock in 22 of the 28 affected scans,
the efficient method in 4 and the exact solver in 2.

**Provenance caveat.** That table comes from the sibling script
`ravens_parser_parallell_multicluster_not_exact.py`. Its method set is close to but not
identical with this script's: its efficient column times
`cbt.MulticlusterEfficientMarginals` while this script runs the LBP-conditioned variants,
and it carries no Murty sweep, which adds roughly a second per scan here. It also predates
the BLAS pinning, so it was measured under whatever oversubscription the pool happened to
have. Treat the numbers as the right order of magnitude rather than exact predictions.

Now that this script records per-method runtimes itself, prefer a completed unbudgeted
sweep as the source. `plot_convergence_stats.py --timed-only` restricts every figure to
scans carrying a `MulticlusterTimings` record, which keeps a directory that mixes an
interrupted run with an older one from comparing some methods over one population and the
rest over another.
