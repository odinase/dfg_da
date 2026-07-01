# Session summary — multi-cluster DA eval, IE method, accuracy & cyclomatic analysis

Work done in this session on the `pmbm-python-impl` branch, run against the
existing PMBM output data in `data/pmbm_output_files/` (1396 `priorLikelihood*.mat`
scans — no filter was re-run).

Environment used throughout: a single root venv `.venv` (formerly
`pmbm-cm-python/.venv`) with the GTSAM-free `py_dfg_da` extension + the `pyehm`
fork (see `SETUP_AND_RUN.md` §1 and the root `requirements.txt`). Full runs used
`EVAL_WORKERS = nproc − 2` (30 on this machine). Run scripts with
`./.venv/bin/python` from the repo root.

---

## 1. Got the multi-cluster eval working with the IE LBP+Bethe method

**File: `ravens_parser_parallell_multicluster.py`** — replaced its outdated
contents (LBP-only, exact solver commented out, a hard `!= 10_000` file-count
crash) with the full four-method eval, porting the proven machinery that had been
developed in `ravens_parser_parallell_multicluster_not_exact.py`.

Per scan it now runs and times:

- **LBP** — multi-cluster multi-hypothesis LBP + Bethe (`py_dfg_da.lbp.lbp_multicluster`)
- **Exact EHM2** — `MulticlusterExactEHM2` (the truth)
- **Efficient marginals** — cluster-Bayes-tree `MulticlusterEfficientMarginals`
- **IE LBP+Bethe** — the inclusion-exclusion (overlapping event space, thesis
  Sec. 7.5.1) method from `cluster_partition/`, `mode="overlap_ie"`, with the
  compiled single-hypothesis LBP+Bethe inner solver (`CppLBPBetheSolver`).

The IE wiring (`CppLBPBetheSolver`, `ie_lbp_bethe_solver`, `_to_cp_hypotheses`,
and the `MulticlusterPartitionedMarginals(...)` call in `loop_func`) is the core
addition requested. Two correctness pitfalls handled: convert hypotheses to
`cluster_partition` types *before* any upstream solver reindexes them, and use
the *single-hypothesis* kernel + total-probability assembly.

**Full run result** (1396 scans, 30 workers, ~21.6 min):

| method | scans | median | max |
|---|---|---|---|
| LBP | 1396 | 1.87 ms | 0.05 s |
| Exact EHM2 | 1393 | 159 ms | 402 s |
| Efficient marginals | 1393 | 91 ms | 291 s |
| IE LBP+Bethe | 1393 | 203 ms | 659 s |

- Exact/efficient agreement: **1393/1393** scans (0 mismatches).
- 3 scans skipped by heavy solvers (`enum_product > MAX_ENUM_FOR_EXACT = 5e7`).

Outputs: `ravens_output_multicluster/metrics/metrics.csv` (+ `runtime_histogram`,
`metric_histograms`, `runtime_vs_enum` PNGs).

---

## 2. Accuracy of estimates vs. exact truth

**File: `accuracy_metrics_multicluster.py`** (new) — compares the approximate
methods against the exact marginals + normalization constant, reusing the eval
module's solvers (single source of truth) and `dfg_da.stats_logger`'s
`Marginals` / `MarginalsErrors` primitives (same convention as
`plotting_multicluster.py`).

Outputs in `ravens_output_multicluster/accuracy/`:

- `accuracy.csv` — per-scan `Z_exact/Z_lbp/Z_ie`, relative-Z errors, marginal
  error summaries, `cyclomatic`.
- `marginal_abs_error_hist`, `marginal_signed_error_hist`,
  `marginal_max_error_per_scan` — marginal-probability accuracy.
- `normconst_scatter`, `normconst_relerror_hist` — normalization-constant accuracy.
- `accuracy_vs_cyclomatic` — see §4.

**Aggregate over 1393 scans** (exact truth): IE LBP+Bethe is ~10× more accurate
on marginals and near-exact on `Z` (median relative-Z error **0.0028** vs LBP's
**0.50**).

---

## 3. Reproduced the thesis Chapter 10 figures

**File: `chapter10_plots.py`** (new) — reproduces the Ch. 10 (Results) figures of
the thesis (`~/Downloads/no.ntnu_inspera_140443607_46178819.pdf`) in their own
styles, using the two methods we have data for, mapped to the thesis names:

- **MCMH-LBP** = our `lbp_multicluster`
- **Approx Efficient Bethe** = our IE LBP+Bethe (`overlap_ie`)

Figures in `ravens_output_multicluster/chapter10/` (png + pdf each):

| fig | thesis content | what we produce |
|---|---|---|
| 10.1 | norm-const accuracy | `fig10_1_normconst_boxplot` (relative error, symlog-y) + `fig10_1_normconst_scatter` (exact vs approx `Z`, log-log, perfect-correlation line) |
| 10.2 | marginals correlation | `fig10_2_marginal_heatmap` — 2D-histogram heatmap (LogNorm/Reds), exact on y, per-method panel |
| 10.3 | survival of marginal errors | `fig10_3_survival_marginal_errors` — 5 panels (Max/Abs/Misdetection/Detection/Nonexistence), symlog-x linthresh 1e-6, log-y |
| 10.4 | hypothesis posterior `Pr{θ\|Z}` | `fig10_4_theta_posterior_heatmap` — correlation heatmap (MCMH-LBP only) |
| 10.5 | hyp-posterior signed error | `fig10_5_theta_posterior_signed_error` — histogram + symlog boxplot |

Scope note: Figs 10.4/10.5 show **MCMH-LBP only** — the IE `overlap_ie` solver
collapses θ to one value per cluster and doesn't expose an aligned per-hypothesis
posterior (exact and LBP θ's are verified per-cluster count-aligned). The thesis
also compared Approx Efficient PHD / Efficient MHLBP / Murty, which we don't
compute — as requested, we focused on *what* is plotted and *how*, with the
methods we have data for.

---

## 4. Cyclomatic number of the LBP graph + accuracy vs. loopiness

Added the **cyclomatic number** (circuit rank / first Betti number)
`μ = E − V + C` of the bipartite **track↔measurement association graph that LBP
runs message-passing on**: vertices = tracks + measurements, an edge for each
feasible association (finite `R_LC[i, j≥1]`), `C` = connected components.
`μ = 0` ⟺ forest ⟺ LBP exact; larger `μ` = more independent loops = expected
worse LBP.

- Helper `cyclomatic_number(R_LC)` added to
  `ravens_parser_parallell_multicluster.py` (single source of truth; uses
  `networkx`). Shared by all three scripts.
- New **`cyclomatic`** column in `metrics.csv` and `accuracy.csv`.
- New plots: `accuracy/accuracy_vs_cyclomatic.{png,pdf}` and
  `chapter10/fig_accuracy_vs_cyclomatic.{png,pdf}` — max marginal error and
  relative-`Z` error vs. `μ` (per-scan scatter + binned median/IQR).

**Result:** across all 1393 scans, accuracy degrades as `μ` grows exactly as
theory predicts. Approx Efficient Bethe is near-exact at low `μ` (errors ~1e-13)
and climbs with `μ`; MCMH-LBP is loop-sensitive even at modest `μ`.
corr(`μ`, relative-Z error) ≈ 0.53.

---

## Files added / changed

| file | status | purpose |
|---|---|---|
| `ravens_parser_parallell_multicluster.py` | rewritten | 4-method eval incl. IE method; `cyclomatic_number()` helper; `cyclomatic` metric |
| `accuracy_metrics_multicluster.py` | new | accuracy vs exact (marginals + `Z`); accuracy-vs-cyclomatic |
| `chapter10_plots.py` | new | thesis Ch. 10 figures + accuracy-vs-cyclomatic |
| `SESSION_SUMMARY.md` | new | this file |

## How to re-run

```bash
cd $REPO
W=$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))
V=pmbm-cm-python/.venv/bin/python
EVAL_WORKERS=$W "$V" -W ignore ravens_parser_parallell_multicluster.py   # metrics.csv (~21 min)
EVAL_WORKERS=$W "$V" -W ignore accuracy_metrics_multicluster.py          # accuracy/ (~15 min)
EVAL_WORKERS=$W "$V" -W ignore chapter10_plots.py                        # chapter10/ (~16 min)
```

(Note: zsh does not word-split unquoted variables — keep the interpreter path
quoted and pass `-W ignore` as a separate argument, don't stuff both into one var.)

## Open items / decisions to confirm

- **Cyclomatic graph choice:** `μ` is computed on the bipartite association graph
  (the BP-for-data-association factor graph the thesis builds on). Alternatives
  if preferred: per-supercluster `μ`, or the full multi-hypothesis factor graph
  including hypothesis nodes.
- **IE hypothesis posteriors:** not available from `overlap_ie` (θ collapsed per
  cluster), so Figs 10.4/10.5 are MCMH-LBP only.
- Data dir has **1396** scans, not the 1397 the scripts expect (soft warning,
  runs fine) — one edge/empty scan may be missing from `data/pmbm_output_files/`.
- Nothing has been committed; all changes are in the working tree.
