# RESOLVED 2026-06-30: supercluster balancing loop ported — timeout fixed

The real fix is in. Ported `pmbm-cm-matlab/clusteringPreprocess.m` lines 74-289 (the
iterative supercluster size/cardinality balancing loop) into
`pmbm-cm-python/cm/clustering.py:_clustering_heavy`. After regenerating
(`run_cm.py --steps 200` → `runs/run_20260630_162346`):

- superclusters per scan: 1 → 3-11; max clusters/supercluster ≤ 5 (was up to 20);
- enumeration `∏(1+#clusters per linking measurement)`: 10^13-10^15 → ≤ 1.2e7;
- `MulticlusterExactEHM2` + `MulticlusterEfficientMarginals` now complete on every
  probed file (exact 0.003-2.5s, efficient 0.004-20s) and agree under strict
  `np.allclose` (marginals + likelihood), no NaN in final output;
- also fixed `ravens_parser_parallell_multicluster_not_exact.py:80` (was unpacking 2
  values from a 3-tuple) so its real `loop_func` runs end-to-end;
- new tests: `pmbm-cm-python/tests/test_cm_clustering.py` (2 passing); full suite 44 passed.

Not bit-exact vs MATLAB (verification is behavioral). Full details in memory note
`dfg-da-scripts-env.md` ("RESOLVED 2026-06-30" paragraph).

---

# STATUS 2026-06-30: nHypoMax cap implemented, but it alone did NOT fix the timeout

The per-cluster `nHypoMax` cap below was implemented (`cm/reductions.py:pruning_pmbm_bid`
n-best cut + `cm/carryover.py` threading) and **verified working** (regenerated run
`pmbm-cm-python/runs/run_20260630_155200`: max 30 hypotheses/cluster, tracks 92→34).
**Keep it** — it correctly matches the thesis pruning and is a prerequisite.

**But it does not resolve the exact-solver timeout.** Measurement disproved the premise
that capping hypotheses thins gating enough to shrink superclusters:

- Every scan still collapses into ONE supercluster (`nSuper=1`) holding all 8–20 prior
  clusters. The EHM2 enumeration `∏(1+#clusters per linking measurement)` depends only on
  this topology, not on hypotheses/cluster — it stays 10^13–10^15 (NEW file10 = 1.36e13,
  *worse* than OLD 3.35e9: pruning left 10→20 smaller clusters).

**Real root cause: `cm/clustering.py:_clustering_heavy` (lines 133-135).** The MATLAB
"iterative super-cluster size/cardinality balancing loop (script_pmbm91 lines ~74-289)"
is NOT ported. That loop keeps superclusters at the thesis's ~3 clusters / ~2.6 linking
measurements each; without it superclusters grow unbounded. (That whole heavy graph branch
is also flagged "unvalidated against MATLAB.") **Next fix to chase: port that balancing
loop**, not more hypothesis pruning. Fast tractability check without running the solver:
compute the enumeration product via `cbt.ClusterLinks(...).linking_mappings_per_merging_clusters()`
(cross-check: OLD priorLikelihood10.mat → 3,353,011,200).

---

# Fix plan (DONE): restore per-cluster hypothesis cap (`nHypoMax`) in the Python CM pipeline

## Problem

The Python CM pipeline (`pmbm-cm-python/`, driven by `run_cm.py`) generates
`priorLikelihood{k}.mat` files that are **not pruned**: clusters keep every
generated hypothesis (observed 68–150 per cluster) instead of being capped at
`nHypoMax = 30`. This inflates retained tracks, which densifies measurement
gating, which produces oversized superclusters. The downstream exact
data-association solvers (`MulticlusterExactEHM2` and
`MulticlusterEfficientMarginals` in `dfg_da/`) assume pruned input and therefore
blow up (timeout > 90 s, ~35 GB RAM) on this data.

This is **not** a bug in the solvers or in the `∏(1+#clusters per linking
measurement)` enumeration count (verified correct to the digit, e.g.
`3,353,011,200` for `priorLikelihood10.mat`). It is missing pruning upstream.

### Evidence vs. the reference dataset (thesis Ch. 9, Table 9.1 — the "9 ravens" set)

| Metric | Thesis Table 9.1 (validated) | Our generated data (file 10) |
|---|---|---|
| Avg linking measurements / supercluster | 2.663 | 11 |
| Avg prior clusters / supercluster | 3.081 | 10 |
| Avg prior hypotheses / prior cluster | 23.395 | 68 (max in file) |
| Pruning limit (total hypotheses) | 150 | violated (merged ≈ 15,600) |

Thesis PDF: `~/Downloads/no.ntnu_inspera_140443607_46178819.pdf`, Chapter 9
("Introduction to the dataset used for testing"), Table 9.1 and Figure 9.2
("Pruning limit of 150 hypotheses").

## Root cause (exact location)

`pmbm-cm-python/cm/reductions.py` → `pruning_pmbm_bid()` (port of
`pruningPmbmBid.m`). The MATLAB "bid loop" that marks low-probability hypotheses
for removal was **deliberately omitted** (see the module docstring). Concretely:

```python
to_be_removed = np.zeros(int(np.sum(clusters_card)), dtype=bool)  # stays all-False
to_be_kept    = np.arange(1, n_clusters_hyp + 1)                  # keeps ALL hypotheses
```

- The per-cluster cap `nHypoMax` (=30) is **never applied** anywhere in `cm/`
  (it is only stored in config and dumped to the `.mat`).
- The `n_hypo_total_max` argument is **passed** (`cm/carryover.py:175`) but
  **unused** in the body.
- Only the *global* `nHypoTotalMax` cap survives, in `cm/branchbound.py`
  (bounds combined frame hypotheses, not per-cluster counts).

Helpful precondition: `sort_hypos_in_cluster()` (`cm/bbhelpers.py`) already sorts
hypotheses **descending by log-prob within each cluster**, so an n-best cut is
trivial.

## The fix

1. **Thread the cap in.** Add a `n_hypo_max` parameter to
   `pruning_pmbm_bid(...)` in `cm/reductions.py`. Pass it from
   `cm/carryover.py` (the value is already available as `int(config["nHypoMax"])`,
   next to the existing `n_hypo_total_max = int(config["nHypoTotalMax"])` at
   `carryover.py:93` and the call at `carryover.py:172-175`).

2. **Implement the n-best cut.** In `pruning_pmbm_bid`, after the existing
   `sort_hypos_in_cluster(...)` call, replace the all-False `to_be_removed`
   stub: for each cluster `c`, keep the first `min(clusters_card[c], n_hypo_max)`
   hypotheses (already the most probable, due to the descending sort) and set
   `to_be_removed = True` for the rest. Use `tcloud_to_beg(clusters_card)` /
   `tcloud_to_end(clusters_card)` to get per-cluster hypothesis spans, and build
   `to_be_kept = np.nonzero(~to_be_removed)[0] + 1`. Leave everything else
   (`pruning_adjust_clusters2`, unsupported-track removal + renumbering, the
   empty-two-hypothesis simplification) unchanged — it already operates off
   `to_be_kept` / `to_be_removed`.

3. **(Optional) Total cap.** If after step 2 the supercluster stats still run
   hot, also enforce the supercluster-level `nHypoTotalMax = 150` reenumeration
   limit (the thesis "pruning limit of 150"). Hold until step-4 numbers say so.

## Verification

1. Regenerate a modest run:
   ```bash
   cd /home/odin/dfg_da/pmbm-cm-python
   .venv/bin/python run_cm.py --steps 200
   ```
2. Re-point `./data/pmbm_output_files` symlinks at the new run dir.
3. Re-run the supercluster/hypothesis probes (scratchpad
   `probe_supercluster.py` and the hypothesis-count probe) and confirm the
   regenerated data matches Table 9.1 order-of-magnitude: ≤ 30 hypotheses/cluster,
   linking measurements ≈ 2–3, prior clusters/supercluster ≈ 3.
4. Confirm scripts run to completion on the regenerated data:
   - `ravens_parser_parallell_multicluster.py` (LBP) — already runs.
   - `ravens_parser_parallell_multicluster_not_exact.py` (exact EHM2 +
     efficient) — should now finish per file instead of timing out.

## Related changes already made this session (keep)

- `dfg_da/marginals_computers.py` — int-cast on `assocLocal` in
  `ClusterHypothesesPosterior.__init__` (fixes float-index `IndexError`).
- Installed the **pyehm fork** (`git+https://github.com/odinase/pyehm.git@feature/add_return_likelihood`)
  into `pmbm-cm-python/.venv` (provides `EHM2.run_and_likelihood`).
- Built GTSAM-free `py_dfg_da` into the venv (see memory note below for recipe).
- Build artifact `pmbm-cm-python/module_nogtsam.cpp`; staged `data/` and
  `ravens_output/` dirs.

## Memory pointer

The persistent project memory for this work is:

- **`/home/odin/.claude/projects/-home-odin-dfg-da/memory/dfg-da-scripts-env.md`**
  (indexed in `MEMORY.md` as "dfg_da scripts env").

That note contains the GTSAM-free `py_dfg_da` build recipe, the env/deps, the
`assocLocal` cast and pyehm-fork details, and the root-cause analysis summarized
here. To find this plan again, see the "ROOT CAUSE FOUND" paragraph in that
memory note, which points to this `FIX_PLAN.md`.
