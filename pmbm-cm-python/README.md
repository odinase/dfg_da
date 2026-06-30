# PMBM filter (Python port)

Python port of the reference MATLAB implementation of the **Poisson
Multi-Bernoulli Mixture (PMBM)** multi-target tracking filter by Ángel F.
García-Fernández et al., found in `../pmbm-cm-matlab/PMBM filter/`.

Reference paper:

> Á. F. García-Fernández, J. L. Williams, K. Granström and L. Svensson,
> "Poisson Multi-Bernoulli Mixture Filter: Direct Derivation and
> Implementation," *IEEE Transactions on Aerospace and Electronic Systems*,
> vol. 54, no. 4, pp. 1883–1901, Aug. 2018.

Performance is measured with the GOSPA metric (`alpha = 2`) and its
decomposition into localisation, missed-target and false-target costs.

## What was ported

The clean, self-contained reference filter in the MATLAB `PMBM filter/`
subfolder. The hundreds of top-level "cluster management" scripts and
exploratory/plotting files in `pmbm-cm-matlab/` were **not** ported.

| MATLAB file                       | Python equivalent                |
| --------------------------------- | -------------------------------- |
| `PoissonMBMtarget_pred.m`         | `pmbm/filter.py` → `predict`     |
| `PoissonMBMtarget_update.m`       | `pmbm/filter.py` → `update`      |
| `PoissonMBMtarget_pruning.m`      | `pmbm/filter.py` → `prune`       |
| `PoissonMBMtarget_estimate1/2/3.m`| `pmbm/filter.py` → `estimate1/2/3` |
| `murty.m` + `assignmentoptimal.m` | `pmbm/murty.py` → `murty`        |
| `CardinalityMB.m`                 | `pmbm/cardinality.py`            |
| `GOSPA.m`                         | `pmbm/gospa.py` → `gospa`        |
| `ComputeGOSPAerror.m`             | `pmbm/gospa.py` → `compute_gospa_error` |
| `ScenarioWilliams15.m`            | `pmbm/scenario.py` → `build_scenario` |
| `TrajectoryWilliams15.m`          | `pmbm/scenario.py` → `williams15_trajectory` |
| `CreateMeasurement.m`             | `pmbm/scenario.py` → `create_measurement` |
| `DrawFilterEstimates.m`           | `pmbm/plotting.py`               |
| `PoissonMBMtarget_filter.m`       | `run_demo.py`                    |

## Install & run

```bash
pip install -r requirements.txt

# Run the Monte Carlo demo (10 runs, estimator 1) and save gospa_results.png
python run_demo.py

# Options
python run_demo.py --mc 20 --estimator 2 --seed 9 --no-plot
```

Run the component tests:

```bash
PYTHONPATH=. python tests/test_components.py
```

## Expected output

On the Williams-15 scenario the RMS GOSPA totals are roughly:

```
RMS GOSPA total            : ~2.9
RMS GOSPA localisation tot : ~1.8
RMS GOSPA false target tot : ~1.2
RMS GOSPA missed target tot: ~1.9
```

The per-time-step curves show high error during initial track establishment,
a steady-state localisation error of ~1.5–2 m, and a transient spike around
time step 41 where the first target dies.

## Running on the `scenario1MC.mat` dataset

`run_scenario_mat.py` runs the filter on the "9 ravens" benchmark dataset
(`../scenario1MC.mat`) from Brekke & Hem, *A long simulation scenario for
evaluation of multi-target tracking methods*, ICECCME 2023 — 8 targets in close
formation, 1397 Cartesian radar scans from a moving ownship, with ground truth
and clutter.

```bash
# Full 1397-step run, p_d = 0.9, estimator 3 (defaults)
python run_scenario_mat.py

# Quick sanity run over the first 30 steps
python run_scenario_mat.py --steps 30

# Options
python run_scenario_mat.py --mat ../scenario1MC.mat --pd 0.5 --estimator 2 \
    --steps 300 --outbase runs --no-dump --no-plot
```

`pmbm/scenario_mat.py` (`load_mat_scenario`) converts the MATLAB `params` /
`system` / `scenario` structs into the filter inputs:

* model matrices straight from `system` (`F=fMat, Q=qMat, H=hMat, R=rCart`);
  the state here is `[px, py, vx, vy]` (positions at indices 0,1);
* per-step measurements sliced from `scenario.zList` using `scenario.zCard`;
* clutter intensity `faRate / areaCircle` (= 0.00127);
* an **ownship-centred PPP birth**: a single wide Gaussian on a disk of radius
  `rMax = 100 m` centred on `stateFullOwn(:,k)` each step (the "flat-P"
  approximation from the MATLAB script);
* ground truth from `scenario.targetsTrue` (all 8 targets present for the whole
  run).

`compute_gospa_error` is called with `pos_idx=(0, 1)` for this layout.

### Per-step output dumps

Each run creates a fresh folder `runs/run_<timestamp>/` (configurable with
`--outbase`) so output never clutters an existing directory. Into it the run
writes:

* `priorLikelihood{k}.mat` for every step `k` (unless `--no-dump`) — the analog
  of `script_pmbm91.m:1236`. Because the reference filter has no
  cluster-management internals (`hypos`, `clusters`, …), each file instead holds
  the equivalent multi-Bernoulli-mixture state: `globHyp`, `globHypWeight`,
  `tracks` (a struct array with `meanB`, `covB`, `eB`, `aHis`, `weightBLog`,
  `t_ini`), the PPP component (`weightPois`, `meanPois`, `covPois`), plus
  `measurements`, `X_estimate`, `k`, `pD` and `ownship`;
* `results_mat.mat` — per-step GOSPA / decomposition / hypothesis-count arrays;
* `gospa_results_mat.png` — summary plot.

### Performance note

This is a deliberately hard, closely-spaced scenario. With the basic ported
`update` (see Deviation 1 below) and the flat-P birth, the filter establishes
tracks gradually and is subject to the track-coalescence behaviour the paper
itself documents for PMBM here, so it typically tracks ~6–8 of the 8 targets
with a few-metre localisation error rather than reproducing the paper's exact
GOSPA. Exact parity would require porting `PoissonMBMtarget_update_rPC`.

## State representation

The single-target state is `[px, vx, py, vy]` (positions at indices 0 and 2).
The filter state is a `dict`; see the module docstring in `pmbm/filter.py` for
the full description of `weightPois`, `meanPois`, `covPois`, `tracks`,
`globHyp` and `globHypWeight`.

All indexing is 0-based (MATLAB is 1-based). In `globHyp`, an entry of `-1`
denotes a track that is absent from that global hypothesis (MATLAB used `0`).

## Deviations from the MATLAB source

1. **k-best assignment.** `murty.py` implements Murty's algorithm using
   `scipy.optimize.linear_sum_assignment` as the base optimal-assignment
   solver, instead of the hand-written Munkres + bespoke partitioning in
   `murty.m`. It returns the same set of k-best assignments ranked by cost.

2. **GOSPA optimal assignment.** `gospa.py` uses
   `scipy.optimize.linear_sum_assignment` in place of the auction algorithm.
   This yields the same metric value.

3. **Position-index bug fix.** The original `ComputeGOSPAerror.m` extracts the
   target "position" from state rows `[1, 2]` (1-based) = `(px, vx)` rather
   than `(px, py)`, so the GOSPA never used the y-position. This port uses the
   true positions `(px, py)` for both ground truth and estimates. This is the
   only intentional behavioural fix; everything else mirrors the MATLAB.

4. **Random number generation.** Uses `numpy.random.default_rng(seed)`. Exact
   numerical reproduction of the MATLAB RNG streams is not possible, so
   per-realisation results differ, but the Monte Carlo statistics match.

5. **`scenario1MC.mat` update variant.** The Angel branch of
   `script_pmbm91.m` calls `PoissonMBMtarget_update_rPC` (a range-dependent
   variant). `run_scenario_mat.py` uses the basic ported `update` with
   `H=hMat`, `R=rCart` and constant `p_d` — the natural fit for the reference
   filter. See the performance note above.

# Cluster-management (CM) PMBM filter — `cm/` (work in progress)

The `cm/` package is a **separate** port of the *cluster-management* PMBM filter
(`../pmbm-cm-matlab/script_pmbm91.m`), whose goal is to reproduce the per-step
`priorLikelihood{k}.mat` dump written at `script_pmbm91.m:1236-1237`. This is a
different, much larger architecture than the reference filter above.

Because each dump's `hypos`/`clusters`/`probLogHypos` are carried over from the
previous step's branch-and-bound hypothesis generation, reproducing
`priorLikelihood{k}` for **k>1** requires the entire CM filter. The port is
therefore **staged**.

## Stage 1 (done): prior/likelihood pipeline + the k=1 dump

Stage 1 ports the per-step **prediction → validation → filtering → clustering**
pipeline and produces the **k=1** dump end-to-end (at k=1 all prior state is
empty, so `clusteringPreprocess` takes its trivial branch and the dump is
deterministic).

```bash
# Produce runs/run_<timestamp>/priorLikelihood1.mat
python run_cm.py --steps 1
```

Modules (`cm/`) and what they port:

| Module | MATLAB source |
| --- | --- |
| `cm/columns.py` | `inCol` track-file layout (`script:516-527`) |
| `cm/mathutils.py` | `covMat2Vec`, `covVec2Mat`, `v2m`, `normpdfLog`, `c2p`, `cmAlternative`, `gmReduce` |
| `cm/clouds.py` | `tCloud2BegInd/EndInd`, `a2bc`, `bc2a`, `removeIndA`, `pickIndC`, `track2Cluster`, `cluster2Tracks`, `union_sorted`, `probLogs2Probabilities` |
| `cm/validation.py` | `phdValidation`, `bernoulliValidation`, `trackProbAccumulatePure`, `pruneSelectedTracks` |
| `cm/filtering.py` | `trackFiltering` |
| `cm/clustering.py` | `clusteringPreprocess` (graph via `networkx`) |
| `cm/precompute.py` | `preClusterThreshold` recursion (`script:629-688`) |
| `cm/pipeline.py` | per-step orchestration (`script:966-1166`) |
| `cm/dump.py` | writes the exact 24-variable `priorLikelihood{k}.mat` |
| `cm/state.py`, `cm/loader.py`, `run_cm.py` | state, scenario loading, driver |

Tests: `PYTHONPATH=. python tests/test_cm_components.py` (math/index helpers)
and `PYTHONPATH=. python tests/test_cm_pipeline.py` (k=1 and k=2 dump structure).

## Stage 2a (done): step-1 carry-over → the k=2 dump

At k=1 there are no clusters, so `branchAndBoundExplore` is not called and the
reduction functions (`pruningPmbmBid`, `nScanMerge`, `clusterSplittingMajTraj`,
PHD recycling / `Runnalls` — with `doRunnall=false`, `doPostDMRecycle=false`)
are no-ops on the singleton-cluster state. `cm/carryover.py` therefore advances
the state after step 1 with just the newborn-clustering carry-over (each
measurement becomes its own singleton cluster), asserting the no-op
preconditions. This is enough to produce `priorLikelihood2.mat`, which first
exercises the non-empty prior/likelihood path (prediction of real tracks,
`trackProbAccumulatePure`, the adolescent/semi-two-point logic) and the heavy
graph branch of `clusteringPreprocess`.

```bash
python run_cm.py --steps 2   # writes priorLikelihood1.mat and priorLikelihood2.mat
```

## Stage 2b (done): branch-and-bound hypothesis generation → k≥3 dumps

The full k≥2 carry-over is implemented, so `priorLikelihood{k}.mat` is produced
for arbitrary `k`:

```bash
python run_cm.py --steps 3   # writes priorLikelihood1..3.mat
python run_cm.py --steps 12  # runs the full carry-over loop, 12 steps
```

New modules:

| Module | MATLAB source |
| --- | --- |
| `cm/bbhelpers.py` | `assign2D` / `crouse2D`, `reOrderC`, `insertElements`, `sortHyposInCluster`, `probLogs2ProbabilitiesForHypos`, the share/repeat invariant checks |
| `cm/branchbound.py` | `branchAndBoundExplore` + `pqInitialize`, `pqsolve`, `pqsolve2`, `pqAppend`, `pqAppendSwitch`, `constructSwitchReward`, `boundCalculate`, `testBAndCpqI` |
| `cm/reductions.py` | `pruningPmbmBid` (+`pruningAdjustClusters2`), `trackPruningPmbm`, `nScanMerge` (+`repeatedTracks`, `checkSameTrackInDifferentClusters`), `majorityTracks`, `clusterSplittingMajTraj` |

`cm/carryover.py` drives the per-supercluster `branchAndBoundExplore` →
newborn clustering → `pruningPmbmBid` → `nScanMerge` → `clusterSplittingMajTraj`
→ PHD recycling sequence (`doRunnall=false`, `doPostDMRecycle=false`).

Tests: `tests/test_cm_bbhelpers.py`, `tests/test_cm_branchbound.py`,
`tests/test_cm_reductions.py`, plus `test_k3_carryover_and_shapes` in
`tests/test_cm_pipeline.py`.

### Important caveats

- **Equivalent, not bit-exact.** There is no MATLAB to validate against, and the
  CM pipeline uses MATLAB graph algorithms (`centrality`, `conncomp`) and
  assignment tie-breaking that Python cannot reproduce bit-for-bit. Validation is
  by structural/shape equivalence and the ABC bookkeeping invariants; the
  branch-and-bound core is checked against hand-computed optimal hypotheses on
  small synthetic clusters.
- **Faithfully reproduced MATLAB quirks** (documented in the source): the
  constant-`cost` indexing `gainMatFull(newTrackSubs(1),newTrackSubs(2))` in
  `trackFiltering`; the `lambdauParents = find(isinf(...))` PHD-parent selection
  (which gives the new-track mixture's PHD components zero weight); and the
  `tRow(p.meaEntireOpt(tt)) = superMinVal` out-of-bounds auto-grow in the
  expansion heuristic of `branchAndBoundExplore` (reproduced as MATLAB's
  zero-padding grow).
- **Not ported:** the rare cross-cluster duplicate-recovery branch of
  `nScanMerge` (taken only when a track ends up in two clusters after merging) —
  it raises `NotImplementedError` if reached. It did not trigger over the first
  12 steps of the benchmark scenario. The `doRunnall=true` Runnalls mixture
  reduction is also omitted (it is disabled in the reference script).
