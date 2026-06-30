# Reproducing the PMBM data and running the data-association evaluation

This document describes the **end-to-end workflow** to (1) regenerate the PMBM
`priorLikelihood{k}.mat` prior/likelihood files from the "9 ravens" scenario with
the Python cluster-management (CM) filter, and (2) run the multi-cluster
data-association methods (LBP, exact EHM2, efficient/cluster-Bayes-tree) over that
data and produce timing + topology metrics with plots.

All commands assume the repo root `dfg_da/` as the working directory and the
virtualenv at `pmbm-cm-python/.venv` (Python 3.10). Adjust paths to taste.

---

## 0. What gets produced

```
pmbm-cm-python/runs/run_<timestamp>/priorLikelihood{1..1397}.mat   <- regenerated filter data
data/pmbm_output_files/priorLikelihood{1..1397}.mat                 <- symlinks the eval scripts read
ravens_output_multicluster/metrics/metrics.csv                      <- per-scan metrics table
ravens_output_multicluster/metrics/runtime_histogram.png           <- per-method runtime distribution
ravens_output_multicluster/metrics/metric_histograms.png           <- enum size / supercluster stats
ravens_output_multicluster/metrics/runtime_vs_enum.png             <- runtime vs enumeration size
```

---

## 1. Environment

### 1a. Python venv + dependencies

```bash
cd /home/odin/dfg_da
python3.10 -m venv pmbm-cm-python/.venv
pmbm-cm-python/.venv/bin/pip install -U pip
pmbm-cm-python/.venv/bin/pip install numpy scipy matplotlib networkx tqdm "pybind11[global]" pytest
```

### 1b. The `py_dfg_da` C++ extension

The data-association library is a pybind11 C++ extension. **Standard build** (needs
the GTSAM submodule + system GTSAM installed — see the root `README.md`):

```bash
git submodule update --init --recursive
pmbm-cm-python/.venv/bin/pip install .
```

**GTSAM-free build (what was used in this sandbox).** Only the `factor_graph`
submodule needs GTSAM; the CM scripts use only `pdd.hypothesis.*` and `pdd.lbp.*`,
which come from `src/hypothesis.cpp` + `src/lbp.cpp` (Eigen + stdlib only). A
stripped `pmbm-cm-python/module_nogtsam.cpp` (GTSAM includes / `gtsam_test()` /
`factor_graph` submodule removed) is compiled directly into the venv:

```bash
cd /home/odin/dfg_da/pmbm-cm-python
g++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(.venv/bin/python -m pybind11 --includes) \
    -I../include -I/usr/include/eigen3 \
    module_nogtsam.cpp ../src/hypothesis.cpp ../src/lbp.cpp \
    -o .venv/lib/python3.10/site-packages/py_dfg_da$(.venv/bin/python3-config --extension-suffix)
cd /home/odin/dfg_da
```

### 1c. The pyehm fork (required for the exact solver)

The exact EHM2 solver needs `EHM2.run_and_likelihood`, which only exists in the
fork (it additionally returns the hypothesis-conditioned normalization constant):

```bash
pmbm-cm-python/.venv/bin/pip install \
    "git+https://github.com/odinase/pyehm.git@feature/add_return_likelihood"
```

### 1d. Verify the environment

```bash
pmbm-cm-python/.venv/bin/python -c "import py_dfg_da, pyehm, numpy, scipy, networkx; print('ok')"
cd pmbm-cm-python && .venv/bin/python -m pytest tests -q   # 44 passed (3 pre-existing collection errors are harmless)
cd ..
```

---

## 2. Regenerate the PMBM data (the filter run)

`run_cm.py` runs the CM PMBM prior/likelihood pipeline on `scenario1MC.mat`
(the 1397-scan "9 ravens" dataset at the repo root) from an empty initial state,
dumping one `priorLikelihood{k}.mat` per scan.

```bash
cd /home/odin/dfg_da/pmbm-cm-python
.venv/bin/python run_cm.py --steps 2000      # --steps caps at Nsteps (=1397); 2000 => full run
cd ..
```

This prints `Output directory: runs/run_<timestamp>` — note that path. A full run
writes 1397 files. For a quick smoke test use `--steps 200`.

Key filter parameters live near the top of `run_cm.py` (mirroring
`script_pmbm91.m`): `NHYP_MAX=30` (per-cluster hypothesis cap), `NHYP_TOTAL_MAX=150`,
gating `GAMMA_GATE=9`, etc. Two fixes in `pmbm-cm-python/cm/` are what make the
generated data tractable for the exact solvers (see [§7](#7-context-the-two-fixes)).

---

## 3. Point the evaluation data directory at the new run

The evaluation scripts read from `data/pmbm_output_files/` (their
`PMBM_DATA_PATH`). Repoint that directory's symlinks at the run you just generated:

```bash
cd /home/odin/dfg_da/data/pmbm_output_files
NEW=/home/odin/dfg_da/pmbm-cm-python/runs/run_<timestamp>   # <- use the path printed in step 2
rm -f priorLikelihood*.mat
for f in "$NEW"/priorLikelihood*.mat; do ln -s "$f" "$(basename "$f")"; done
ls | wc -l        # expect 1397
cd /home/odin/dfg_da
```

(Symlinks keep the large `.mat` files in the run dir; you can also copy them if you
prefer.)

---

## 4. Run the evaluation

### Multi-cluster exact + efficient + LBP (with metrics & plots)

`ravens_parser_parallell_multicluster_not_exact.py` runs, per scan: multi-cluster
LBP, the exact merged-hypothesis solver (`MulticlusterExactEHM2`), and the
memory-efficient cluster-Bayes-tree solver (`MulticlusterEfficientMarginals`); it
cross-checks that exact and efficient agree, then writes timing + topology metrics
and plots.

```bash
cd /home/odin/dfg_da
pmbm-cm-python/.venv/bin/python -W ignore ravens_parser_parallell_multicluster_not_exact.py
```

- Runs over **all** files in `data/pmbm_output_files` (the old hard requirement of
  exactly 10,000 files is now a soft warning).
- `MAX_ENUM_FOR_EXACT` (top of the script, default `5e7`) caps the exact/efficient
  enumeration: any scan whose conditioning enumeration exceeds it is timed for LBP
  only and reported as a "heavy-solver skip", so the run can never hang.
- `-W ignore` silences benign `RuntimeWarning`s (intermediate `0/0` in
  zero-likelihood branches; the final marginals are NaN-free).

At the end it prints a summary like:

```
Processed N non-empty scans in T s
  LBP                 : n=...  median=... ms  mean=... ms  max=... s  total=... s
  Exact EHM2          : n=...  ...
  Efficient marginals : n=...  ...
  Heavy-solver skips (enum > 5e+07): ...
  Exact enumeration errors          : ...
  Exact/efficient agreement         : K/K scans
```

### The other two parser scripts (optional)

- `ravens_parser_parallell.py` — single-cluster.
- `ravens_parser_parallell_multicluster.py` — multi-cluster LBP variants.

Both read the same `data/pmbm_output_files` directory.

---

## 5. Interpreting the metrics

`ravens_output_multicluster/metrics/metrics.csv` has one row per non-empty scan:

| column | meaning |
|---|---|
| `num_clusters` | prior clusters in the scan |
| `n_superclusters` | superclusters after gating/merging (`sum(assocLocal[1])`) |
| `max_clusters_per_supercluster` | largest supercluster (should be **≤ 5** after the fix) |
| `n_tracks`, `n_measurements` | reward-matrix dimensions |
| `n_linking_measurements` | measurements gating ≥2 clusters |
| `enum_product` | exact-conditioning enumeration `∏(1+#clusters per linking meas)` |
| `t_lbp`, `t_exact`, `t_efficient` | per-method runtime (s); `NaN` if skipped |
| `exact_enum_error` | exact solver hit `ExplicitHypothesisEnumerationError` |
| `skipped_heavy` | enumeration exceeded `MAX_ENUM_FOR_EXACT` |
| `marg_match`, `like_match` | exact vs efficient agreement for that scan |

Plots: `runtime_histogram.png` (log-x per-method runtimes), `metric_histograms.png`
(enumeration size, largest supercluster, #superclusters, #linking measurements),
`runtime_vs_enum.png` (runtime vs enumeration size, log-log).

**Exact vs efficient:** both compute the *same* exact marginals/likelihood. "Exact"
merges each supercluster and walks the merged-hypothesis list with the compiled
EHM2 kernel; "efficient" is *memory*-efficient — it conditions on the linking
measurements (Python loop over `enum_product`) to avoid materialising the joint.
On these instances efficient is slower in wall-clock but bounded in memory.

---

## 6. Sanity-checking that the fixes are active

After regenerating, confirm the data is well-conditioned (this is the fast way to
verify without running the solvers):

```bash
cd /home/odin/dfg_da
pmbm-cm-python/.venv/bin/python - <<'PY'
import sys, glob, numpy as np, scipy.io as sio
sys.path.insert(0, ".")
import dfg_da.stats_logger as sl, dfg_da.cluster_bayes_tree as cbt
from collections import Counter
RUN = "pmbm-cm-python/runs/run_<timestamp>"   # <- set this
files = sorted(glob.glob(RUN + "/priorLikelihood*.mat"),
               key=lambda p: int("".join(c for c in p.split("priorLikelihood")[-1] if c.isdigit())))
for i in [10, 50, 100, 150]:
    f = files[i]
    d = sio.loadmat(f); al = np.asarray(d["assocLocal"])
    grp = Counter(al[0].astype(int).tolist())
    cc = np.asarray(d["clustersCard"]).ravel().astype(int)
    md = sl.MatFileParser(f, use_cpp=True)
    cl = cbt.ClusterLinks(R_LC=np.asfortranarray(md.reward_matrix_lc),
                          prior_hypotheses_per_cluster=md.prior_hypotheses_per_cluster,
                          assocLocal=al)
    enum = 1
    for lm in cl.linking_mappings_per_merging_clusters():
        for _m, cs in lm.linking_measurements_to_clusters.items():
            enum *= (1 + len(cs))
    print(f"{f.split('/')[-1]:>22}: nSuper={int(al[1].sum()):2d} "
          f"maxClus/Super={max(grp.values()):2d} maxHyp/clus={cc.max():3d} enum={enum:,}")
PY
```

Expected after the fix: `maxClus/Super ≤ 5`, `maxHyp/clus ≤ 30`, `enum` from the
hundreds up to a few×1e7 across the dataset (not 1e13+ as before the fix). Scans
above `MAX_ENUM_FOR_EXACT` are the ones the eval run skips for the heavy solvers.

---

## 7. Context: the two fixes

The generated data is only tractable for the exact solvers because of two changes
in `pmbm-cm-python/cm/` (both ports of behaviour that was missing from the Python
pipeline; details in `FIX_PLAN.md`). These are what `§2`'s data generation relies
on:

1. **Per-cluster hypothesis cap** (`cm/reductions.py:pruning_pmbm_bid` +
   `cm/carryover.py`): enforces `nHypoMax=30` as an n-best cut so clusters keep at
   most 30 hypotheses (was 68–150).
2. **Supercluster size/cardinality balancing loop**
   (`cm/clustering.py:_clustering_heavy`, port of `clusteringPreprocess.m`
   74–289): iteratively cuts the weakest inter-cluster edges until every
   supercluster has ≤5 clusters (previously every scan collapsed into one giant
   supercluster, making the exact enumeration 1e13–1e15 and the solvers time out /
   exhaust memory).

These are behaviorally verified, not bit-exact against MATLAB.

---

## 8. Troubleshooting

- **`No module named py_dfg_da`** — the extension isn't built/installed into the
  venv; redo [§1b](#1b-the-py_dfg_da-c-extension).
- **`EHM2 has no attribute run_and_likelihood`** — install the pyehm **fork**
  ([§1c](#1c-the-pyehm-fork-required-for-the-exact-solver)), not pip `pyehm`.
- **`could not find library libmetis-gtsam.so`** (standard build only) — see the
  root `README.md`: `sudo make install && sudo ldconfig` in the GTSAM build dir, or
  set `LD_LIBRARY_PATH`.
- **The eval run is slow** — high-cluster scans dominate; lower
  `MAX_ENUM_FOR_EXACT` in the script to skip the heaviest scans, or run on a subset
  by trimming the file list in `__main__`.
