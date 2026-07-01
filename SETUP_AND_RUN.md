# Setup & run: PMBM data + multi-cluster data-association eval (with the IE LBP+Bethe method)

A complete, **standalone** runbook to reproduce, on a fresh machine, everything done in
this session:

1. build the environment (venv, the `py_dfg_da` C++ extension, `pyehm` fork),
2. regenerate the PMBM `priorLikelihood{k}.mat` data with the Python CM filter,
3. run the multi-cluster eval — LBP, exact EHM2, efficient marginals, **and the new
   inclusion-exclusion LBP+Bethe method using the compiled single-hypothesis C++ LBP** —
   in parallel, producing `metrics.csv` + plots.

It supersedes `REPRODUCE.md` (which is Linux/`python3.10`-specific and predates the IE
method). Where the two differ, follow this file. Commands assume the repo root as the
working directory unless stated; `$REPO` denotes that absolute path.

> **Environment update (single root venv):** the project now uses **one** venv at the repo
> root, `$REPO/.venv`, instead of `pmbm-cm-python/.venv`. `requirements.txt` at the repo
> root pins the full dependency set (incl. the `pyehm` fork). Quick install:
> ```bash
> cd $REPO
> python3.12 -m venv .venv
> .venv/bin/pip install -U pip
> .venv/bin/pip install -r requirements.txt          # replaces §1a/§1c below
> # then build py_dfg_da into it (§1b), using ../.venv instead of .venv paths:
> #   SUFFIX/SITE from ../.venv/bin/python, output into $REPO/.venv site-packages
> ```
> Wherever the sections below say `pmbm-cm-python/.venv/bin/python`, read `./.venv/bin/python`
> (e.g. `run_cm.py` in §3 becomes `./.venv/bin/python pmbm-cm-python/run_cm.py`, and the eval
> in §6 becomes `EVAL_WORKERS=$W ./.venv/bin/python -W ignore <script>.py`).

---

## 0. Before you clone: push the working-tree change

The eval script `ravens_parser_parallell_multicluster_not_exact.py` was modified in this
session (parallel `Pool` execution **and** the new IE LBP+Bethe method). Everything else
this runbook needs — `cluster_partition/`, `pmbm-cm-python/run_cm.py`,
`pmbm-cm-python/module_nogtsam.cpp`, `dfg_da/`, `src/`, `include/` — is already committed.

On **this** machine, commit and push the change so the other machine gets it:

```bash
cd $REPO
git add ravens_parser_parallell_multicluster_not_exact.py
git commit -m "Parallel eval + IE LBP+Bethe (C++ single-hypothesis) method"
git push
```

On the **other** machine, clone and check out the same branch (`pmbm-python-impl` here):

```bash
git clone <repo-url> dfg_da && cd dfg_da
git checkout pmbm-python-impl
# sanity: the parallel + IE method must be present
grep -q "EVAL_WORKERS" ravens_parser_parallell_multicluster_not_exact.py && \
grep -q "CppLBPBetheSolver" ravens_parser_parallell_multicluster_not_exact.py && \
echo "OK: parallel + IE method present"
```

The generated `.mat` data is **not** in git (it is regenerated in §3), so a fresh clone has
no `pmbm-cm-python/runs/` and no `data/pmbm_output_files/`.

---

## 1. Environment

### 1a. Python venv + dependencies

`python3.10` is ideal (matches `REPRODUCE.md`) but **any 3.10–3.13 works** — this session
used Homebrew `python3.13` because no 3.10 was available. Pick whatever you have:

```bash
cd $REPO
# Linux: python3.10  |  macOS (Homebrew): /opt/homebrew/bin/python3.13
PYBIN=python3.10                       # or python3.13, python3, ...
"$PYBIN" -m venv pmbm-cm-python/.venv
VENV=pmbm-cm-python/.venv/bin
"$VENV/pip" install -U pip
"$VENV/pip" install numpy scipy matplotlib networkx tqdm "pybind11[global]" pytest
```

> **Gotcha (Python ≥ 3.12):** the `pyehm` fork (§1c) imports `pkg_resources`, which modern
> `setuptools` (≥ 81) removed. Pin an older setuptools into the venv:
> ```bash
> "$VENV/pip" install "setuptools<80"
> ```
> On Python 3.10/3.11 this is usually unnecessary but harmless.

### 1b. Build the `py_dfg_da` C++ extension (GTSAM-free)

The CM scripts and the new IE solver use only `pdd.hypothesis.*` and `pdd.lbp.*` (from
`src/hypothesis.cpp` + `src/lbp.cpp`, Eigen + stdlib only). Compile the stripped
`pmbm-cm-python/module_nogtsam.cpp` directly into the venv's `site-packages`.

The extension filename suffix must come from `sysconfig` (**the venv has no
`python3-config`** — that bit us; an empty suffix builds an unimportable `py_dfg_da`).

**macOS (clang + Homebrew Eigen)** — what this session used:

```bash
cd $REPO/pmbm-cm-python
EIGEN="$(brew --prefix eigen)/include/eigen3"
SUFFIX="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
SITE="$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
clang++ -O3 -shared -std=c++17 -fPIC -undefined dynamic_lookup \
    $(.venv/bin/python -m pybind11 --includes) \
    -I../include -I"$EIGEN" \
    module_nogtsam.cpp ../src/hypothesis.cpp ../src/lbp.cpp \
    -o "$SITE/py_dfg_da$SUFFIX"
cd $REPO
```

**Linux (g++ + system Eigen):**

```bash
cd $REPO/pmbm-cm-python
SUFFIX="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
SITE="$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
g++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(.venv/bin/python -m pybind11 --includes) \
    -I../include -I/usr/include/eigen3 \
    module_nogtsam.cpp ../src/hypothesis.cpp ../src/lbp.cpp \
    -o "$SITE/py_dfg_da$SUFFIX"
cd $REPO
```

(If you have GTSAM + the submodules, `pip install .` per the root `README.md` also works;
the GTSAM-free build above avoids that dependency.)

### 1c. The `pyehm` fork (required for the exact EHM2 solver)

The exact solver needs `EHM2.run_and_likelihood`, which only exists in the fork:

```bash
pmbm-cm-python/.venv/bin/pip install \
    "git+https://github.com/odinase/pyehm.git@feature/add_return_likelihood"
```

### 1d. Verify the environment

```bash
cd $REPO
pmbm-cm-python/.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
import py_dfg_da, pyehm, numpy, scipy, networkx, matplotlib
from pyehm.core import EHM2
import dfg_da.marginals_computers, dfg_da.cluster_bayes_tree, dfg_da.stats_logger
import cluster_partition
print('py_dfg_da.lbp.lbp_single_cluster:', hasattr(py_dfg_da.lbp, 'lbp_single_cluster'))
print('EHM2.run_and_likelihood:', hasattr(EHM2, 'run_and_likelihood'))
print('all imports ok')
"
```

Expected: both `True` and `all imports ok`.

---

## 2. (Skip) why not reuse old `runs/`

Any pre-existing `pmbm-cm-python/runs/run_*` from before the two filter fixes is **not**
well-conditioned (large superclusters, `maxHyp/clus` up to ~140, enumeration up to 1e17).
Always regenerate (§3) and verify (§4). On a fresh clone there are no old runs anyway.

---

## 3. Regenerate the PMBM data

`run_cm.py` runs the CM PMBM prior/likelihood pipeline on `scenario1MC.mat` (the 1397-scan
"9 ravens" dataset at the repo root), dumping one `priorLikelihood{k}.mat` per scan into a
fresh timestamped run dir.

```bash
cd $REPO/pmbm-cm-python
.venv/bin/python run_cm.py --steps 2000        # --steps caps at Nsteps (=1397)
cd $REPO
```

It prints `Output directory: runs/run_<timestamp>` — **note that path**, call it `$RUN`.
A full run writes 1397 files and took ~13 min on this session's machine. For a quick smoke
test use `--steps 60` (~35 s). Key params live near the top of `run_cm.py`
(`NHYP_MAX=30`, gating, etc.).

```bash
RUN=pmbm-cm-python/runs/run_<timestamp>        # <- set to the printed path
ls "$RUN"/priorLikelihood*.mat | wc -l         # expect 1397
```

---

## 4. Verify the data is well-conditioned

```bash
cd $REPO
RUN=pmbm-cm-python/runs/run_<timestamp>        # <- set this
pmbm-cm-python/.venv/bin/python - "$RUN" <<'PY'
import sys, glob, numpy as np, scipy.io as sio
sys.path.insert(0, ".")
import dfg_da.stats_logger as sl, dfg_da.cluster_bayes_tree as cbt
from collections import Counter
RUN = sys.argv[1]
files = sorted(glob.glob(RUN + "/priorLikelihood*.mat"),
               key=lambda p: int("".join(c for c in p.split("priorLikelihood")[-1] if c.isdigit())))
worst_clus = worst_hyp = worst_enum = 0; nonempty = 0
for f in files:
    d = sio.loadmat(f); al = np.asarray(d["assocLocal"])
    if al.size == 0 or al.shape[1] == 0:        # guard empty scans
        continue
    nonempty += 1
    grp = Counter(al[0].astype(int).tolist())
    cc = np.asarray(d["clustersCard"]).ravel().astype(int)
    md = sl.MatFileParser(f, use_cpp=True)
    cl = cbt.ClusterLinks(R_LC=np.asfortranarray(md.reward_matrix_lc),
                          prior_hypotheses_per_cluster=md.prior_hypotheses_per_cluster, assocLocal=al)
    enum = 1
    for lm in cl.linking_mappings_per_merging_clusters():
        for _m, cs in lm.linking_measurements_to_clusters.items():
            enum *= (1 + len(cs))
    worst_clus = max(worst_clus, max(grp.values()))
    worst_hyp  = max(worst_hyp, int(cc.max()) if cc.size else 0)
    worst_enum = max(worst_enum, enum)
print(f"WORST over {nonempty} non-empty scans: maxClus/Super={worst_clus} "
      f"maxHyp/clus={worst_hyp} maxEnum={worst_enum:,}")
PY
```

Expected after the fixes: `maxClus/Super <= 5`, `maxHyp/clus <= 30`, `maxEnum` up to a
few×1e7 (this session saw `5 / 30 / 32,400,000`). If you see `maxClus/Super` in the teens
or `maxEnum` ~1e13+, the filter fixes are not active — do not proceed.

---

## 5. Point the eval data dir at the new run

The eval reads `./data/pmbm_output_files/*.mat`. **This directory does not exist in a fresh
clone — create it** (do not assume it is just empty), then symlink the run with absolute
paths:

```bash
cd $REPO
RUN="$REPO/pmbm-cm-python/runs/run_<timestamp>"   # <- absolute path
DEST="$REPO/data/pmbm_output_files"
mkdir -p "$DEST"
for f in "$RUN"/priorLikelihood*.mat; do ln -s "$f" "$DEST/$(basename "$f")"; done
ls "$DEST" | wc -l            # expect 1397
```

(Symlinks keep the large `.mat` files in the run dir. Copy instead if you prefer.)

---

## 6. Run the evaluation (parallel)

```bash
cd $REPO
# workers = cores - 2 (leave 2 free), floor 1; or hardcode EVAL_WORKERS=8
CORES=$( (nproc 2>/dev/null || sysctl -n hw.ncpu) )
EVAL_WORKERS=$(( CORES > 2 ? CORES - 2 : 1 ))

EVAL_WORKERS=$EVAL_WORKERS pmbm-cm-python/.venv/bin/python -W ignore \
    ravens_parser_parallell_multicluster_not_exact.py
```

Per scan it runs, then times: multi-cluster **LBP**, **exact EHM2**, **efficient
marginals**, and the new **IE LBP+Bethe** (see §8); it cross-checks exact vs efficient and
writes metrics + plots. Notes:

- `EVAL_WORKERS` (env var, default 8) sets the process-pool size. This session changed the
  `__main__` loop from serial to `Pool(...).imap_unordered`. Parallelism speeds the run but
  **inflates the per-method `t_*` runtimes** under CPU contention (topology and
  exact/efficient-agreement metrics are unaffected). For clean timings, set
  `EVAL_WORKERS=1`.
- `MAX_ENUM_FOR_EXACT` (top of the script, default `5e7`) caps the exact/efficient/IE
  conditioning enumeration; heavier scans are LBP-only and reported as "heavy-solver
  skips", so the run can never hang.
- `-W ignore` silences benign `RuntimeWarning`s (intermediate `0/0` in zero-likelihood
  branches; final marginals are NaN-free).
- The IE method is pure-Python in its enumeration loop, so heavy scans are slow (tens of
  seconds) regardless of LBP backend; the full run takes meaningfully longer than the
  LBP/exact/efficient-only run.

Check progress while it runs (it prints a tqdm bar to stderr):

```bash
# if you redirected to a log: tail -c 120 <logfile>
# confirm worker count (≈ EVAL_WORKERS python children, each ~100% CPU):
MAIN=$(pgrep -f ravens_parser_parallell_multicluster | head -1)
ps -o pid,%cpu -p $(pgrep -P "$MAIN" | tr '\n' ' ')
```

---

## 7. Outputs & metrics

```
pmbm-cm-python/runs/run_<ts>/priorLikelihood{1..1397}.mat   regenerated filter data
data/pmbm_output_files/priorLikelihood{1..1397}.mat          symlinks the eval reads
ravens_output_multicluster/metrics/metrics.csv               per-scan metrics
ravens_output_multicluster/metrics/runtime_histogram.png     per-method runtime dist.
ravens_output_multicluster/metrics/metric_histograms.png     enum size / supercluster stats
ravens_output_multicluster/metrics/runtime_vs_enum.png       runtime vs enumeration size
```

`metrics.csv` columns include the standard ones plus the new IE method:

| column | meaning |
|---|---|
| `t_lbp`, `t_exact`, `t_efficient` | per-method runtime (s); `NaN` if skipped |
| `t_ie_lbp_bethe` | runtime of the IE LBP+Bethe method (s) |
| `like_ie_lbp_bethe` | the IE method's (Bethe) normalization constant |
| `marg_match`, `like_match` | exact vs efficient agreement for that scan |
| `enum_product`, `n_superclusters`, `max_clusters_per_supercluster`, … | topology |

The summary printed at the end now has four method lines (LBP / Exact EHM2 / Efficient
marginals / IE LBP+Bethe).

---

## 8. The new method: IE LBP+Bethe with the compiled single-hypothesis LBP

Added to `ravens_parser_parallell_multicluster_not_exact.py`. It is the
linking-measurement / cluster-conditioning marginalization from `cluster_partition/`, run in
the **overlapping event space with inclusion-exclusion** (`mode="overlap_ie"`, thesis Sec.
7.5.1), with an inner solver backed by the **compiled** `py_dfg_da` LBP applied **per single
hypothesis** and combined by total probability (Eq. (7.16)).

Pieces:

- `MulticlusterPartitionedMarginals(R_LC, hyps, solver, mode="overlap_ie")` from
  `cluster_partition` — the inclusion-exclusion driver. `overlap_ie` reproduces the exact
  normalization constant with an exact inner solver and de-biases an LBP inner solver
  (`overlap_firststage` overcounts; `disjoint_exact` is the upstream efficient method).
- `CppLBPBetheSolver` (defined in the eval script) — satisfies the `cluster_partition`
  inner-solver contract `solver(R_cluster, prior_hypotheses, enforce_meas=(), reindex=True)
  -> (marginals (n, m+2), theta_posterior, likelihood)`. For **each** prior hypothesis it
  builds the hypothesis's existing-track sub-matrix, converts the track-oriented "lc" layout
  to the "edmund" layout the C++ wants (`[meas (m) | n×n existence block]`, misdetection on
  the diagonal, `-inf` off it — verified bit-exact against
  `MatFileParser.reward_matrix_edmund`), calls `py_dfg_da.lbp.lbp_single_cluster` with a
  **one-element** hypothesis list (so it reduces to single-hypothesis JPDA LBP), and lets the
  shared `cluster_partition.solvers._assemble_multihypothesis` combine the per-hypothesis
  Bethe constants by total probability.

### Two correctness pitfalls (already handled in the committed code)

1. **Convert hypotheses *before* any upstream solver runs.** The upstream LBP/exact/efficient
   solvers call `reindex_tracks()`, which **mutates** the compiled hypotheses in place
   (global → per-cluster-local track ids). The eval snapshots them to `cluster_partition`
   types at the top of `loop_func` (`cp_prior_hypotheses = _to_cp_hypotheses(...)`) before
   that happens. Converting afterwards yields wrong global track ids and a garbage constant.
2. **Single-hypothesis, not multi-hypothesis, kernel.** Calling `lbp_single_cluster` with the
   *full* multi-hypothesis list runs Bethe over the whole hypothesis structure at once and
   gives a biased constant (~0.6–0.7× exact). The per-single-hypothesis + total-probability
   assembly above matches the pure-Python Bethe solver and de-biases to ~exact (verified:
   IE constant within 0–2.4% of the exact Z across sampled scans).

### Sanity-check the IE method (small, fast)

```bash
cd $REPO
pmbm-cm-python/.venv/bin/python -W ignore - <<'PY'
import sys; sys.path.insert(0, ".")
import importlib.util
spec = importlib.util.spec_from_file_location("ev", "ravens_parser_parallell_multicluster_not_exact.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("IE inner solver:", type(m.ie_lbp_bethe_solver).__name__)   # CppLBPBetheSolver
for fn in ["priorLikelihood101.mat", "priorLikelihood51.mat"]:
    r = m.loop_func("data/pmbm_output_files/" + fn)
    print(fn, "like_ie=%.6g" % r["like_ie_lbp_bethe"],
          "exact/eff match:", r["marg_match"], r["like_match"])
PY
```

Expected: `CppLBPBetheSolver`, `like_ie` close to the exact Z, and both matches `True`.

To compare against the **pure-Python** Bethe inner solver instead, swap
`ie_lbp_bethe_solver = CppLBPBetheSolver()` for `LBPBetheSolver()` (import it from
`cluster_partition`) — both should give ~the same IE constant.

---

## 9. Troubleshooting

- **`No module named py_dfg_da`** — extension not built into the venv; redo §1b. If a
  suffix-less `py_dfg_da` file exists in `site-packages`, delete it; the `EXT_SUFFIX` step
  is mandatory.
- **`No module named pkg_resources`** (importing pyehm) — `setuptools` too new; install
  `"setuptools<80"` (§1a gotcha).
- **`EHM2 has no attribute run_and_likelihood`** — install the pyehm **fork** (§1c), not
  pip `pyehm`.
- **`No module named cluster_partition`** — run the eval from the **repo root** (the package
  lives there and is found via `sys.path[0]`), not from inside a subdirectory.
- **`No .mat files found under ./data/pmbm_output_files`** — you skipped §5; the directory
  must be created and populated (it is absent in a fresh clone).
- **IE constant looks wrong (~0 or ~0.6× exact)** — the two pitfalls in §8: convert
  hypotheses before upstream solvers run, and use the single-hypothesis kernel. Both are in
  the committed code; only relevant if you modify the solver.
- **Eval too slow** — lower `EVAL_WORKERS` only changes contention, not work; the heavy
  scans (large `enum_product`) dominate. Lower `MAX_ENUM_FOR_EXACT` to skip the heaviest
  scans for the exact/efficient/IE solvers (LBP still timed), or trim the file list in
  `__main__`.
```
