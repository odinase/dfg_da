# Build from scratch (fresh machine)

The single authoritative runbook to build **everything** in this repo on a new computer:

1. the Python env (venv + `requirements.txt`, incl. the `pyehm` fork),
2. the `py_dfg_da` C++ extension (GTSAM-free),
3. the Rust workspace — C API → `bindgen` → safe wrappers → the `dfg_da_py` pyo3 module,
4. computing metrics from **existing** PMBM output data.

Two paths: **native** (§1–§7) or **Docker** (§8). `$REPO` = the repo root.

> For the deeper PMBM data *regeneration* pipeline (running the CM filter to produce fresh
> `priorLikelihood*.mat`), see `SETUP_AND_RUN.md`. That is **not** required here — this doc
> assumes you already have output data (or a partner produces it) and you want to build the
> code and compute metrics from it.

---

## 1. System prerequisites

Confirmed toolchain: Ubuntu 24.04, Python 3.12.3, g++ 13.3, CMake 3.31, Eigen 3.4.0, Rust 1.96.

**Ubuntu / Debian:**
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential g++ cmake ninja-build libeigen3-dev \
    python3.12 python3.12-venv python3.12-dev \
    clang libclang-dev patchelf \
    git curl ca-certificates
```
- `libclang-dev` is **required by Rust `bindgen`** — the `dfg-da-sys` build fails to parse the
  C header without it. This is the easiest prerequisite to miss.
- `libeigen3-dev` supplies Eigen at `/usr/include/eigen3` (used by both the C++ extension and
  the Rust C-API lib). `patchelf` silences a benign maturin rpath warning.

**Rust** (any recent stable; 1.96 used here):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.96.0
. "$HOME/.cargo/env"
```

**macOS (Homebrew) equivalents:** `brew install eigen cmake ninja llvm rustup-init` (then
`rustup-init`), and use a Homebrew `python@3.12`. Eigen lives at
`$(brew --prefix eigen)/include/eigen3` — substitute that for `/usr/include/eigen3` in §3, and
build the extension with `clang++ … -undefined dynamic_lookup` per `SETUP_AND_RUN.md` §1b.

## 2. Clone

```bash
git clone <repo-url> dfg_da && cd dfg_da   # this is $REPO
git checkout pmbm-python-impl
```

## 3. Python environment + C++ extension

```bash
cd $REPO
python3.12 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt          # numpy/scipy/... + pyehm fork (built from git)

# Build the py_dfg_da extension into the venv (EXT_SUFFIX is mandatory):
cd pmbm-cm-python
SUFFIX="$(../.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
SITE="$(../.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
g++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(../.venv/bin/python -m pybind11 --includes) \
    -I../include -I/usr/include/eigen3 \
    module_nogtsam.cpp ../src/hypothesis.cpp ../src/lbp.cpp \
    -o "$SITE/py_dfg_da$SUFFIX"
cd $REPO
```
This is the GTSAM-free build — it needs only Eigen + the stdlib, and avoids GTSAM and the
`pybindings/pybind11` submodule entirely.

## 4. Rust workspace + the `dfg_da_py` Python module

```bash
cd $REPO
cargo build --workspace          # builds dfg_da_c (CMake) -> dfg-da-sys (bindgen) -> dfg-da -> dfg-da-py
cargo test -p dfg-da             # C++ <-> C API <-> Rust smoke test

.venv/bin/pip install -r requirements-dev.txt      # adds maturin
cd dfg-da-py && ../.venv/bin/maturin develop && cd ..   # installs dfg_da_py into ./.venv
```
See `RUST_BINDINGS.md` for the crate layout and how to extend the C API.

> If `cargo build --workspace` fails with **"no Python 3.x interpreter found"** (the pyo3
> crate `dfg-da-py` can't find an interpreter — happens on minimal machines that only have
> `python3.12`, not a bare `python3`/`python`), point pyo3 at the venv:
> `export PYO3_PYTHON=$REPO/.venv/bin/python` and rebuild.

## 5. Verify

```bash
cd $REPO
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
import py_dfg_da, pyehm, dfg_da_py, numpy, scipy, networkx, matplotlib
from pyehm.core import EHM2
import dfg_da.stats_logger, cluster_partition
assert hasattr(py_dfg_da.lbp, 'lbp_single_cluster')
assert hasattr(EHM2, 'run_and_likelihood')
print('all imports ok')
"
```
Cross-check the Rust path is bit-identical to the C++ extension (see `RUST_BINDINGS.md`):
```bash
.venv/bin/python - <<'PY'
import numpy as np, py_dfg_da, dfg_da_py
R = np.array([[0.30,0.10,0.00,-np.inf],[0.05,0.40,-np.inf,0.00]])
h = py_dfg_da.hypothesis.Hypotheses([py_dfg_da.hypothesis.Hypothesis([1,2],0.0)])
z_cpp  = py_dfg_da.lbp.lbp_single_cluster(np.asfortranarray(R), h, 300).bethe_pseudodual_normalization_constant()
z_rust = dfg_da_py.lbp_single_cluster_bethe(R.flatten().tolist(), 2, 4, [([1,2],0.0)], 300)
assert np.isclose(z_cpp, z_rust); print("Rust<->C++ match:", z_cpp)
PY
```

## 6. Provide the existing PMBM output data

The eval reads `./data/pmbm_output_files/*.mat` (the `priorLikelihood{k}.mat` scans). These
are **not** in git. Put them there by symlink or copy:
```bash
mkdir -p $REPO/data/pmbm_output_files
# e.g. symlink from wherever the scans live:
for f in /path/to/priorLikelihood*.mat; do ln -s "$f" "$REPO/data/pmbm_output_files/$(basename "$f")"; done
ls $REPO/data/pmbm_output_files/*.mat | wc -l    # sanity: number of scans
```

## 7. Compute metrics from the existing data

```bash
cd $REPO
W=$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))
EVAL_WORKERS=$W ./.venv/bin/python -W ignore ravens_parser_parallell_multicluster_not_exact.py
```
Outputs land in `ravens_output_multicluster/metrics/` (`metrics.csv` + PNGs). Related scripts:
`ravens_parser_parallell_multicluster.py`, `accuracy_metrics_multicluster.py`,
`chapter10_plots.py` (same env, run from the repo root). A fast sanity run on 1–2 scans is the
`SETUP_AND_RUN.md` §8 `loop_func(...)` snippet.

---

## 8. Docker

The `Dockerfile` builds §1–§5 (toolchain, venv, C++ extension, Rust workspace + `dfg_da_py`)
at image-build time, then drops you into an interactive **zsh** (oh-my-zsh + powerlevel10k,
using the committed `docker/zshrc` + `docker/p10k.zsh`) as the **non-root user `odin`**
(uid/gid 1000). Nothing runs by default — you enter and run things yourself. Data is
**mounted at runtime**, not baked into the image.

```bash
cd $REPO
docker build -t dfg-da .

# Enter the container (primary use): a p10k zsh at /app, as odin.
docker run -it --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/ravens_output_multicluster:/app/ravens_output_multicluster" \
  dfg-da
# ...then inside:  python -W ignore ravens_parser_parallell_multicluster_not_exact.py
#                  (the venv is on PATH, so `python` is the project interpreter)

# Or run metrics in one shot (overrides the default shell):
docker run --rm \
  -e EVAL_WORKERS=$(nproc) \
  -v "$PWD/data:/app/data" \
  -v "$PWD/ravens_output_multicluster:/app/ravens_output_multicluster" \
  dfg-da .venv/bin/python -W ignore ravens_parser_parallell_multicluster_not_exact.py
```
The eval reads `/app/data/pmbm_output_files/*.mat` (mounted) and imports the local `dfg_da/` +
`cluster_partition/` packages (baked into the image); results are written to the mounted
`ravens_output_multicluster/`. Because the container runs as `odin` (uid 1000), those outputs
are **owned by you** on the host — no root-owned files.

Notes:
- **Fonts are host-side.** The powerlevel10k prompt uses `nerdfont-v3`; its icons render only
  if your *host terminal* uses a Nerd Font (e.g. MesloLGS NF). The container only emits the
  escape sequences.
- **Different host user?** Build with `--build-arg UID=$(id -u) --build-arg GID=$(id -g)
  --build-arg USERNAME=$(id -un)` so volume ownership matches your account.
- `docker build` needs network (apt, rustup, cargo crates, oh-my-zsh/p10k clones, and the
  `pyehm` git dependency).
