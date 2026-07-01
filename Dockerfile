# Reproducible build of the whole dfg_da stack on a fresh machine:
#   - Python venv + all deps (incl. the pyehm fork) from requirements.txt
#   - the py_dfg_da C++ extension (GTSAM-free) built into the venv
#   - the Rust workspace (C API -> bindgen -> safe -> pyo3) + the dfg_da_py module
#
# PMBM data regeneration is intentionally out of scope; mount existing output
# data at runtime to compute metrics (see BUILD_FROM_SCRATCH.md).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Toolchain. NOTE: clang/libclang-dev is required by Rust `bindgen`; without it
# the dfg-da-sys build fails to parse the C header.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ cmake ninja-build libeigen3-dev \
        python3.12 python3.12-venv python3.12-dev \
        clang libclang-dev patchelf \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Rust (pinned to match the host toolchain).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
        --default-toolchain 1.96.0
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY . /app

# Python env + dependencies (+ maturin for the Rust->Python module).
RUN python3.12 -m venv .venv \
 && .venv/bin/pip install --no-cache-dir -U pip \
 && .venv/bin/pip install --no-cache-dir -r requirements.txt maturin

# Point pyo3 at the venv interpreter. This minimal image has no `python3`/`python`
# on PATH (only `python3.12`), so `cargo build --workspace` (which compiles the
# pyo3 crate dfg-da-py) would otherwise fail with "no Python 3.x interpreter found".
ENV PYO3_PYTHON=/app/.venv/bin/python

# Build the py_dfg_da C++ extension into the venv's site-packages.
# The EXT_SUFFIX is mandatory (a suffix-less file is unimportable).
RUN cd pmbm-cm-python \
 && SUFFIX="$(../.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')" \
 && SITE="$(../.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')" \
 && g++ -O3 -Wall -shared -std=c++17 -fPIC \
        $(../.venv/bin/python -m pybind11 --includes) \
        -I../include -I/usr/include/eigen3 \
        module_nogtsam.cpp ../src/hypothesis.cpp ../src/lbp.cpp \
        -o "$SITE/py_dfg_da$SUFFIX"

# Build the Rust workspace, run its cross-path test, and install dfg_da_py.
RUN cargo build --workspace --release \
 && cargo test -p dfg-da \
 && cd dfg-da-py && ../.venv/bin/maturin develop --release

# Default command: prove the whole stack imports.
CMD [".venv/bin/python", "-c", "import sys; sys.path.insert(0, '.'); import py_dfg_da, pyehm, dfg_da_py, dfg_da.stats_logger, cluster_partition; from pyehm.core import EHM2; assert hasattr(EHM2, 'run_and_likelihood'); print('build OK')"]
