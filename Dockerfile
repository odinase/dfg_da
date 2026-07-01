# Interactive dev container for the dfg_da stack. It BUILDS everything at image
# build time (venv + all deps incl. the pyehm fork, the py_dfg_da C++ extension,
# the Rust workspace + the dfg_da_py pyo3 module), then drops you into a zsh
# (oh-my-zsh + powerlevel10k) shell as the non-root user `odin`. Nothing runs by
# default -- enter and run things yourself.
#
#   docker build -t dfg-da .
#   docker run -it --rm -v "$PWD/data:/app/data" \
#       -v "$PWD/ravens_output_multicluster:/app/ravens_output_multicluster" dfg-da
#
# PMBM data regeneration is out of scope; mount existing output data at runtime
# to compute metrics (see BUILD_FROM_SCRATCH.md).
FROM ubuntu:24.04

# Match these to the host user so files written to mounted volumes are owned by you.
ARG USERNAME=odin
ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive

# Toolchain + interactive-shell tools. NOTE: clang/libclang-dev is required by
# Rust `bindgen`; without it the dfg-da-sys build fails to parse the C header.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ cmake ninja-build libeigen3-dev \
        python3.12 python3.12-venv python3.12-dev \
        clang libclang-dev patchelf \
        git curl ca-certificates zsh sudo \
    && rm -rf /var/lib/apt/lists/*

# Rust installed SYSTEM-WIDE (not /root) so both the root build below and the
# runtime user `odin` can use cargo/maturin. Official rust-image pattern.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
        --default-toolchain 1.96.0 --no-modify-path \
 && chmod -R a+rwX "$RUSTUP_HOME" "$CARGO_HOME"

# Point pyo3 at the venv interpreter -- this minimal image has no bare
# `python3`/`python`, so `cargo build --workspace` would otherwise fail with
# "no Python 3.x interpreter found".
ENV PYO3_PYTHON=/app/.venv/bin/python

WORKDIR /app
COPY . /app

# --- Full build (as root) ---------------------------------------------------
# Python env + dependencies (+ maturin for the Rust->Python module).
RUN python3.12 -m venv .venv \
 && .venv/bin/pip install --no-cache-dir -U pip \
 && .venv/bin/pip install --no-cache-dir -r requirements.txt maturin

# py_dfg_da C++ extension into the venv's site-packages (EXT_SUFFIX is mandatory).
RUN cd pmbm-cm-python \
 && SUFFIX="$(../.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')" \
 && SITE="$(../.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')" \
 && g++ -O3 -Wall -shared -std=c++17 -fPIC \
        $(../.venv/bin/python -m pybind11 --includes) \
        -I../include -I/usr/include/eigen3 \
        module_nogtsam.cpp ../src/hypothesis.cpp ../src/lbp.cpp \
        -o "$SITE/py_dfg_da$SUFFIX"

# Rust workspace + cross-path test + install dfg_da_py into the venv.
RUN cargo build --workspace --release \
 && cargo test -p dfg-da \
 && cd dfg-da-py && ../.venv/bin/maturin develop --release

# --- Create the desktop user ------------------------------------------------
# ubuntu:24.04 ships a default `ubuntu` user at uid 1000 -- remove it first so we
# can reuse uid/gid 1000. Give `odin` passwordless sudo and ownership of /app so
# it can re-run cargo/maturin/the eval and write to mounts as uid 1000.
RUN userdel -r ubuntu 2>/dev/null || true; \
    groupadd -g "$GID" "$USERNAME" 2>/dev/null || true; \
    useradd -m -u "$UID" -g "$GID" -s /usr/bin/zsh "$USERNAME"; \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/"$USERNAME"; \
    chmod 0440 /etc/sudoers.d/"$USERNAME"; \
    chown -R "$UID":"$GID" /app

USER $USERNAME
ENV HOME=/home/$USERNAME

# --- oh-my-zsh + powerlevel10k + the vendored desktop config (as `odin`) -----
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
        "" --unattended --keep-zshrc \
 && git clone --depth=1 https://github.com/romkatv/powerlevel10k.git \
        "$HOME/.oh-my-zsh/custom/themes/powerlevel10k"
COPY --chown=$UID:$GID docker/zshrc    $HOME/.zshrc
COPY --chown=$UID:$GID docker/p10k.zsh $HOME/.p10k.zsh
# Put the venv on PATH inside the container (no stdout -> p10k instant-prompt safe).
RUN printf '\n# --- container extras ---\nexport PATH="/app/.venv/bin:$PATH"\n' >> "$HOME/.zshrc"

# No auto-run: just drop into the interactive shell at /app.
WORKDIR /app
CMD ["zsh"]
