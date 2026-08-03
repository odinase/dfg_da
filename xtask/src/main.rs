//! Workspace automation. One command to build + install the `dfg_da_py` Python
//! extension into the active Python environment and regenerate its `.pyi` stub.
//!
//! Invoked as `cargo dev-py` (alias in `.cargo/config.toml`). Works identically on
//! macOS and Linux/Ubuntu; only the dynamic-loader env var for the stub step differs.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, process};

fn main() {
    // repo root is the parent of this crate's manifest dir (…/dfg_da/xtask).
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must live in a subdirectory of the repo")
        .to_path_buf();

    let python = resolve_python();
    let bindir = python
        .parent()
        .expect("python executable has no parent dir")
        .to_path_buf();

    ensure_maturin(&python, &bindir);
    let maturin = maturin_path(&bindir);

    // 1. Regenerate dfg_da_py.pyi *first*, so the subsequent `maturin develop` copies
    //    the fresh stub into site-packages (as dfg_da_py/__init__.pyi). If this ran
    //    after develop, the installed stub would always be one build stale.
    //    The stub_gen binary links libpython but embeds no rpath, so point the dynamic
    //    loader at the interpreter's lib dir for this child.
    let libdir = capture(
        Command::new(&python).args([
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')",
        ]),
        "read sysconfig LIBDIR",
    );
    let libdir = libdir.trim();

    eprintln!("==> regenerating dfg_da_py.pyi");
    let mut stub = Command::new("cargo");
    stub.current_dir(&repo_root)
        .args(["run", "--quiet", "-p", "dfg-da-py", "--bin", "stub_gen"]);
    if let Some(var) = loader_env_var() {
        if !libdir.is_empty() {
            let value = match env::var(var) {
                Ok(existing) if !existing.is_empty() => format!("{libdir}:{existing}"),
                _ => libdir.to_string(),
            };
            stub.env(var, value);
        }
    }
    run(&mut stub, "stub_gen");

    // 2. Build the extension and install it (editable) into the active env. maturin
    //    picks up the freshly generated dfg-da-py/dfg_da_py.pyi and installs it.
    eprintln!("==> maturin develop  (interpreter: {})", python.display());
    run(
        Command::new(&maturin).current_dir(&repo_root).args([
            "develop",
            "--manifest-path",
            "dfg-da-py/Cargo.toml",
        ]),
        "maturin develop",
    );

    eprintln!(
        "\n✓ built + installed dfg_da_py into {} and refreshed {}",
        bindir.display(),
        repo_root.join("dfg-da-py/dfg_da_py.pyi").display()
    );
}

/// Pick the interpreter of the active environment, then canonicalize it to an
/// absolute path (so we know its bin dir). Prefers an explicit venv, then conda,
/// then `python3` on PATH.
fn resolve_python() -> PathBuf {
    let candidate = if let Some(venv) = non_empty_env("VIRTUAL_ENV") {
        PathBuf::from(venv).join("bin").join("python")
    } else if let Some(conda) = non_empty_env("CONDA_PREFIX") {
        PathBuf::from(conda).join("bin").join("python")
    } else {
        PathBuf::from("python3")
    };
    let exe = capture(
        Command::new(&candidate).args(["-c", "import sys; print(sys.executable)"]),
        &format!("locate python ({})", candidate.display()),
    );
    let exe = exe.trim();
    if exe.is_empty() {
        fail("could not determine the active Python interpreter");
    }
    PathBuf::from(exe)
}

fn maturin_path(bindir: &Path) -> PathBuf {
    let name = if cfg!(windows) {
        "maturin.exe"
    } else {
        "maturin"
    };
    bindir.join(name)
}

/// Install maturin into the active env if its console script isn't present.
fn ensure_maturin(python: &Path, bindir: &Path) {
    if maturin_path(bindir).exists() {
        return;
    }
    eprintln!(
        "==> maturin not found in {}; installing via pip",
        bindir.display()
    );
    run(
        Command::new(python).args(["-m", "pip", "install", "maturin"]),
        "pip install maturin",
    );
    if !maturin_path(bindir).exists() {
        fail(&format!(
            "pip install maturin succeeded but no maturin at {}. Is this a virtualenv/conda env?",
            bindir.display()
        ));
    }
}

fn loader_env_var() -> Option<&'static str> {
    if cfg!(target_os = "macos") {
        Some("DYLD_FALLBACK_LIBRARY_PATH")
    } else if cfg!(target_os = "linux") {
        Some("LD_LIBRARY_PATH")
    } else {
        None
    }
}

fn non_empty_env(key: &str) -> Option<String> {
    env::var(key).ok().filter(|v| !v.is_empty())
}

/// Run a command with inherited stdio; abort with a clear message on failure.
fn run(cmd: &mut Command, what: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| fail(&format!("failed to spawn {what}: {e}")));
    if !status.success() {
        fail(&format!("{what} failed ({status})"));
    }
}

/// Run a command and capture stdout; abort on failure.
fn capture(cmd: &mut Command, what: &str) -> String {
    let out = cmd
        .output()
        .unwrap_or_else(|e| fail(&format!("failed to spawn {what}: {e}")));
    if !out.status.success() {
        eprint!("{}", String::from_utf8_lossy(&out.stderr));
        fail(&format!("{what} failed ({})", out.status));
    }
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn fail(msg: &str) -> ! {
    eprintln!("xtask: error: {msg}");
    process::exit(1);
}
