use std::env;
use std::path::PathBuf;
use std::process::Command;

// Build the dfg_da C API (CMake) and generate raw FFI with bindgen.
// Modeled on ~/kalib-rs/kalib-sys/build.rs, reduced to a single static library.
fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir.parent().unwrap().to_path_buf();
    let c_api_dir = repo_root.join("dfg_da_c");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cmake_build = out_dir.join("cmake-build");

    // --- Configure + build the static C API library ---
    let status = Command::new("cmake")
        .args([
            "-S",
            c_api_dir.to_str().unwrap(),
            "-B",
            cmake_build.to_str().unwrap(),
            "-DCMAKE_BUILD_TYPE=Release",
        ])
        .status()
        .expect("failed to spawn cmake configure");
    assert!(status.success(), "cmake configure failed");

    let status = Command::new("cmake")
        .args([
            "--build",
            cmake_build.to_str().unwrap(),
            "--target",
            "dfg_da_c",
            "--parallel",
        ])
        .status()
        .expect("failed to spawn cmake build");
    assert!(status.success(), "cmake build failed");

    // --- Link: the static C API + the C++ runtime it needs ---
    println!("cargo:rustc-link-search=native={}", cmake_build.display());
    // dfg_da_c must precede pyehm_core: its ehm2.cpp references pyehm_core symbols
    // (the linker resolves left-to-right).
    println!("cargo:rustc-link-lib=static=dfg_da_c");
    println!("cargo:rustc-link-lib=static=pyehm_core");
    // macOS/clang uses libc++; Linux/gcc uses libstdc++.
    let cxx_runtime = if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        "c++"
    } else {
        "stdc++"
    };
    println!("cargo:rustc-link-lib=dylib={}", cxx_runtime);

    // --- Rebuild triggers ---
    for rel in [
        "dfg_da_c/CMakeLists.txt",
        "dfg_da_c/src/lbp_c.cpp",
        "dfg_da_c/src/ehm2.cpp",
        "dfg_da_c/include/dfg_da_c/lbp.h",
        "dfg_da_c/include/dfg_da_c/ehm2.h",
        "src/hypothesis.cpp",
        "src/lbp.cpp",
        "include/dfg_da/lbp.h",
        "include/dfg_da/hypothesis.h",
    ] {
        println!("cargo:rerun-if-changed={}", repo_root.join(rel).display());
    }

    // --- bindgen: raw FFI from the C API headers ---
    let include_dir = c_api_dir.join("include");
    let bindings = bindgen::Builder::default()
        .header(include_dir.join("dfg_da_c/lbp.h").to_str().unwrap())
        .header(include_dir.join("dfg_da_c/ehm2.h").to_str().unwrap())
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_function("dfg_.*")
        .allowlist_type("Dfg.*")
        .opaque_type("Dfg.*_s")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed");
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings.rs");
}
