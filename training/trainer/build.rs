// Link libtorch's CUDA backend from the wheel pointed to by LIBTORCH (set in
// training/.cargo/config.toml), without calling Python.
//
// --no-as-needed: a binary that references no CUDA symbol directly would
// otherwise drop torch_cuda; its static initializers then never run and
// tch::Cuda::is_available() reports false (same fix as fox-lite's build.rs).
//
// The rpath lets the binary find libtorch at runtime without LD_LIBRARY_PATH;
// the torch libs locate the nvidia/* CUDA libs through their own RUNPATH.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    let libtorch = std::env::var("LIBTORCH")
        .expect("LIBTORCH must point at the wheel's torch/ dir (see training/.cargo/config.toml)");
    let lib = std::path::Path::new(&libtorch).join("lib");
    let lib = lib.canonicalize().unwrap_or(lib);
    let lib = lib.display();
    println!("cargo:rustc-link-search=native={lib}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch_cuda");
    println!("cargo:rustc-link-arg=-lc10_cuda");
}
