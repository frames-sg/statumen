//! Build script for statumen.
//!
//! Only emits link instructions for libopenslide when the
//! `openslide-bench` feature is enabled. This keeps default builds
//! free of any system dependency.

fn main() {
    if std::env::var("CARGO_FEATURE_OPENSLIDE_BENCH").is_ok() {
        match pkg_config::Config::new()
            .atleast_version("3.4")
            .probe("openslide")
        {
            Ok(_lib) => {
                // pkg_config emits cargo:rustc-link-* directives automatically.
                println!("cargo:rerun-if-changed=build.rs");
            }
            Err(e) => {
                panic!(
                    "feature `openslide-bench` requires libopenslide on the system; \
                     pkg-config could not find it: {e}\n\
                     On macOS: brew install openslide"
                );
            }
        }
    } else {
        println!("cargo:rerun-if-changed=build.rs");
    }
}
