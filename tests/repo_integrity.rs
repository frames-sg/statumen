use std::{fs, path::Path};

fn crate_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn registry_does_not_use_type_erased_region_cache_tokens() {
    let registry =
        fs::read_to_string(crate_root().join("src/core/registry.rs")).expect("read registry");

    assert!(
        !registry.contains("RegionCacheToken"),
        "region cache plumbing must use SlideReadContext, not RegionCacheToken"
    );
    assert!(
        !registry.contains("std::any::Any"),
        "SlideReader cache plumbing must not use Any/downcast"
    );
}

#[test]
fn format_registry_does_not_silently_rewrite_svcache_paths() {
    let registry =
        fs::read_to_string(crate_root().join("src/core/registry.rs")).expect("read registry");

    assert!(
        !registry.contains("resolve_svcache"),
        "FormatRegistry must not silently resolve .svcache paths"
    );
    assert!(
        !registry.contains("resolve_open_path("),
        "implicit .svcache resolution belongs behind SlideOpenOptions"
    );
}

#[test]
fn public_wsi_api_does_not_reexport_ashlar_backend_request() {
    let lib = fs::read_to_string(crate_root().join("src/lib.rs")).expect("read lib");

    assert!(
        !lib.contains("pub use ashlar_core::BackendRequest"),
        "ziggurat public output policy must use OutputBackendRequest"
    );
}
