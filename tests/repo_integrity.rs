use std::{fs, path::Path};

fn crate_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn package_metadata_uses_statumen_identity() {
    let manifest = fs::read_to_string(crate_root().join("Cargo.toml")).expect("read manifest");

    assert!(
        manifest.contains("name = \"statumen\""),
        "crate package name must be statumen"
    );
    assert!(
        manifest.contains("description = \"Statumen whole-slide image reader\""),
        "crate description must use the Statumen name"
    );
    assert!(
        manifest.contains("repository = \"https://github.com/jcwal1516/statumen\""),
        "crate repository metadata must point at the renamed Statumen repo"
    );
}

#[test]
fn public_docs_use_statumen_entrypoint() {
    let readme = fs::read_to_string(crate_root().join("README.md")).expect("read README");
    let architecture =
        fs::read_to_string(crate_root().join("docs/architecture.md")).expect("read docs");

    assert!(
        readme.contains("# statumen"),
        "README must title the crate statumen"
    );
    assert!(
        readme.contains("use statumen::"),
        "README quick start must import statumen"
    );
    assert!(
        architecture.contains("# statumen Architecture"),
        "architecture docs must title the crate statumen"
    );
}

#[test]
fn environment_knobs_use_statumen_prefix() {
    for relative in [
        "benches/read_paths.rs",
        "src/bin/release_gate.rs",
        "src/core/cache.rs",
        "src/decode/jp2k.rs",
        "src/formats/tiff_family/pixel_access.rs",
        "tests/dicom_parity.rs",
        "tests/fixtures/parity_corpus.public.toml",
        "tests/openslide_compare.rs",
        "tests/openslide_parity.rs",
        "tests/openslide_test_support.rs",
        "tests/real_wsi_behavior.rs",
        "tests/signinum_parity.rs",
        "tests/support/corpus.rs",
        "tests/support/openslide_shim.rs",
    ] {
        let source = fs::read_to_string(crate_root().join(relative)).unwrap_or_else(|err| {
            panic!("read {relative}: {err}");
        });
        assert!(
            !source.contains("ZIGGURAT_"),
            "{relative} must use STATUMEN_ environment variable names"
        );
    }
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
fn public_wsi_api_does_not_reexport_signinum_backend_request() {
    let lib = fs::read_to_string(crate_root().join("src/lib.rs")).expect("read lib");

    assert!(
        !lib.contains("pub use signinum_core::BackendRequest"),
        "statumen public output policy must use OutputBackendRequest"
    );
}
