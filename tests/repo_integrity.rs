use std::{
    fs,
    path::{Path, PathBuf},
};

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
        let retired_prefix = ["ZIG", "GURAT_"].concat();
        assert!(
            !source.contains(&retired_prefix),
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

#[test]
fn default_manifest_uses_cpu_jp2k_facade_and_optional_metal_adapter() {
    let manifest = fs::read_to_string(crate_root().join("Cargo.toml")).expect("read manifest");
    let manifest = manifest
        .parse::<toml::Value>()
        .expect("Cargo.toml must parse as TOML");

    let dependencies = manifest
        .get("dependencies")
        .and_then(toml::Value::as_table)
        .expect("Cargo.toml must define [dependencies]");
    assert!(
        dependencies.contains_key("signinum-j2k"),
        "statumen default JP2K decode must depend on signinum-j2k facade"
    );

    let j2k_metal = dependencies
        .get("signinum-j2k-metal")
        .and_then(toml::Value::as_table)
        .expect("signinum-j2k-metal dependency must use table syntax");
    assert!(
        j2k_metal.get("optional").and_then(toml::Value::as_bool) == Some(true),
        "signinum-j2k-metal must be optional"
    );

    let features = manifest
        .get("features")
        .and_then(toml::Value::as_table)
        .expect("Cargo.toml must define [features]");
    let metal_feature = features
        .get("metal")
        .and_then(toml::Value::as_array)
        .expect("metal feature must be an array");
    assert!(
        metal_feature
            .iter()
            .any(|value| value.as_str() == Some("dep:signinum-j2k-metal")),
        "metal feature must be the only feature that enables signinum-j2k-metal"
    );

    let enabling_features = features
        .iter()
        .filter_map(|(name, value)| {
            value.as_array().and_then(|items| {
                items
                    .iter()
                    .any(|item| item.as_str() == Some("dep:signinum-j2k-metal"))
                    .then_some(name.as_str())
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        enabling_features,
        vec!["metal"],
        "only the metal feature may enable signinum-j2k-metal"
    );
}

#[test]
fn tracked_text_files_do_not_contain_local_user_paths() {
    let mut offenders = Vec::new();
    let local_user_path = ["/Users", "user", ""].join("/");
    for path in tracked_text_files(crate_root()) {
        let source = fs::read_to_string(&path).unwrap_or_else(|err| {
            panic!("read {}: {err}", path.display());
        });
        if source.contains(&local_user_path) {
            offenders.push(relative_path(&path));
        }
    }

    assert!(
        offenders.is_empty(),
        "tracked text files must not contain local /Users/user paths:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn referenced_parity_corpus_fetch_script_exists() {
    let script = crate_root().join("scripts/parity-corpus-fetch.sh");
    assert!(
        script.is_file(),
        "tests reference scripts/parity-corpus-fetch.sh, so the script must exist"
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(&script)
            .expect("stat fetch script")
            .permissions()
            .mode();
        assert!(
            mode & 0o111 != 0,
            "scripts/parity-corpus-fetch.sh must be executable"
        );
    }
}

#[test]
fn public_docs_do_not_advertise_unregistered_zeiss_support() {
    let formats_mod =
        fs::read_to_string(crate_root().join("src/formats/mod.rs")).expect("read formats mod");
    let registry =
        fs::read_to_string(crate_root().join("src/core/registry.rs")).expect("read registry");
    let zeiss_registered = formats_mod.contains("mod zeiss") && registry.contains("ZeissBackend");

    if !zeiss_registered {
        let docs = [
            (
                "README.md",
                fs::read_to_string(crate_root().join("README.md")).expect("read README"),
            ),
            (
                "docs/architecture.md",
                fs::read_to_string(crate_root().join("docs/architecture.md")).expect("read docs"),
            ),
            (
                "architecture.md",
                fs::read_to_string(crate_root().join("architecture.md"))
                    .expect("read root architecture"),
            ),
        ];
        let offenders = docs
            .iter()
            .filter_map(|(path, text)| text.contains("Zeiss").then_some(*path))
            .collect::<Vec<_>>();
        assert!(
            offenders.is_empty(),
            "public docs must not advertise Zeiss until the backend is registered: {}",
            offenders.join(", ")
        );
    }
}

#[test]
fn unregistered_zeiss_backend_is_not_left_as_packaged_source() {
    let zeiss_source = crate_root().join("src/formats/zeiss.rs");
    if !zeiss_source.exists() {
        return;
    }

    let formats_mod =
        fs::read_to_string(crate_root().join("src/formats/mod.rs")).expect("read formats mod");
    let registry =
        fs::read_to_string(crate_root().join("src/core/registry.rs")).expect("read registry");
    assert!(
        formats_mod.contains("mod zeiss") && registry.contains("ZeissBackend"),
        "src/formats/zeiss.rs exists but the Zeiss backend is not registered"
    );
}

fn tracked_text_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_text_files(root, &mut files);
    files
}

fn collect_text_files(path: &Path, files: &mut Vec<PathBuf>) {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    if matches!(name, ".git" | "target") {
        return;
    }

    let metadata = fs::metadata(path).unwrap_or_else(|err| {
        panic!("stat {}: {err}", path.display());
    });
    if metadata.is_dir() {
        let mut entries = fs::read_dir(path)
            .unwrap_or_else(|err| panic!("read dir {}: {err}", path.display()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|err| panic!("read dir entry under {}: {err}", path.display()));
        entries.sort_by_key(|entry| entry.path());
        for entry in entries {
            collect_text_files(&entry.path(), files);
        }
        return;
    }

    if is_text_file(path) {
        files.push(path.to_path_buf());
    }
}

fn is_text_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    if matches!(name, ".gitignore" | "LICENSE") {
        return true;
    }
    matches!(
        path.extension().and_then(|value| value.to_str()),
        Some("rs" | "md" | "toml" | "yml" | "yaml" | "sh" | "py" | "txt" | "lock" | "example")
    )
}

fn relative_path(path: &Path) -> String {
    path.strip_prefix(crate_root())
        .unwrap_or(path)
        .display()
        .to_string()
}
