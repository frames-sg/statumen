//! Parity-corpus manifest loader.

use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct CorpusEntry {
    pub name: String,
    pub alias: String,
    #[serde(default)]
    pub path: String,
    pub format: String,
    pub codecs: Vec<String>,
    #[serde(default)]
    pub must_decode: Vec<String>,
    pub source: String,
    pub license: String,
    pub redistributable: bool,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub citation: String,
    #[serde(default)]
    pub phi_reviewed: bool,
    #[serde(default)]
    pub tolerant_regions: Vec<String>,
    #[serde(default)]
    pub expected_failures: Vec<String>,
    #[serde(default)]
    pub url: String,
}

impl CorpusEntry {
    pub fn must_decode_level(&self, level: u32) -> bool {
        self.must_decode.iter().any(|item| {
            if item == "base" {
                return level == 0;
            }
            item.strip_prefix("level")
                .and_then(|n| n.parse::<u32>().ok())
                == Some(level)
        })
    }

    pub fn expected_failure(&self, pair: &str, level: u32) -> bool {
        let numbered = format!("{pair}:level{level}");
        let alias = if level == 0 {
            Some(format!("{pair}:base"))
        } else {
            None
        };
        self.expected_failures
            .iter()
            .any(|item| item == &numbered || alias.as_ref() == Some(item))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CorpusManifest {
    #[serde(rename = "slide", default)]
    pub slides: Vec<CorpusEntry>,
}

pub fn parse_manifest(toml_text: &str) -> Result<CorpusManifest, String> {
    toml::from_str::<CorpusManifest>(toml_text).map_err(|e| format!("manifest parse: {e}"))
}

pub fn load_public() -> Result<CorpusManifest, String> {
    let path = public_manifest_path();
    let text =
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    parse_manifest(&text)
}

pub fn load_private() -> Result<Option<CorpusManifest>, String> {
    let path = if let Some(p) = std::env::var_os("ZIGGURAT_PARITY_PRIVATE_MANIFEST") {
        PathBuf::from(p)
    } else {
        private_manifest_path()
    };
    if !path.is_file() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(Some(parse_manifest(&text)?))
}

pub fn corpus_cache_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("ZIGGURAT_PARITY_CORPUS_CACHE") {
        return PathBuf::from(p);
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache")
        .join("slideviewer")
        .join("parity-corpus")
}

pub fn resolve_entry_path(entry: &CorpusEntry) -> PathBuf {
    if !entry.path.is_empty() {
        let p = PathBuf::from(&entry.path);
        if p.is_file() {
            return p;
        }
        if p.is_relative() {
            let repo_relative = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join(&p);
            if repo_relative.is_file() {
                return repo_relative;
            }
        }
    }

    for candidate in cache_candidates(entry) {
        if let Some(path) = resolve_candidate(entry, &candidate) {
            return path;
        }
    }

    cache_candidates(entry)
        .into_iter()
        .next()
        .unwrap_or_else(|| corpus_cache_dir().join(&entry.alias))
}

pub fn find_slide_by_alias(alias: &str) -> Option<PathBuf> {
    if let Ok(public) = load_public() {
        for entry in &public.slides {
            if entry.alias == alias {
                let path = resolve_entry_path(entry);
                if path.is_file() {
                    return Some(path);
                }
            }
        }
    }
    if let Ok(Some(private)) = load_private() {
        for entry in &private.slides {
            if entry.alias == alias {
                let path = resolve_entry_path(entry);
                if path.is_file() {
                    return Some(path);
                }
            }
        }
    }
    None
}

fn cache_candidates(entry: &CorpusEntry) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let cache = corpus_cache_dir();
    candidates.push(cache.join(&entry.alias));

    candidates.push(cache.join(format!("{}.d", entry.alias)));

    if let Some(ext) = format_default_extension(&entry.format) {
        candidates.push(cache.join(format!("{}.{}", entry.alias, ext)));
    }

    if let Some(name) = url_file_name(&entry.url) {
        candidates.push(cache.join(name));
    }

    candidates
}

fn resolve_candidate(entry: &CorpusEntry, candidate: &Path) -> Option<PathBuf> {
    if candidate.is_file() {
        return Some(candidate.to_path_buf());
    }
    if !candidate.is_dir() {
        return None;
    }
    let wanted_ext = match entry.format.as_str() {
        "hamamatsu_vms" => "vms",
        "mirax" => "mrxs",
        "dicom" => "dcm",
        _ => return None,
    };
    find_file_with_extension(candidate, wanted_ext)
}

fn find_file_with_extension(root: &Path, ext: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|value| value.to_str())
                .is_some_and(|value| value.eq_ignore_ascii_case(ext))
        {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_file_with_extension(&path, ext) {
                return Some(found);
            }
        }
    }
    None
}

fn url_file_name(url: &str) -> Option<&str> {
    url.rsplit('/').next().filter(|name| !name.is_empty())
}

fn format_default_extension(format: &str) -> Option<&'static str> {
    match format {
        "aperio" => Some("svs"),
        "leica" => Some("scn"),
        "ventana" => Some("bif"),
        "philips_tiff" | "tiff" => Some("tif"),
        "ndpi" => Some("ndpi"),
        "hamamatsu_vms" => Some("zip"),
        "dicom" => Some("dcm"),
        "zeiss_czi" => Some("czi"),
        "mirax" => Some("zip"),
        _ => None,
    }
}

pub fn public_manifest_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("parity_corpus.public.toml")
}

pub fn private_manifest_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("parity_corpus.private.toml")
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    const SAMPLE: &str = r#"
        [[slide]]
        name             = "aperio_svs_brightfield_he_typical"
        alias            = "svs-001"
        path             = ""
        format           = "aperio"
        codecs           = ["jpeg"]
        must_decode      = ["base", "level1", "level2", "label", "macro"]
        source           = "openslide-testdata"
        license          = "CC0-1.0"
        redistributable  = true
        sha256           = "deadbeef"
        citation         = "Goode A. et al. OpenSlide..."
        phi_reviewed     = true
        url              = "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs"
    "#;

    #[test]
    fn parses_minimal_manifest() {
        let m = parse_manifest(SAMPLE).expect("parse");
        assert_eq!(m.slides.len(), 1);
        let s = &m.slides[0];
        assert_eq!(s.alias, "svs-001");
        assert_eq!(s.format, "aperio");
        assert!(s.redistributable);
        assert_eq!(s.codecs, vec!["jpeg"]);
        assert_eq!(s.must_decode.len(), 5);
    }

    #[test]
    fn unknown_format_extension_returns_none() {
        assert!(format_default_extension("nonsense").is_none());
        assert_eq!(format_default_extension("aperio"), Some("svs"));
    }

    #[test]
    fn cache_dir_respects_env() {
        let prev = std::env::var_os("ZIGGURAT_PARITY_CORPUS_CACHE");
        std::env::set_var("ZIGGURAT_PARITY_CORPUS_CACHE", "/tmp/sv-corpus-test");
        let p = corpus_cache_dir();
        assert_eq!(p, PathBuf::from("/tmp/sv-corpus-test"));
        if let Some(v) = prev {
            std::env::set_var("ZIGGURAT_PARITY_CORPUS_CACHE", v);
        } else {
            std::env::remove_var("ZIGGURAT_PARITY_CORPUS_CACHE");
        }
    }

    #[test]
    fn public_manifest_parses() {
        let p = public_manifest_path();
        let text = std::fs::read_to_string(&p).expect("read public manifest");
        let m = parse_manifest(&text).expect("parse public manifest");
        assert!(!m.slides.is_empty(), "public manifest has no slides");
        for s in &m.slides {
            assert!(
                s.redistributable,
                "public entry {} not redistributable",
                s.alias
            );
            assert!(!s.alias.is_empty());
            assert!(!s.format.is_empty());
            assert!(!s.codecs.is_empty());
        }
    }

    #[test]
    fn must_decode_level_matches_base_and_numbered_levels() {
        let mut manifest = parse_manifest(SAMPLE).expect("parse");
        let entry = manifest.slides.first_mut().expect("slide");
        entry.must_decode = vec!["base".into(), "level1".into(), "level12".into()];

        assert!(entry.must_decode_level(0));
        assert!(entry.must_decode_level(1));
        assert!(entry.must_decode_level(12));
        assert!(!entry.must_decode_level(2));
        assert!(!entry.must_decode_level(10));
    }

    #[test]
    fn expected_failure_matches_pair_and_level_aliases() {
        let mut manifest = parse_manifest(SAMPLE).expect("parse");
        let entry = manifest.slides.first_mut().expect("slide");
        entry.expected_failures = vec![
            "ashlar-vs-reference:base".into(),
            "reference-vs-openslide:level2".into(),
        ];

        assert!(entry.expected_failure("ashlar-vs-reference", 0));
        assert!(entry.expected_failure("reference-vs-openslide", 2));
        assert!(!entry.expected_failure("ashlar-vs-reference", 1));
        assert!(!entry.expected_failure("ashlar-vs-openslide", 0));
    }
}
