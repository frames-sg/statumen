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
    let mut manifest = parse_manifest(&text)?;
    apply_alias_filter(
        &mut manifest,
        std::env::var("STATUMEN_PARITY_ALIASES").ok().as_deref(),
    );
    Ok(manifest)
}

pub fn load_private() -> Result<Option<CorpusManifest>, String> {
    let path = if let Some(p) = std::env::var_os("STATUMEN_PARITY_PRIVATE_MANIFEST") {
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
    if let Some(p) = std::env::var_os("STATUMEN_PARITY_CORPUS_CACHE") {
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

pub(crate) fn format_default_extension(format: &str) -> Option<&'static str> {
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

pub(crate) fn apply_alias_filter(manifest: &mut CorpusManifest, raw_aliases: Option<&str>) {
    let Some(raw_aliases) = raw_aliases else {
        return;
    };
    let aliases = raw_aliases
        .split(|ch: char| ch == ',' || ch == ';' || ch.is_whitespace())
        .filter(|alias| !alias.is_empty())
        .collect::<std::collections::HashSet<_>>();
    if aliases.is_empty() {
        return;
    }
    manifest
        .slides
        .retain(|entry| aliases.contains(entry.alias.as_str()));
}
