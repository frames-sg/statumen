use std::ffi::CStr;
use std::fs;
use std::io::{Read, Write};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformLibraryNames {
    MacOS,
    Linux,
}

impl PlatformLibraryNames {
    pub fn current() -> Result<Self, String> {
        if cfg!(target_os = "macos") {
            Ok(Self::MacOS)
        } else if cfg!(target_os = "linux") {
            Ok(Self::Linux)
        } else {
            Err("statumen OpenSlide shim install supports macOS and Linux only".into())
        }
    }

    pub fn names(self) -> [&'static str; 3] {
        match self {
            Self::MacOS => [
                "libopenslide.1.dylib",
                "libopenslide.dylib",
                "libopenslide.4.dylib",
            ],
            Self::Linux => ["libopenslide.so.1", "libopenslide.so", "libopenslide.so.4"],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallStep {
    Backup { from: PathBuf, to: PathBuf },
    CopyShim { from: PathBuf, to: PathBuf },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoreEntry {
    pub destination: PathBuf,
    pub backup: Option<PathBuf>,
}

pub fn install_destinations(prefix: &Path, platform: PlatformLibraryNames) -> Vec<PathBuf> {
    platform
        .names()
        .into_iter()
        .map(|name| prefix.join("lib").join(name))
        .collect()
}

pub fn plan_install(
    prefix: &Path,
    shim: &Path,
    platform: PlatformLibraryNames,
    stamp: u64,
    exists: impl Fn(&Path) -> bool,
) -> Vec<InstallStep> {
    let mut steps = Vec::new();
    for destination in install_destinations(prefix, platform) {
        if exists(&destination) {
            steps.push(InstallStep::Backup {
                from: destination.clone(),
                to: backup_path(&destination, stamp),
            });
        }
        steps.push(InstallStep::CopyShim {
            from: shim.to_path_buf(),
            to: destination,
        });
    }
    steps
}

pub fn execute_install(
    prefix: &Path,
    shim: &Path,
    platform: PlatformLibraryNames,
    stamp: u64,
) -> Result<PathBuf, String> {
    if !shim.is_file() {
        return Err(format!("shim library does not exist: {}", shim.display()));
    }
    let lib_dir = prefix.join("lib");
    fs::create_dir_all(&lib_dir).map_err(|err| format!("create {}: {err}", lib_dir.display()))?;

    let steps = plan_install(prefix, shim, platform, stamp, Path::exists);
    let mut restore_entries = Vec::new();
    for destination in install_destinations(prefix, platform) {
        let backup = steps.iter().find_map(|step| match step {
            InstallStep::Backup { from, to } if from == &destination => Some(to.clone()),
            _ => None,
        });
        restore_entries.push(RestoreEntry {
            destination,
            backup,
        });
    }

    for step in &steps {
        match step {
            InstallStep::Backup { from, to } => {
                if to.exists() {
                    return Err(format!("backup path already exists: {}", to.display()));
                }
                fs::rename(from, to).map_err(|err| {
                    format!("backup {} to {}: {err}", from.display(), to.display())
                })?;
            }
            InstallStep::CopyShim { from, to } => {
                fs::copy(from, to)
                    .map_err(|err| format!("copy {} to {}: {err}", from.display(), to.display()))?;
            }
        }
    }

    let manifest = manifest_path(prefix);
    write_manifest(&manifest, &restore_entries)?;
    let verify_target = install_destinations(prefix, platform)
        .into_iter()
        .next()
        .ok_or_else(|| "no install destinations planned".to_string())?;
    verify_library_version(&verify_target)?;
    Ok(manifest)
}

pub fn execute_restore(prefix: &Path, stamp: u64) -> Result<(), String> {
    let manifest = manifest_path(prefix);
    let entries = read_manifest(&manifest)?;
    for entry in entries {
        if entry.destination.exists() {
            let removed = removed_path(&entry.destination, stamp);
            if removed.exists() {
                return Err(format!(
                    "restore side path already exists: {}",
                    removed.display()
                ));
            }
            fs::rename(&entry.destination, &removed).map_err(|err| {
                format!(
                    "move installed shim {} to {}: {err}",
                    entry.destination.display(),
                    removed.display()
                )
            })?;
        }
        if let Some(backup) = entry.backup {
            if !backup.exists() {
                return Err(format!("backup is missing: {}", backup.display()));
            }
            fs::rename(&backup, &entry.destination).map_err(|err| {
                format!(
                    "restore {} to {}: {err}",
                    backup.display(),
                    entry.destination.display()
                )
            })?;
        }
    }
    Ok(())
}

pub fn manifest_path(prefix: &Path) -> PathBuf {
    prefix
        .join("lib")
        .join(".statumen-openslide-shim-install.tsv")
}

fn backup_path(destination: &Path, stamp: u64) -> PathBuf {
    PathBuf::from(format!("{}.statumen-backup-{stamp}", destination.display()))
}

fn removed_path(destination: &Path, stamp: u64) -> PathBuf {
    PathBuf::from(format!(
        "{}.statumen-removed-{stamp}",
        destination.display()
    ))
}

fn write_manifest(path: &Path, entries: &[RestoreEntry]) -> Result<(), String> {
    let mut file =
        fs::File::create(path).map_err(|err| format!("create {}: {err}", path.display()))?;
    for entry in entries {
        let backup = entry
            .backup
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_default();
        writeln!(file, "{}\t{}", entry.destination.display(), backup)
            .map_err(|err| format!("write {}: {err}", path.display()))?;
    }
    Ok(())
}

fn read_manifest(path: &Path) -> Result<Vec<RestoreEntry>, String> {
    let mut text = String::new();
    fs::File::open(path)
        .map_err(|err| format!("open {}: {err}", path.display()))?
        .read_to_string(&mut text)
        .map_err(|err| format!("read {}: {err}", path.display()))?;
    let mut entries = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let Some((destination, backup)) = line.split_once('\t') else {
            return Err(format!("manifest line {} is malformed", idx + 1));
        };
        entries.push(RestoreEntry {
            destination: PathBuf::from(destination),
            backup: (!backup.is_empty()).then(|| PathBuf::from(backup)),
        });
    }
    Ok(entries)
}

fn verify_library_version(path: &Path) -> Result<(), String> {
    // SAFETY: The loaded library is immediately queried for the documented
    // OpenSlide version symbol, and the returned pointer is checked for NULL
    // before conversion to a C string.
    unsafe {
        let library = libloading::Library::new(path)
            .map_err(|err| format!("load {}: {err}", path.display()))?;
        let get_version: libloading::Symbol<unsafe extern "C" fn() -> *const c_char> = library
            .get(b"openslide_get_version\0")
            .map_err(|err| format!("load openslide_get_version from {}: {err}", path.display()))?;
        let version = get_version();
        if version.is_null() {
            return Err("openslide_get_version returned NULL".into());
        }
        let version = CStr::from_ptr(version).to_string_lossy();
        if !version.starts_with("OpenSlide-statumen") {
            return Err(format!("unexpected OpenSlide shim version: {version}"));
        }
    }
    Ok(())
}
