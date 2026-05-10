use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use statumen_openslide_shim::install::{execute_install, execute_restore, PlatformLibraryNames};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let command = args
        .next()
        .ok_or_else(|| usage("missing command: install or restore"))?;
    let mut prefix = default_prefix();
    let mut shim = None::<PathBuf>;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--prefix" => {
                prefix = PathBuf::from(
                    args.next()
                        .ok_or_else(|| usage("--prefix requires a path"))?,
                );
            }
            "--shim" => {
                shim = Some(PathBuf::from(
                    args.next().ok_or_else(|| usage("--shim requires a path"))?,
                ));
            }
            "-h" | "--help" => return Err(usage("")),
            other => return Err(usage(&format!("unknown argument: {other}"))),
        }
    }

    let platform = PlatformLibraryNames::current()?;
    let stamp = current_stamp()?;
    match command.as_str() {
        "install" => {
            let shim = shim.ok_or_else(|| usage("install requires --shim <path>"))?;
            let manifest = execute_install(&prefix, &shim, platform, stamp)?;
            println!(
                "installed statumen OpenSlide shim; restore manifest: {}",
                manifest.display()
            );
        }
        "restore" => {
            execute_restore(&prefix, stamp)?;
            println!("restored OpenSlide libraries from statumen shim manifest");
        }
        other => return Err(usage(&format!("unknown command: {other}"))),
    }
    Ok(())
}

fn usage(message: &str) -> String {
    let prefix = if message.is_empty() {
        String::new()
    } else {
        format!("{message}\n\n")
    };
    format!(
        "{prefix}usage:\n  statumen-openslide-install install --shim <path> [--prefix <prefix>]\n  statumen-openslide-install restore [--prefix <prefix>]"
    )
}

fn default_prefix() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        PathBuf::from("/opt/homebrew")
    }
    #[cfg(target_os = "linux")]
    {
        PathBuf::from("/usr/local")
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        PathBuf::from("/usr/local")
    }
}

fn current_stamp() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|err| format!("system clock before Unix epoch: {err}"))
}
