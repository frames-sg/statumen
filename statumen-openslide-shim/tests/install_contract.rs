use std::path::Path;

use statumen_openslide_shim::install::{
    install_destinations, plan_install, InstallStep, PlatformLibraryNames,
};

#[test]
fn install_destinations_include_all_loader_compatible_names() {
    let mac = PlatformLibraryNames::MacOS;
    assert_eq!(
        mac.names(),
        [
            "libopenslide.1.dylib",
            "libopenslide.dylib",
            "libopenslide.4.dylib"
        ]
    );

    let linux = PlatformLibraryNames::Linux;
    assert_eq!(
        linux.names(),
        ["libopenslide.so.1", "libopenslide.so", "libopenslide.so.4"]
    );

    let destinations = install_destinations(Path::new("/prefix"), mac);
    assert_eq!(
        destinations[0],
        Path::new("/prefix/lib/libopenslide.1.dylib")
    );
    assert_eq!(
        destinations[2],
        Path::new("/prefix/lib/libopenslide.4.dylib")
    );
}

#[test]
fn install_plan_backs_up_existing_destinations_before_copying_shim() {
    let prefix = Path::new("/prefix");
    let shim = Path::new("/build/libstatumen_openslide_shim.dylib");
    let steps = plan_install(prefix, shim, PlatformLibraryNames::MacOS, 42, |path| {
        path.ends_with("libopenslide.1.dylib")
    });

    assert_eq!(
        steps,
        vec![
            InstallStep::Backup {
                from: Path::new("/prefix/lib/libopenslide.1.dylib").to_path_buf(),
                to: Path::new("/prefix/lib/libopenslide.1.dylib.statumen-backup-42").to_path_buf(),
            },
            InstallStep::CopyShim {
                from: shim.to_path_buf(),
                to: Path::new("/prefix/lib/libopenslide.1.dylib").to_path_buf(),
            },
            InstallStep::CopyShim {
                from: shim.to_path_buf(),
                to: Path::new("/prefix/lib/libopenslide.dylib").to_path_buf(),
            },
            InstallStep::CopyShim {
                from: shim.to_path_buf(),
                to: Path::new("/prefix/lib/libopenslide.4.dylib").to_path_buf(),
            },
        ]
    );
}
