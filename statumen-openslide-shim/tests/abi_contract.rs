use std::ffi::{CStr, CString};
use std::ptr;

use statumen_openslide_shim::*;

fn fixture_path() -> CString {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("jp2k")
        .join("rgb_nomct.j2k");
    CString::new(path.to_string_lossy().as_bytes()).expect("fixture path has no NUL")
}

unsafe fn c_string(ptr: *const std::os::raw::c_char) -> String {
    CStr::from_ptr(ptr).to_string_lossy().into_owned()
}

#[test]
fn null_inputs_are_safe_error_values() {
    unsafe {
        assert!(openslide_detect_vendor(ptr::null()).is_null());
        assert!(openslide_open(ptr::null()).is_null());
        assert!(openslide_get_error(ptr::null_mut()).is_null());
        assert_eq!(openslide_get_level_count(ptr::null_mut()), -1);
        assert_eq!(openslide_get_level_downsample(ptr::null_mut(), 0), -1.0);
        assert_eq!(
            openslide_get_best_level_for_downsample(ptr::null_mut(), 1.0),
            -1
        );
        assert_eq!(openslide_get_icc_profile_size(ptr::null_mut()), -1);

        let mut w = 123;
        let mut h = 456;
        openslide_get_level0_dimensions(ptr::null_mut(), &mut w, &mut h);
        assert_eq!((w, h), (-1, -1));

        let names = openslide_get_property_names(ptr::null_mut());
        assert!(!names.is_null());
        assert!((*names).is_null());

        let associated = openslide_get_associated_image_names(ptr::null_mut());
        assert!(!associated.is_null());
        assert!((*associated).is_null());

        let mut dest = [0xdead_beefu32; 4];
        openslide_read_region(ptr::null_mut(), dest.as_mut_ptr(), 0, 0, 0, 2, 2);
        assert_eq!(dest, [0; 4]);
    }
}

#[test]
fn unsupported_file_returns_null_without_error_handle() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("not-a-slide.txt");
    std::fs::write(&path, b"not a slide").expect("write fixture");
    let cpath = CString::new(path.to_string_lossy().as_bytes()).expect("path has no NUL");

    unsafe {
        assert!(openslide_detect_vendor(cpath.as_ptr()).is_null());
        assert!(openslide_open(cpath.as_ptr()).is_null());
    }
}

#[test]
fn opens_supported_slide_and_exposes_core_metadata() {
    let path = fixture_path();

    unsafe {
        let vendor = openslide_detect_vendor(path.as_ptr());
        assert!(!vendor.is_null());
        assert_eq!(c_string(vendor), "raw-jp2k");

        let osr = openslide_open(path.as_ptr());
        assert!(!osr.is_null());
        assert!(openslide_get_error(osr).is_null());
        assert!(c_string(openslide_get_version()).starts_with("OpenSlide-statumen"));

        assert_eq!(openslide_get_level_count(osr), 1);
        let mut w = 0;
        let mut h = 0;
        openslide_get_level0_dimensions(osr, &mut w, &mut h);
        assert_eq!((w, h), (16, 12));
        assert_eq!(openslide_get_level_downsample(osr, 0), 1.0);
        assert_eq!(openslide_get_best_level_for_downsample(osr, 4.0), 0);

        let level_count_key = CString::new("openslide.level-count").unwrap();
        let vendor_key = CString::new("openslide.vendor").unwrap();
        assert_eq!(
            c_string(openslide_get_property_value(osr, level_count_key.as_ptr())),
            "1"
        );
        assert_eq!(
            c_string(openslide_get_property_value(osr, vendor_key.as_ptr())),
            "raw-jp2k"
        );

        let names = openslide_get_property_names(osr);
        assert!(!names.is_null());
        let mut saw_vendor = false;
        let mut idx = 0;
        while !(*names.add(idx)).is_null() {
            saw_vendor |= c_string(*names.add(idx)) == "openslide.vendor";
            idx += 1;
        }
        assert!(saw_vendor);

        let mut argb = vec![0u32; 16];
        openslide_read_region(osr, argb.as_mut_ptr(), 0, 0, 0, 4, 4);
        assert!(openslide_get_error(osr).is_null());
        assert!(argb.iter().any(|pixel| *pixel != 0));
        assert!(argb.iter().all(|pixel| (pixel >> 24) == 0xff));

        openslide_close(osr);
    }
}

#[test]
fn read_errors_zero_dest_and_make_handle_terminal() {
    let path = fixture_path();

    unsafe {
        let osr = openslide_open(path.as_ptr());
        assert!(!osr.is_null());

        let mut argb = [0xdead_beefu32; 4];
        openslide_read_region(osr, argb.as_mut_ptr(), 0, 0, 99, 2, 2);
        assert_eq!(argb, [0; 4]);

        let err = openslide_get_error(osr);
        assert!(!err.is_null());
        assert!(c_string(err).contains("level"));
        assert_eq!(openslide_get_level_count(osr), -1);

        let mut w = 123;
        let mut h = 456;
        openslide_get_level_dimensions(osr, 0, &mut w, &mut h);
        assert_eq!((w, h), (-1, -1));

        openslide_close(osr);
    }
}

#[test]
fn associated_icc_and_cache_apis_have_v4_safe_defaults() {
    let path = fixture_path();

    unsafe {
        let osr = openslide_open(path.as_ptr());
        assert!(!osr.is_null());

        let associated = openslide_get_associated_image_names(osr);
        assert!(!associated.is_null());
        assert!((*associated).is_null());

        assert_eq!(openslide_get_icc_profile_size(osr), 0);

        let cache = openslide_cache_create(1024);
        assert!(!cache.is_null());
        openslide_set_cache(osr, cache);
        openslide_cache_release(cache);

        openslide_close(osr);
    }
}
