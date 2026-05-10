//! libloading-backed OpenSlide FFI shim for parity tests.

#![allow(non_camel_case_types)]
#![cfg(feature = "parity-openslide")]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use libloading::{Library, Symbol};

#[repr(C)]
pub struct openslide_t {
    _private: [u8; 0],
}

type Open = unsafe extern "C" fn(filename: *const c_char) -> *mut openslide_t;
type Close = unsafe extern "C" fn(osr: *mut openslide_t);
type GetError = unsafe extern "C" fn(osr: *mut openslide_t) -> *const c_char;
type LevelCount = unsafe extern "C" fn(osr: *mut openslide_t) -> c_int;
type LevelDimensions =
    unsafe extern "C" fn(osr: *mut openslide_t, level: c_int, w: *mut i64, h: *mut i64);
type GetPropertyValue =
    unsafe extern "C" fn(osr: *mut openslide_t, name: *const c_char) -> *const c_char;
type ReadRegion = unsafe extern "C" fn(
    osr: *mut openslide_t,
    dest: *mut u32,
    x: i64,
    y: i64,
    level: c_int,
    w: i64,
    h: i64,
);

pub struct OpenSlide {
    raw: *mut openslide_t,
    api: Arc<OpenSlideApi>,
}

unsafe impl Send for OpenSlide {}
unsafe impl Sync for OpenSlide {}

struct OpenSlideApi {
    _lib: Library,
    open: Open,
    close: Close,
    get_error: GetError,
    get_property_value: GetPropertyValue,
    level_count: LevelCount,
    level_dimensions: LevelDimensions,
    read_region: ReadRegion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenSlideBounds {
    pub x: i64,
    pub y: i64,
    pub width: u64,
    pub height: u64,
}

pub fn parse_bounds_from_properties<F>(mut property: F) -> Option<OpenSlideBounds>
where
    F: FnMut(&str) -> Option<String>,
{
    let x = property("openslide.bounds-x")?.parse::<i64>().ok()?;
    let y = property("openslide.bounds-y")?.parse::<i64>().ok()?;
    let width = property("openslide.bounds-width")?.parse::<u64>().ok()?;
    let height = property("openslide.bounds-height")?.parse::<u64>().ok()?;
    (width > 0 && height > 0).then_some(OpenSlideBounds {
        x,
        y,
        width,
        height,
    })
}

fn discover_lib_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = std::env::var_os("OPENSLIDE_LIB_PATH") {
        out.push(PathBuf::from(p));
    }
    if let Some(p) = std::env::var_os("STATUMEN_OPENSLIDE_LIBRARY") {
        out.push(PathBuf::from(p));
    }
    #[cfg(target_os = "macos")]
    {
        out.push(PathBuf::from("/opt/homebrew/lib/libopenslide.dylib"));
        out.push(PathBuf::from("/opt/homebrew/lib/libopenslide.1.dylib"));
        out.push(PathBuf::from("/usr/local/lib/libopenslide.dylib"));
        out.push(PathBuf::from("/usr/local/lib/libopenslide.1.dylib"));
        out.push(PathBuf::from("libopenslide.dylib"));
        out.push(PathBuf::from("libopenslide.1.dylib"));
    }
    #[cfg(target_os = "linux")]
    {
        out.push(PathBuf::from("/usr/lib/x86_64-linux-gnu/libopenslide.so.0"));
        out.push(PathBuf::from("/usr/lib/x86_64-linux-gnu/libopenslide.so"));
        out.push(PathBuf::from("/usr/lib/libopenslide.so.0"));
        out.push(PathBuf::from("/usr/lib/libopenslide.so"));
        out.push(PathBuf::from("libopenslide.so.0"));
        out.push(PathBuf::from("libopenslide.so"));
    }
    #[cfg(target_os = "windows")]
    {
        out.push(PathBuf::from(
            r"C:\Program Files\OpenSlide\bin\libopenslide.dll",
        ));
    }
    out
}

fn try_load_api() -> Option<Arc<OpenSlideApi>> {
    for path in discover_lib_paths() {
        let lib = match unsafe { Library::new(&path) } {
            Ok(lib) => lib,
            Err(_) => continue,
        };
        let symbols = unsafe {
            let open: Symbol<Open> = lib.get(b"openslide_open\0").ok()?;
            let close: Symbol<Close> = lib.get(b"openslide_close\0").ok()?;
            let get_error: Symbol<GetError> = lib.get(b"openslide_get_error\0").ok()?;
            let get_property_value: Symbol<GetPropertyValue> =
                lib.get(b"openslide_get_property_value\0").ok()?;
            let level_count: Symbol<LevelCount> = lib.get(b"openslide_get_level_count\0").ok()?;
            let level_dimensions: Symbol<LevelDimensions> =
                lib.get(b"openslide_get_level_dimensions\0").ok()?;
            let read_region: Symbol<ReadRegion> = lib.get(b"openslide_read_region\0").ok()?;
            (
                *open,
                *close,
                *get_error,
                *get_property_value,
                *level_count,
                *level_dimensions,
                *read_region,
            )
        };
        let (
            open,
            close,
            get_error,
            get_property_value,
            level_count,
            level_dimensions,
            read_region,
        ) = symbols;
        return Some(Arc::new(OpenSlideApi {
            _lib: lib,
            open,
            close,
            get_error,
            get_property_value,
            level_count,
            level_dimensions,
            read_region,
        }));
    }
    None
}

pub fn try_load() -> Option<LoadedOpenSlide> {
    try_load_api().map(LoadedOpenSlide)
}

#[derive(Clone)]
pub struct LoadedOpenSlide(Arc<OpenSlideApi>);

impl LoadedOpenSlide {
    pub fn open(&self, path: &Path) -> Result<OpenSlide, String> {
        let cpath = CString::new(path.to_str().ok_or("path is not valid UTF-8")?.as_bytes())
            .map_err(|e| e.to_string())?;
        let raw = unsafe { (self.0.open)(cpath.as_ptr()) };
        if raw.is_null() {
            return Err("openslide_open returned NULL".into());
        }
        let err = unsafe { (self.0.get_error)(raw) };
        if !err.is_null() {
            let msg = unsafe { CStr::from_ptr(err) }
                .to_string_lossy()
                .into_owned();
            unsafe { (self.0.close)(raw) };
            return Err(format!("openslide error: {msg}"));
        }
        Ok(OpenSlide {
            raw,
            api: Arc::clone(&self.0),
        })
    }
}

impl OpenSlide {
    pub fn level_count(&self) -> u32 {
        let n = unsafe { (self.api.level_count)(self.raw) };
        n.max(0) as u32
    }

    pub fn level_dimensions(&self, level: u32) -> (u64, u64) {
        let mut width = 0i64;
        let mut height = 0i64;
        unsafe {
            (self.api.level_dimensions)(self.raw, level as c_int, &mut width, &mut height);
        }
        (width.max(0) as u64, height.max(0) as u64)
    }

    pub fn property(&self, name: &str) -> Option<String> {
        let cname = CString::new(name).ok()?;
        let value = unsafe { (self.api.get_property_value)(self.raw, cname.as_ptr()) };
        if value.is_null() {
            return None;
        }
        Some(
            unsafe { CStr::from_ptr(value) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    pub fn bounds(&self) -> Option<OpenSlideBounds> {
        parse_bounds_from_properties(|name| self.property(name))
    }

    pub fn read_region(
        &self,
        x: i64,
        y: i64,
        level: u32,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, String> {
        let mut argb = vec![0u32; width as usize * height as usize];
        unsafe {
            (self.api.read_region)(
                self.raw,
                argb.as_mut_ptr(),
                x,
                y,
                level as c_int,
                i64::from(width),
                i64::from(height),
            );
        }
        let err = unsafe { (self.api.get_error)(self.raw) };
        if !err.is_null() {
            return Err(unsafe { CStr::from_ptr(err) }
                .to_string_lossy()
                .into_owned());
        }

        let mut rgba = Vec::with_capacity(argb.len() * 4);
        for pixel in argb {
            let a = ((pixel >> 24) & 0xff) as u8;
            let r = ((pixel >> 16) & 0xff) as u8;
            let g = ((pixel >> 8) & 0xff) as u8;
            let b = (pixel & 0xff) as u8;
            if a == 0 {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }
            let unpremultiply = |channel: u8| -> u8 {
                ((u16::from(channel) * 255 + u16::from(a) / 2) / u16::from(a)).min(255) as u8
            };
            rgba.push(unpremultiply(r));
            rgba.push(unpremultiply(g));
            rgba.push(unpremultiply(b));
            rgba.push(a);
        }
        Ok(rgba)
    }
}

impl Drop for OpenSlide {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { (self.api.close)(self.raw) };
        }
    }
}
