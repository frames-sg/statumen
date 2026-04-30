#![allow(non_camel_case_types)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::sync::OnceLock;

#[repr(C)]
pub struct openslide_t {
    _private: [u8; 0],
}

type openslide_open_fn = unsafe extern "C" fn(filename: *const c_char) -> *mut openslide_t;
type openslide_close_fn = unsafe extern "C" fn(osr: *mut openslide_t);
type openslide_get_error_fn = unsafe extern "C" fn(osr: *mut openslide_t) -> *const c_char;
type openslide_read_region_fn = unsafe extern "C" fn(
    osr: *mut openslide_t,
    dest: *mut u32,
    x: i64,
    y: i64,
    level: i32,
    w: i64,
    h: i64,
);
type openslide_get_associated_image_names_fn =
    unsafe extern "C" fn(osr: *mut openslide_t) -> *const *const c_char;
type openslide_get_associated_image_dimensions_fn =
    unsafe extern "C" fn(osr: *mut openslide_t, name: *const c_char, w: *mut i64, h: *mut i64);

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

const RTLD_NOW: c_int = 0x2;

struct OpenSlideApi {
    _handle: usize,
    openslide_open: openslide_open_fn,
    openslide_close: openslide_close_fn,
    openslide_get_error: openslide_get_error_fn,
    openslide_read_region: openslide_read_region_fn,
    openslide_get_associated_image_names: openslide_get_associated_image_names_fn,
    openslide_get_associated_image_dimensions: openslide_get_associated_image_dimensions_fn,
}

fn dlerror_message() -> String {
    let err = unsafe { dlerror() };
    if err.is_null() {
        "unknown dlerror".to_string()
    } else {
        unsafe { CStr::from_ptr(err) }
            .to_string_lossy()
            .into_owned()
    }
}

fn load_symbol<T: Copy>(handle: *mut c_void, symbol: &[u8]) -> Result<T, String> {
    let name = CStr::from_bytes_with_nul(symbol).expect("symbol name must be NUL terminated");
    let ptr = unsafe { dlsym(handle, name.as_ptr()) };
    if ptr.is_null() {
        return Err(format!(
            "dlsym({}) failed: {}",
            name.to_string_lossy(),
            dlerror_message()
        ));
    }
    Ok(unsafe { std::mem::transmute_copy(&ptr) })
}

fn load_openslide_api() -> Result<OpenSlideApi, String> {
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("ZIGGURAT_OPENSLIDE_LIBRARY") {
        candidates.push(path.to_string_lossy().into_owned());
    }
    candidates.extend(
        [
            "/opt/homebrew/lib/libopenslide.1.dylib",
            "/opt/homebrew/lib/libopenslide.dylib",
            "/usr/local/lib/libopenslide.1.dylib",
            "/usr/local/lib/libopenslide.dylib",
            "libopenslide.1.dylib",
            "libopenslide.dylib",
            "libopenslide.so.1",
            "libopenslide.so",
        ]
        .into_iter()
        .map(str::to_owned),
    );

    let mut errors = Vec::new();
    for candidate in candidates {
        let cpath = CString::new(candidate.as_str()).map_err(|e| e.to_string())?;
        let handle = unsafe { dlopen(cpath.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            errors.push(format!("{candidate}: {}", dlerror_message()));
            continue;
        }

        return Ok(OpenSlideApi {
            _handle: handle as usize,
            openslide_open: load_symbol(handle, b"openslide_open\0")?,
            openslide_close: load_symbol(handle, b"openslide_close\0")?,
            openslide_get_error: load_symbol(handle, b"openslide_get_error\0")?,
            openslide_read_region: load_symbol(handle, b"openslide_read_region\0")?,
            openslide_get_associated_image_names: load_symbol(
                handle,
                b"openslide_get_associated_image_names\0",
            )?,
            openslide_get_associated_image_dimensions: load_symbol(
                handle,
                b"openslide_get_associated_image_dimensions\0",
            )?,
        });
    }

    Err(format!(
        "failed to load libopenslide; tried: {}",
        errors.join(" | ")
    ))
}

fn openslide_api() -> Result<&'static OpenSlideApi, String> {
    static API: OnceLock<Result<OpenSlideApi, String>> = OnceLock::new();
    API.get_or_init(load_openslide_api)
        .as_ref()
        .map_err(|e| e.clone())
}

pub struct OpenSlide {
    raw: *mut openslide_t,
    api: &'static OpenSlideApi,
}

impl OpenSlide {
    pub fn open(path: &Path) -> Result<Self, String> {
        let api = openslide_api()?;
        let cpath = CString::new(path.to_str().ok_or("path is not valid UTF-8")?.as_bytes())
            .map_err(|e| e.to_string())?;
        let raw = unsafe { (api.openslide_open)(cpath.as_ptr()) };
        if raw.is_null() {
            return Err("openslide_open returned NULL".into());
        }
        let err = unsafe { (api.openslide_get_error)(raw) };
        if !err.is_null() {
            let msg = unsafe { CStr::from_ptr(err) }
                .to_string_lossy()
                .into_owned();
            unsafe { (api.openslide_close)(raw) };
            return Err(format!("openslide error: {msg}"));
        }
        Ok(Self { raw, api })
    }

    pub fn associated_names(&self) -> Vec<String> {
        let names_ptr = unsafe { (self.api.openslide_get_associated_image_names)(self.raw) };
        if names_ptr.is_null() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut idx = 0usize;
        loop {
            let ptr = unsafe { *names_ptr.add(idx) };
            if ptr.is_null() {
                break;
            }
            out.push(
                unsafe { CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .into_owned(),
            );
            idx += 1;
        }
        out
    }

    pub fn associated_dimensions(&self, name: &str) -> Result<(u32, u32), String> {
        let cname = CString::new(name).map_err(|e| e.to_string())?;
        let mut width = 0i64;
        let mut height = 0i64;
        unsafe {
            (self.api.openslide_get_associated_image_dimensions)(
                self.raw,
                cname.as_ptr(),
                &mut width,
                &mut height,
            )
        };
        self.check_error()?;
        let width = u32::try_from(width)
            .map_err(|_| format!("associated image width out of range for {name}: {width}"))?;
        let height = u32::try_from(height)
            .map_err(|_| format!("associated image height out of range for {name}: {height}"))?;
        Ok((width, height))
    }

    pub fn read_region_rgba(
        &self,
        x: i64,
        y: i64,
        level: i32,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, String> {
        let mut argb = vec![0u32; width as usize * height as usize];
        unsafe {
            (self.api.openslide_read_region)(
                self.raw,
                argb.as_mut_ptr(),
                x,
                y,
                level,
                i64::from(width),
                i64::from(height),
            )
        };
        self.check_error()?;

        let mut rgba = Vec::with_capacity(argb.len() * 4);
        for pixel in argb {
            let a = ((pixel >> 24) & 0xFF) as u8;
            let r = ((pixel >> 16) & 0xFF) as u8;
            let g = ((pixel >> 8) & 0xFF) as u8;
            let b = (pixel & 0xFF) as u8;
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

    fn check_error(&self) -> Result<(), String> {
        let err = unsafe { (self.api.openslide_get_error)(self.raw) };
        if err.is_null() {
            Ok(())
        } else {
            Err(unsafe { CStr::from_ptr(err) }
                .to_string_lossy()
                .into_owned())
        }
    }
}

impl Drop for OpenSlide {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { (self.api.openslide_close)(self.raw) };
        }
    }
}
