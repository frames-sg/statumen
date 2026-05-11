#![deny(unsafe_op_in_unsafe_fn)]
// The public unsafe functions in this crate are OpenSlide C ABI exports, not
// idiomatic Rust APIs. Their safety contract is the upstream openslide.h ABI.
#![allow(clippy::missing_safety_doc)]

pub mod install;

mod handle;
mod pixels;

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::ptr;

use handle::{empty_names, OpenSlideHandle};
use statumen::{PlaneSelection, RegionRequest};

const VERSION: &[u8] = b"OpenSlide-statumen 4.0.0+statumen-0.2.0\0";

#[repr(C)]
pub struct openslide_t {
    _private: [u8; 0],
}

#[repr(C)]
pub struct openslide_cache_t {
    _private: [u8; 0],
}

struct ShimCache {
    _capacity: usize,
}

type FfiResult<T> = Result<T, FfiPanic>;

struct FfiPanic;

fn guard<T>(fallback: T, f: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(fallback)
}

unsafe fn handle_ref<'a>(osr: *mut openslide_t) -> Option<&'a OpenSlideHandle> {
    if osr.is_null() {
        return None;
    }
    // SAFETY: Non-null handles returned by `openslide_open` are Box allocations
    // of OpenSlideHandle cast to the opaque C handle type.
    Some(unsafe { &*(osr.cast::<OpenSlideHandle>()) })
}

unsafe fn take_handle(osr: *mut openslide_t) -> Option<Box<OpenSlideHandle>> {
    if osr.is_null() {
        return None;
    }
    // SAFETY: `openslide_close` is the single owner that reconstructs the Box
    // from the raw handle returned by `openslide_open`.
    Some(unsafe { Box::from_raw(osr.cast::<OpenSlideHandle>()) })
}

unsafe fn cache_from_raw(cache: *mut openslide_cache_t) -> Option<Box<ShimCache>> {
    if cache.is_null() {
        return None;
    }
    // SAFETY: `openslide_cache_release` is the single owner that reconstructs
    // the Box from the raw cache pointer returned by `openslide_cache_create`.
    Some(unsafe { Box::from_raw(cache.cast::<ShimCache>()) })
}

unsafe fn path_from_c(filename: *const c_char) -> Option<PathBuf> {
    if filename.is_null() {
        return None;
    }
    // SAFETY: OpenSlide callers pass NUL-terminated C path strings; null was
    // checked above before constructing the borrowed CStr.
    let bytes = unsafe { CStr::from_ptr(filename) }.to_bytes();
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        Some(PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
    }
    #[cfg(not(unix))]
    {
        Some(PathBuf::from(String::from_utf8_lossy(bytes).into_owned()))
    }
}

fn checked_pixel_len(w: i64, h: i64) -> Option<usize> {
    if w < 0 || h < 0 {
        return None;
    }
    (w as usize).checked_mul(h as usize)
}

unsafe fn clear_u32(dest: *mut u32, len: usize) {
    if !dest.is_null() && len > 0 {
        // SAFETY: Caller supplies a writable `dest` buffer with at least `len`
        // u32 elements by OpenSlide ABI contract; null and zero length are
        // handled above.
        unsafe { ptr::write_bytes(dest, 0, len) };
    }
}

fn first_series(handle: &OpenSlideHandle) -> Option<&statumen::Series> {
    handle
        .slide()?
        .dataset()
        .scenes
        .first()
        .and_then(|scene| scene.series.first())
}

#[no_mangle]
pub unsafe extern "C" fn openslide_detect_vendor(filename: *const c_char) -> *const c_char {
    guard(ptr::null(), || {
        // SAFETY: The C ABI accepts null; `path_from_c` validates before
        // borrowing the filename.
        let Some(path) = (unsafe { path_from_c(filename) }) else {
            return ptr::null();
        };
        OpenSlideHandle::detect_vendor(path)
    })
}

#[no_mangle]
pub unsafe extern "C" fn openslide_open(filename: *const c_char) -> *mut openslide_t {
    guard(ptr::null_mut(), || {
        // SAFETY: The C ABI accepts null; `path_from_c` validates before
        // borrowing the filename.
        let Some(path) = (unsafe { path_from_c(filename) }) else {
            return ptr::null_mut();
        };
        OpenSlideHandle::open(path)
            .map(Box::into_raw)
            .map(|ptr| ptr.cast::<openslide_t>())
            .unwrap_or(ptr::null_mut())
    })
}

#[no_mangle]
pub unsafe extern "C" fn openslide_close(osr: *mut openslide_t) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The C ABI permits null, and `take_handle` validates before
        // reconstructing ownership of the handle allocation.
        let _ = unsafe { take_handle(osr) };
    }));
}

#[no_mangle]
pub unsafe extern "C" fn openslide_get_error(osr: *mut openslide_t) -> *const c_char {
    guard(ptr::null(), || {
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        let Some(handle) = (unsafe { handle_ref(osr) }) else {
            return ptr::null();
        };
        handle.error_ptr()
    })
}

#[no_mangle]
pub unsafe extern "C" fn openslide_get_version() -> *const c_char {
    VERSION.as_ptr().cast::<c_char>()
}

#[no_mangle]
pub unsafe extern "C" fn openslide_get_level_count(osr: *mut openslide_t) -> c_int {
    guard(-1, || {
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        let Some(handle) = (unsafe { handle_ref(osr) }) else {
            return -1;
        };
        let Some(series) = first_series(handle) else {
            return -1;
        };
        i32::try_from(series.levels.len()).unwrap_or(-1)
    })
}

#[no_mangle]
pub unsafe extern "C" fn openslide_get_level0_dimensions(
    osr: *mut openslide_t,
    w: *mut i64,
    h: *mut i64,
) {
    // SAFETY: This forwards the same C ABI pointers to the level-dimensions
    // implementation, which validates null output pointers before writes.
    unsafe { openslide_get_level_dimensions(osr, 0, w, h) };
}

#[no_mangle]
pub unsafe extern "C" fn openslide_get_level_dimensions(
    osr: *mut openslide_t,
    level: c_int,
    w: *mut i64,
    h: *mut i64,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Output pointers are optional in the OpenSlide ABI; each is
        // checked for null before writing the fallback values.
        unsafe {
            if !w.is_null() {
                *w = -1;
            }
            if !h.is_null() {
                *h = -1;
            }
        }
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        let Some(handle) = (unsafe { handle_ref(osr) }) else {
            return;
        };
        let Some(series) = first_series(handle) else {
            return;
        };
        if level < 0 {
            return;
        }
        let Some(level) = series.levels.get(level as usize) else {
            return;
        };
        // SAFETY: Output pointers are optional and are checked for null before
        // writing dimensions.
        unsafe {
            if !w.is_null() {
                *w = i64::try_from(level.dimensions.0).unwrap_or(-1);
            }
            if !h.is_null() {
                *h = i64::try_from(level.dimensions.1).unwrap_or(-1);
            }
        }
    }));
}

#[no_mangle]
pub unsafe extern "C" fn openslide_get_level_downsample(
    osr: *mut openslide_t,
    level: c_int,
) -> f64 {
    guard(-1.0, || {
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        let Some(handle) = (unsafe { handle_ref(osr) }) else {
            return -1.0;
        };
        let Some(series) = first_series(handle) else {
            return -1.0;
        };
        if level < 0 {
            return -1.0;
        }
        series
            .levels
            .get(level as usize)
            .map(|level| level.downsample)
            .unwrap_or(-1.0)
    })
}

#[no_mangle]
pub unsafe extern "C" fn openslide_get_best_level_for_downsample(
    osr: *mut openslide_t,
    downsample: f64,
) -> c_int {
    guard(-1, || {
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        let Some(handle) = (unsafe { handle_ref(osr) }) else {
            return -1;
        };
        let Some(series) = first_series(handle) else {
            return -1;
        };
        if series.levels.is_empty() || !downsample.is_finite() {
            return -1;
        }
        let mut best = 0usize;
        let mut best_delta = f64::INFINITY;
        for (idx, level) in series.levels.iter().enumerate() {
            let delta = (level.downsample - downsample).abs();
            if delta < best_delta {
                best = idx;
                best_delta = delta;
            }
        }
        i32::try_from(best).unwrap_or(-1)
    })
}

#[no_mangle]
pub unsafe extern "C" fn openslide_read_region(
    osr: *mut openslide_t,
    dest: *mut u32,
    x: i64,
    y: i64,
    level: c_int,
    w: i64,
    h: i64,
) {
    let len = checked_pixel_len(w, h).unwrap_or(0);
    let result: FfiResult<()> = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: `len` is derived from requested dimensions, and `clear_u32`
        // checks for null before writing.
        unsafe { clear_u32(dest, len) };
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        let Some(handle) = (unsafe { handle_ref(osr) }) else {
            return Err(FfiPanic);
        };
        if handle.has_error() {
            return Err(FfiPanic);
        }
        if dest.is_null() {
            handle.set_error("openslide_read_region destination is NULL");
            return Err(FfiPanic);
        }
        if level < 0 {
            handle.set_error(format!("level {level} out of range"));
            return Err(FfiPanic);
        }
        let Some(slide) = handle.slide() else {
            return Err(FfiPanic);
        };
        let Some(series) = first_series(handle) else {
            handle.set_error("slide has no readable series");
            return Err(FfiPanic);
        };
        let Some(level_meta) = series.levels.get(level as usize) else {
            handle.set_error(format!("level {level} out of range"));
            return Err(FfiPanic);
        };
        let Some(pixel_len) = checked_pixel_len(w, h) else {
            handle.set_error("region dimensions must be non-negative");
            return Err(FfiPanic);
        };
        let width = match u32::try_from(w) {
            Ok(width) => width,
            Err(_) => {
                handle.set_error(format!("region width {w} exceeds u32 range"));
                return Err(FfiPanic);
            }
        };
        let height = match u32::try_from(h) {
            Ok(height) => height,
            Err(_) => {
                handle.set_error(format!("region height {h} exceeds u32 range"));
                return Err(FfiPanic);
            }
        };
        let downsample = level_meta.downsample;
        let level_x = if downsample > 0.0 {
            (x as f64 / downsample).floor() as i64
        } else {
            x
        };
        let level_y = if downsample > 0.0 {
            (y as f64 / downsample).floor() as i64
        } else {
            y
        };
        let req = RegionRequest::legacy_xywh(
            0,
            0,
            level as u32,
            PlaneSelection::default(),
            level_x,
            level_y,
            width,
            height,
        );
        match slide
            .read_region(&req)
            .and_then(pixels::tile_to_premultiplied_argb)
        {
            Ok(argb) if argb.len() == pixel_len => {
                // SAFETY: `dest` was checked non-null above and the source
                // buffer length exactly matches the requested pixel count.
                unsafe { ptr::copy_nonoverlapping(argb.as_ptr(), dest, argb.len()) };
                Ok(())
            }
            Ok(argb) => {
                handle.set_error(format!(
                    "read_region returned {} pixels for requested {} pixels",
                    argb.len(),
                    pixel_len
                ));
                Err(FfiPanic)
            }
            Err(err) => {
                handle.set_error(err.to_string());
                Err(FfiPanic)
            }
        }
    }))
    .unwrap_or(Err(FfiPanic));
    if result.is_err() {
        // SAFETY: `clear_u32` validates null and zero length before writing.
        unsafe { clear_u32(dest, len) };
    }
}

#[no_mangle]
pub unsafe extern "C" fn openslide_get_icc_profile_size(osr: *mut openslide_t) -> i64 {
    guard(-1, || {
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        let Some(handle) = (unsafe { handle_ref(osr) }) else {
            return -1;
        };
        let Some(profile) = handle.icc_profile() else {
            return -1;
        };
        i64::try_from(profile.len()).unwrap_or(-1)
    })
}

#[no_mangle]
pub unsafe extern "C" fn openslide_read_icc_profile(osr: *mut openslide_t, dest: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        let Some(handle) = (unsafe { handle_ref(osr) }) else {
            return;
        };
        let Some(profile) = handle.icc_profile() else {
            return;
        };
        if dest.is_null() {
            handle.set_error("openslide_read_icc_profile destination is NULL");
            return;
        }
        // SAFETY: `dest` was checked non-null and the caller is responsible for
        // providing a buffer of `openslide_get_icc_profile_size` bytes.
        unsafe { ptr::copy_nonoverlapping(profile.as_ptr(), dest.cast::<u8>(), profile.len()) };
    }));
}

#[no_mangle]
pub unsafe extern "C" fn openslide_get_property_names(
    osr: *mut openslide_t,
) -> *const *const c_char {
    guard(empty_names(), || {
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        let Some(handle) = (unsafe { handle_ref(osr) }) else {
            return empty_names();
        };
        handle.property_names()
    })
}

#[no_mangle]
pub unsafe extern "C" fn openslide_get_property_value(
    osr: *mut openslide_t,
    name: *const c_char,
) -> *const c_char {
    guard(ptr::null(), || {
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        let Some(handle) = (unsafe { handle_ref(osr) }) else {
            return ptr::null();
        };
        if name.is_null() {
            return ptr::null();
        }
        // SAFETY: `name` was checked non-null and must be NUL-terminated by
        // the OpenSlide ABI caller.
        let name = unsafe { CStr::from_ptr(name) };
        handle.property_value(name)
    })
}

#[no_mangle]
pub unsafe extern "C" fn openslide_get_associated_image_names(
    osr: *mut openslide_t,
) -> *const *const c_char {
    guard(empty_names(), || {
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        let Some(handle) = (unsafe { handle_ref(osr) }) else {
            return empty_names();
        };
        handle.associated_names()
    })
}

#[no_mangle]
pub unsafe extern "C" fn openslide_get_associated_image_dimensions(
    osr: *mut openslide_t,
    name: *const c_char,
    w: *mut i64,
    h: *mut i64,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Output pointers are optional in the OpenSlide ABI; each is
        // checked for null before writing fallback values.
        unsafe {
            if !w.is_null() {
                *w = -1;
            }
            if !h.is_null() {
                *h = -1;
            }
        }
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        let Some(handle) = (unsafe { handle_ref(osr) }) else {
            return;
        };
        if name.is_null() {
            handle.set_error("associated image name is NULL");
            return;
        }
        // SAFETY: `name` was checked non-null and must be NUL-terminated by
        // the OpenSlide ABI caller.
        let name = unsafe { CStr::from_ptr(name) };
        let Some(info) = handle.associated_image_info(name) else {
            handle.set_error("associated image not found");
            return;
        };
        // SAFETY: Output pointers are optional and are checked for null before
        // writing dimensions.
        unsafe {
            if !w.is_null() {
                *w = i64::from(info.width);
            }
            if !h.is_null() {
                *h = i64::from(info.height);
            }
        }
    }));
}

#[no_mangle]
pub unsafe extern "C" fn openslide_read_associated_image(
    osr: *mut openslide_t,
    name: *const c_char,
    dest: *mut u32,
) {
    let result: FfiResult<()> = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        let Some(handle) = (unsafe { handle_ref(osr) }) else {
            return Err(FfiPanic);
        };
        if handle.has_error() {
            return Err(FfiPanic);
        }
        if name.is_null() {
            handle.set_error("associated image name is NULL");
            return Err(FfiPanic);
        }
        if dest.is_null() {
            handle.set_error("associated image destination is NULL");
            return Err(FfiPanic);
        }
        // SAFETY: `name` was checked non-null and must be NUL-terminated by
        // the OpenSlide ABI caller.
        let name = unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned();
        let Some(slide) = handle.slide() else {
            return Err(FfiPanic);
        };
        let Some(info) = handle.associated_image_info(&handle::cstring_sanitized(&name)) else {
            handle.set_error(format!("associated image not found: {name}"));
            return Err(FfiPanic);
        };
        let len = usize::try_from(info.width)
            .ok()
            .and_then(|width| width.checked_mul(info.height as usize))
            .unwrap_or(0);
        // SAFETY: `clear_u32` validates null and zero length before writing.
        unsafe { clear_u32(dest, len) };
        match slide
            .read_associated(&name)
            .and_then(pixels::tile_to_premultiplied_argb)
        {
            Ok(argb) if argb.len() == len => {
                // SAFETY: `dest` was checked non-null above and the source
                // buffer length exactly matches the associated image size.
                unsafe { ptr::copy_nonoverlapping(argb.as_ptr(), dest, argb.len()) };
                Ok(())
            }
            Ok(argb) => {
                handle.set_error(format!(
                    "associated image returned {} pixels for expected {} pixels",
                    argb.len(),
                    len
                ));
                Err(FfiPanic)
            }
            Err(err) => {
                handle.set_error(err.to_string());
                Err(FfiPanic)
            }
        }
    }))
    .unwrap_or(Err(FfiPanic));
    if result.is_err() {
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        if let Some(handle) = unsafe { handle_ref(osr) } {
            if !name.is_null() {
                // SAFETY: `name` was checked non-null and must be
                // NUL-terminated by the OpenSlide ABI caller.
                let name = unsafe { CStr::from_ptr(name) };
                if let Some(info) = handle.associated_image_info(name) {
                    let len = info.width as usize * info.height as usize;
                    // SAFETY: `clear_u32` validates null and zero length
                    // before writing.
                    unsafe { clear_u32(dest, len) };
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn openslide_get_associated_image_icc_profile_size(
    osr: *mut openslide_t,
    name: *const c_char,
) -> i64 {
    guard(-1, || {
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        let Some(handle) = (unsafe { handle_ref(osr) }) else {
            return -1;
        };
        if handle.has_error() {
            return -1;
        }
        if name.is_null() {
            handle.set_error("associated image name is NULL");
            return -1;
        }
        // SAFETY: `name` was checked non-null and must be NUL-terminated by
        // the OpenSlide ABI caller.
        let name = unsafe { CStr::from_ptr(name) };
        if handle.associated_image_info(name).is_none() {
            handle.set_error("associated image not found");
            return -1;
        }
        0
    })
}

#[no_mangle]
pub unsafe extern "C" fn openslide_read_associated_image_icc_profile(
    osr: *mut openslide_t,
    name: *const c_char,
    dest: *mut c_void,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        let Some(handle) = (unsafe { handle_ref(osr) }) else {
            return;
        };
        if handle.has_error() {
            return;
        }
        if name.is_null() {
            handle.set_error("associated image name is NULL");
            return;
        }
        if dest.is_null() {
            handle.set_error("associated ICC destination is NULL");
            return;
        }
        // SAFETY: `name` was checked non-null and must be NUL-terminated by
        // the OpenSlide ABI caller.
        let name = unsafe { CStr::from_ptr(name) };
        if handle.associated_image_info(name).is_none() {
            handle.set_error("associated image not found");
        }
    }));
}

#[no_mangle]
pub unsafe extern "C" fn openslide_cache_create(capacity: usize) -> *mut openslide_cache_t {
    guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(ShimCache {
            _capacity: capacity,
        }))
        .cast::<openslide_cache_t>()
    })
}

#[no_mangle]
pub unsafe extern "C" fn openslide_set_cache(
    osr: *mut openslide_t,
    _cache: *mut openslide_cache_t,
) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The C ABI permits null, and `handle_ref` validates before
        // borrowing the opaque handle.
        let Some(handle) = (unsafe { handle_ref(osr) }) else {
            return;
        };
        let _ = handle.has_error();
    }));
}

#[no_mangle]
pub unsafe extern "C" fn openslide_cache_release(cache: *mut openslide_cache_t) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The C ABI permits null, and `cache_from_raw` validates before
        // reconstructing ownership of the cache allocation.
        let _ = unsafe { cache_from_raw(cache) };
    }));
}
