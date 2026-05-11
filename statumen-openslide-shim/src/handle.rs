use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use statumen::{Slide, TileLayout, WsiError};

const FALLBACK_CSTRING_BYTES: &[u8] = b"statumen string conversion failed\0";

static DETECTED_VENDORS: OnceLock<Mutex<Vec<CString>>> = OnceLock::new();

pub(crate) struct OpenSlideHandle {
    slide: Option<Slide>,
    error: Mutex<Option<CString>>,
    properties: BTreeMap<String, CString>,
    property_names: CStringArray,
    associated_images: BTreeMap<String, AssociatedImageInfo>,
    associated_names: CStringArray,
    icc_profile: Vec<u8>,
}

pub(crate) struct AssociatedImageInfo {
    pub(crate) width: u32,
    pub(crate) height: u32,
}

pub(crate) struct CStringArray {
    _strings: Vec<CString>,
    ptrs: Vec<*const c_char>,
}

// SAFETY: CStringArray is immutable after construction. The raw pointers point
// into heap allocations owned by `_strings` and remain valid for the handle
// lifetime.
unsafe impl Send for CStringArray {}
// SAFETY: Shared access cannot mutate `_strings` or `ptrs`, so the stored C
// string pointers remain stable while the owning handle is alive.
unsafe impl Sync for CStringArray {}

impl CStringArray {
    pub(crate) fn empty() -> Self {
        Self {
            _strings: Vec::new(),
            ptrs: vec![std::ptr::null()],
        }
    }

    fn from_names(names: impl IntoIterator<Item = String>) -> Self {
        let strings = names.into_iter().map(cstring_sanitized).collect::<Vec<_>>();
        let mut ptrs = strings
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();
        ptrs.push(std::ptr::null());
        Self {
            _strings: strings,
            ptrs,
        }
    }

    pub(crate) fn as_ptr(&self) -> *const *const c_char {
        self.ptrs.as_ptr()
    }
}

impl OpenSlideHandle {
    pub(crate) fn open(path: PathBuf) -> Option<Box<Self>> {
        match Slide::open(&path) {
            Ok(slide) => Some(Box::new(Self::from_slide(slide))),
            Err(err) if should_open_return_null(&err) => None,
            Err(err) => Some(Box::new(Self::from_error(err.to_string()))),
        }
    }

    pub(crate) fn detect_vendor(path: PathBuf) -> *const c_char {
        let Ok(slide) = Slide::open(path) else {
            return std::ptr::null();
        };
        let Some(vendor) = vendor_for_slide(&slide) else {
            return std::ptr::null();
        };
        intern_detected_vendor(vendor)
    }

    fn from_slide(slide: Slide) -> Self {
        let properties = build_properties(&slide);
        let property_names = CStringArray::from_names(properties.keys().cloned());
        let associated_images = slide
            .dataset()
            .associated_images
            .iter()
            .map(|(name, image)| {
                (
                    name.clone(),
                    AssociatedImageInfo {
                        width: image.dimensions.0,
                        height: image.dimensions.1,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let associated_names = CStringArray::from_names(associated_images.keys().cloned());
        let icc_profile = slide
            .dataset()
            .icc_profiles
            .get(&(0, 0))
            .cloned()
            .unwrap_or_default();

        Self {
            slide: Some(slide),
            error: Mutex::new(None),
            properties,
            property_names,
            associated_images,
            associated_names,
            icc_profile,
        }
    }

    fn from_error(message: String) -> Self {
        Self {
            slide: None,
            error: Mutex::new(Some(cstring_sanitized(message))),
            properties: BTreeMap::new(),
            property_names: CStringArray::empty(),
            associated_images: BTreeMap::new(),
            associated_names: CStringArray::empty(),
            icc_profile: Vec::new(),
        }
    }

    pub(crate) fn slide(&self) -> Option<&Slide> {
        if self.has_error() {
            return None;
        }
        self.slide.as_ref()
    }

    pub(crate) fn set_error(&self, message: impl Into<String>) {
        let mut error = self.error.lock().expect("OpenSlide error mutex poisoned");
        if error.is_none() {
            *error = Some(cstring_sanitized(message.into()));
        }
    }

    pub(crate) fn has_error(&self) -> bool {
        self.error
            .lock()
            .expect("OpenSlide error mutex poisoned")
            .is_some()
    }

    pub(crate) fn error_ptr(&self) -> *const c_char {
        let error = self.error.lock().expect("OpenSlide error mutex poisoned");
        error
            .as_ref()
            .map(|message| message.as_ptr())
            .unwrap_or(std::ptr::null())
    }

    pub(crate) fn property_names(&self) -> *const *const c_char {
        if self.has_error() {
            return empty_names();
        }
        self.property_names.as_ptr()
    }

    pub(crate) fn property_value(&self, name: &CStr) -> *const c_char {
        if self.has_error() {
            return std::ptr::null();
        }
        let name = name.to_string_lossy();
        self.properties
            .get(name.as_ref())
            .map(|value| value.as_ptr())
            .unwrap_or(std::ptr::null())
    }

    pub(crate) fn associated_names(&self) -> *const *const c_char {
        if self.has_error() {
            return empty_names();
        }
        self.associated_names.as_ptr()
    }

    pub(crate) fn associated_image_info(&self, name: &CStr) -> Option<&AssociatedImageInfo> {
        if self.has_error() {
            return None;
        }
        let name = name.to_string_lossy();
        self.associated_images.get(name.as_ref())
    }

    pub(crate) fn icc_profile(&self) -> Option<&[u8]> {
        if self.has_error() {
            return None;
        }
        Some(&self.icc_profile)
    }
}

pub(crate) fn empty_names() -> *const *const c_char {
    static EMPTY_NAMES: [usize; 1] = [0];
    EMPTY_NAMES.as_ptr().cast::<*const c_char>()
}

pub(crate) fn cstring_sanitized(value: impl AsRef<str>) -> CString {
    let sanitized = value.as_ref().replace('\0', " ");
    CString::new(sanitized).unwrap_or_else(|_| {
        CStr::from_bytes_with_nul(FALLBACK_CSTRING_BYTES)
            .expect("fallback CString is valid")
            .to_owned()
    })
}

fn intern_detected_vendor(vendor: &str) -> *const c_char {
    let vendors = DETECTED_VENDORS.get_or_init(|| Mutex::new(Vec::new()));
    let mut vendors = vendors.lock().expect("detected vendor mutex poisoned");
    if let Some(existing) = vendors
        .iter()
        .find(|existing| existing.to_bytes() == vendor.as_bytes())
    {
        return existing.as_ptr();
    }
    vendors.push(cstring_sanitized(vendor));
    vendors.last().expect("vendor was just pushed").as_ptr()
}

fn should_open_return_null(err: &WsiError) -> bool {
    match err {
        WsiError::UnsupportedFormat(_) => true,
        WsiError::Io(source) => source.kind() == std::io::ErrorKind::NotFound,
        WsiError::IoWithPath { source, .. } => source.kind() == std::io::ErrorKind::NotFound,
        _ => false,
    }
}

fn build_properties(slide: &Slide) -> BTreeMap<String, CString> {
    let dataset = slide.dataset();
    let mut properties = dataset
        .properties
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect::<BTreeMap<_, _>>();

    if let Some(vendor) = vendor_for_slide(slide) {
        properties
            .entry("openslide.vendor".to_string())
            .or_insert_with(|| vendor.to_string());
    }

    if let Some(series) = dataset
        .scenes
        .first()
        .and_then(|scene| scene.series.first())
    {
        properties.insert(
            "openslide.level-count".to_string(),
            series.levels.len().to_string(),
        );

        for (idx, level) in series.levels.iter().enumerate() {
            properties.insert(
                format!("openslide.level[{idx}].width"),
                level.dimensions.0.to_string(),
            );
            properties.insert(
                format!("openslide.level[{idx}].height"),
                level.dimensions.1.to_string(),
            );
            properties.insert(
                format!("openslide.level[{idx}].downsample"),
                format_float(level.downsample),
            );
            let (tile_width, tile_height) = tile_size(&level.tile_layout);
            properties.insert(
                format!("openslide.level[{idx}].tile-width"),
                tile_width.to_string(),
            );
            properties.insert(
                format!("openslide.level[{idx}].tile-height"),
                tile_height.to_string(),
            );
        }

        if let Some(level0) = series.levels.first() {
            properties
                .entry("openslide.bounds-x".to_string())
                .or_insert_with(|| "0".to_string());
            properties
                .entry("openslide.bounds-y".to_string())
                .or_insert_with(|| "0".to_string());
            properties
                .entry("openslide.bounds-width".to_string())
                .or_insert_with(|| level0.dimensions.0.to_string());
            properties
                .entry("openslide.bounds-height".to_string())
                .or_insert_with(|| level0.dimensions.1.to_string());
        }
    }

    if !dataset.icc_profiles.is_empty() {
        if let Some(profile) = dataset.icc_profiles.get(&(0, 0)) {
            properties.insert("openslide.icc-size".to_string(), profile.len().to_string());
        }
    }

    properties
        .into_iter()
        .map(|(key, value)| (key, cstring_sanitized(value)))
        .collect()
}

fn vendor_for_slide(slide: &Slide) -> Option<&str> {
    slide.dataset().properties.vendor().or_else(|| {
        slide
            .dataset()
            .scenes
            .first()
            .map(|scene| scene.id.as_str())
            .filter(|id| !id.is_empty())
    })
}

fn tile_size(layout: &TileLayout) -> (u32, u32) {
    match layout {
        TileLayout::Regular {
            tile_width,
            tile_height,
            ..
        } => (*tile_width, *tile_height),
        TileLayout::WholeLevel {
            virtual_tile_width,
            virtual_tile_height,
            ..
        } => (*virtual_tile_width, *virtual_tile_height),
        TileLayout::Irregular { tile_advance, .. } => (
            tile_advance.0.round().max(1.0) as u32,
            tile_advance.1.round().max(1.0) as u32,
        ),
    }
}

fn format_float(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        value.to_string()
    }
}
