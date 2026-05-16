use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ::image::imageops;
use dicom_dictionary_std::{tags, uids};
use dicom_object::{meta::FileMetaTable, DefaultDicomObject, OpenFileOptions};
use dicom_parser::dataset::{lazy_read::LazyDataSetReader, LazyDataToken};
use dicom_parser::stateful::decode::StatefulDecode;
use dicom_transfer_syntax_registry::{TransferSyntaxIndex, TransferSyntaxRegistry};
use lru::LruCache;
use signinum_core::BackendRequest;

use crate::core::hash::Quickhash1;
use crate::core::registry::{
    DatasetReader, FormatProbe, ProbeConfidence, ProbeResult, SlideReader,
};
use crate::core::types::*;
#[cfg(feature = "metal")]
use crate::decode::jp2k::decode_batch_jp2k_pixels;
use crate::decode::jp2k::{decode_batch_jp2k, Jp2kDecodeJob};
#[cfg(feature = "metal")]
use crate::decode::jpeg::decode_batch_jpeg_pixels;
use crate::decode::jpeg::{decode_batch_jpeg, JpegDecodeJob};
use crate::error::WsiError;
use crate::properties::Properties;

const LEVEL_IMAGE_TYPES: &[&[&str]] = &[
    &["ORIGINAL", "PRIMARY", "VOLUME", "NONE"],
    &["DERIVED", "PRIMARY", "VOLUME", "NONE"],
    &["DERIVED", "PRIMARY", "VOLUME", "RESAMPLED"],
];
const LABEL_IMAGE_TYPES: &[&[&str]] = &[
    &["ORIGINAL", "PRIMARY", "LABEL", "NONE"],
    &["DERIVED", "PRIMARY", "LABEL", "NONE"],
];
const OVERVIEW_IMAGE_TYPES: &[&[&str]] = &[
    &["ORIGINAL", "PRIMARY", "OVERVIEW", "NONE"],
    &["DERIVED", "PRIMARY", "OVERVIEW", "NONE"],
];
const THUMBNAIL_IMAGE_TYPES: &[&[&str]] = &[
    &["ORIGINAL", "PRIMARY", "THUMBNAIL", "RESAMPLED"],
    &["DERIVED", "PRIMARY", "THUMBNAIL", "RESAMPLED"],
];
const BASE_ONLY_DICOM_PYRAMID_MESSAGE: &str = "This DICOM WSI contains only a full-resolution base layer and no physical pyramid levels. Open the complete DICOM series/folder, or regenerate the DICOM with DERIVED/PRIMARY/VOLUME/RESAMPLED pyramid instances.";
const BASE_ONLY_GUARD_MIN_TILE_COUNT: u64 = 4_096;
const BASE_ONLY_GUARD_MIN_DIMENSION: u32 = 32_768;
const SUPPORTED_TRANSFER_SYNTAXES: &[&str] = &[
    uids::IMPLICIT_VR_LITTLE_ENDIAN,
    uids::EXPLICIT_VR_LITTLE_ENDIAN,
    EXPLICIT_VR_BIG_ENDIAN_TRANSFER_SYNTAX,
    uids::JPEG_BASELINE8_BIT,
    uids::JPEG2000_LOSSLESS,
    uids::JPEG2000,
    HTJ2K_LOSSLESS_TRANSFER_SYNTAX,
    HTJ2K_LOSSLESS_RPCL_TRANSFER_SYNTAX,
    uids::RLE_LOSSLESS,
];
const JPEG_TRANSFER_SYNTAX: &str = uids::JPEG_BASELINE8_BIT;
const RLE_TRANSFER_SYNTAX: &str = uids::RLE_LOSSLESS;
const EXPLICIT_VR_BIG_ENDIAN_TRANSFER_SYNTAX: &str = "1.2.840.10008.1.2.2";
const HTJ2K_LOSSLESS_TRANSFER_SYNTAX: &str = "1.2.840.10008.1.2.4.201";
const HTJ2K_LOSSLESS_RPCL_TRANSFER_SYNTAX: &str = "1.2.840.10008.1.2.4.202";
const JP2K_TRANSFER_SYNTAXES: &[&str] = &[
    uids::JPEG2000_LOSSLESS,
    uids::JPEG2000,
    HTJ2K_LOSSLESS_TRANSFER_SYNTAX,
    HTJ2K_LOSSLESS_RPCL_TRANSFER_SYNTAX,
];
#[cfg(feature = "metal")]
const DICOM_JP2K_DEVICE_DECODE_ENV: &str = "STATUMEN_JP2K_DEVICE_DECODE";

mod backend;
mod decode;
mod image;
mod manifest;
mod metadata;
mod reader;

pub(crate) use backend::DicomBackend;
use backend::*;
use decode::*;
use image::*;
use manifest::*;
use metadata::*;
use reader::*;

#[cfg(test)]
mod tests;
