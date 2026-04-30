//! Leica SCN layout interpreter.
//!
//! SCN files can contain multiple acquisition scenes, fluorescence channels,
//! z-planes, and overview images. This interpreter exposes those semantics
//! directly through the normalized scene/series/plane model instead of
//! flattening the file into a single brightfield-only view.

use std::collections::{BTreeMap, HashMap};

use base64::prelude::BASE64_STANDARD;
use base64::Engine;

use crate::core::types::*;
use crate::decode::xml;
use crate::formats::tiff_family::container::{tags, TiffContainer};
use crate::formats::tiff_family::error::{IfdId, TiffParseError};
use crate::properties::Properties;

use super::{
    compute_tiff_dataset_identity, DatasetLayout, TiffLayoutInterpreter, TileSource, TileSourceKey,
};

const LEICA_NS_2010_03: &str = "http://www.leica-microsystems.com/scn/2010/03/10";
const LEICA_NS_2010_10: &str = "http://www.leica-microsystems.com/scn/2010/10/01";

pub(crate) struct LeicaInterpreter;

#[derive(Clone)]
struct ParsedDimension {
    ifd_index: usize,
    width: u64,
    height: u64,
    r: u32,
    z: u32,
    c: u32,
}

#[derive(Clone)]
struct ParsedChannel {
    index: u32,
    info: ChannelInfo,
}

struct LeicaImageInfo {
    name: Option<String>,
    ifd_levels: Vec<ParsedDimension>,
    channels: Vec<ParsedChannel>,
    view_size: (u64, u64),
    view_offset: (i64, i64),
    creation_date: Option<String>,
    device_model: Option<String>,
    device_version: Option<String>,
    objective: Option<String>,
    numerical_aperture: Option<String>,
    illumination_source: Option<String>,
    is_macro: bool,
}

fn compression_from_tag(val: u32) -> Compression {
    match val {
        1 => Compression::None,
        5 => Compression::Lzw,
        8 | 32946 => Compression::Deflate,
        7 | 6 => Compression::Jpeg,
        50000 => Compression::Zstd,
        33003 | 33005 => Compression::Jp2kYcbcr,
        33004 => Compression::Jp2kRgb,
        _ => Compression::Other(val as u16),
    }
}

impl TiffLayoutInterpreter for LeicaInterpreter {
    fn vendor_name(&self) -> &'static str {
        "leica"
    }

    fn detect(&self, container: &TiffContainer) -> bool {
        let Some(&first_ifd) = container.top_ifds().first() else {
            return false;
        };
        let Ok(desc) = container.get_string(first_ifd, tags::IMAGE_DESCRIPTION) else {
            return false;
        };
        let lower = desc.to_ascii_lowercase();
        lower.contains("<scn") || desc.contains(LEICA_NS_2010_03) || desc.contains(LEICA_NS_2010_10)
    }

    fn interpret(&self, container: &TiffContainer) -> Result<DatasetLayout, TiffParseError> {
        let first_ifd = *container
            .top_ifds()
            .first()
            .ok_or_else(|| TiffParseError::Structure("No IFDs in container".into()))?;
        let xml_str = container
            .get_string(first_ifd, tags::IMAGE_DESCRIPTION)
            .map_err(|_| {
                TiffParseError::Structure("Leica SCN: missing ImageDescription XML".into())
            })?;
        let root = xml::parse_xml(xml_str).map_err(|e| TiffParseError::Structure(e.to_string()))?;
        let collection = if root.tag == "collection" {
            &root
        } else {
            root.find("collection").ok_or_else(|| {
                TiffParseError::Structure("Leica SCN: no <collection> element found in XML".into())
            })?
        };

        let collection_size = (
            parse_required_u64_attr(collection, "sizeX", "Leica SCN collection")?,
            parse_required_u64_attr(collection, "sizeY", "Leica SCN collection")?,
        );

        let images = collection.find_all("image");
        if images.is_empty() {
            return Err(TiffParseError::Structure(
                "Leica SCN: no <image> elements in collection".into(),
            ));
        }

        let top_ifds = container.top_ifds();
        let mut parsed_images = Vec::new();
        for image in images {
            if let Some(parsed) = self.parse_image_info(image, collection_size)? {
                parsed_images.push(parsed);
            }
        }
        if parsed_images.is_empty() {
            return Err(TiffParseError::Structure(
                "Leica SCN: no supported images found".into(),
            ));
        }

        let mut associated_images = HashMap::new();
        let mut associated_sources = HashMap::new();
        let mut identity_lowest_ifd = None;
        for image in parsed_images.iter().filter(|image| image.is_macro) {
            let best = image
                .ifd_levels
                .iter()
                .max_by_key(|level| level.width * level.height)
                .ok_or_else(|| {
                    TiffParseError::Structure("Leica SCN macro image has no levels".into())
                })?;
            let ifd_id = lookup_ifd(top_ifds, best.ifd_index)?;
            let name = unique_associated_name(&associated_images, "macro");
            associated_images.insert(
                name.clone(),
                AssociatedImage {
                    dimensions: (
                        u32::try_from(best.width).unwrap_or(u32::MAX),
                        u32::try_from(best.height).unwrap_or(u32::MAX),
                    ),
                    sample_type: sample_type_for_ifd(container, ifd_id),
                    channels: associated_channel_count(container, ifd_id),
                },
            );
            associated_sources.insert(name, self.associated_source_for_ifd(container, ifd_id)?);
            identity_lowest_ifd.get_or_insert(
                image
                    .ifd_levels
                    .iter()
                    .max_by_key(|level| level.r)
                    .and_then(|level| top_ifds.get(level.ifd_index).copied())
                    .unwrap_or(ifd_id),
            );
        }

        let main_images = parsed_images
            .into_iter()
            .filter(|image| !image.is_macro)
            .collect::<Vec<_>>();
        if main_images.is_empty() {
            return Err(TiffParseError::Structure(
                "Leica SCN: no main images found".into(),
            ));
        }

        let representative = &main_images[0];
        let representative_ifd0 = representative.level0_dimension().ok_or_else(|| {
            TiffParseError::Structure("Leica SCN representative image is missing level 0".into())
        })?;
        let representative_level0_ifd = lookup_ifd(top_ifds, representative_ifd0.ifd_index)?;
        let representative_lowest_ifd = identity_lowest_ifd
            .or_else(|| {
                representative
                    .ifd_levels
                    .iter()
                    .max_by_key(|level| level.r)
                    .and_then(|level| top_ifds.get(level.ifd_index).copied())
            })
            .ok_or_else(|| {
                TiffParseError::Structure(
                    "Leica SCN representative image has no lowest-resolution IFD".into(),
                )
            })?;
        let identity = compute_tiff_dataset_identity(
            container,
            representative_lowest_ifd,
            representative_level0_ifd,
        )?;

        let level0_cpp_x = main_images
            .iter()
            .filter_map(|image| {
                image
                    .level0_dimension()
                    .map(|level0| image.view_size.0 as f64 / level0.width as f64)
            })
            .fold(f64::INFINITY, f64::min);
        let level0_cpp_y = main_images
            .iter()
            .filter_map(|image| {
                image
                    .level0_dimension()
                    .map(|level0| image.view_size.1 as f64 / level0.height as f64)
            })
            .fold(f64::INFINITY, f64::min);

        let mut scenes = Vec::with_capacity(main_images.len());
        let mut tile_sources = HashMap::new();
        for (scene_index, image) in main_images.iter().enumerate() {
            let (levels, level_index_by_r) = self.levels_for_image(container, top_ifds, image)?;
            let axes = image.axes_shape();
            let sample_type = image
                .level0_dimension()
                .and_then(|level| top_ifds.get(level.ifd_index).copied())
                .map(|ifd_id| sample_type_for_ifd(container, ifd_id))
                .unwrap_or(SampleType::Uint8);

            for dim in &image.ifd_levels {
                let ifd_id = lookup_ifd(top_ifds, dim.ifd_index)?;
                let level = *level_index_by_r.get(&dim.r).ok_or_else(|| {
                    TiffParseError::Structure(format!(
                        "Leica SCN image references unmapped resolution {}",
                        dim.r
                    ))
                })?;
                tile_sources.insert(
                    TileSourceKey {
                        scene: scene_index,
                        series: 0,
                        level,
                        z: dim.z,
                        c: dim.c,
                        t: 0,
                    },
                    self.tiled_source_for_ifd(container, ifd_id)?,
                );
            }

            scenes.push(Scene {
                id: format!("s{scene_index}"),
                name: image.name.clone(),
                series: vec![Series {
                    id: "ser0".into(),
                    axes,
                    levels,
                    sample_type,
                    channels: image.channels_for_axes(axes),
                }],
            });
        }

        let mut properties =
            self.parse_public_properties(collection, &main_images, level0_cpp_x, level0_cpp_y)?;
        if let Some(quickhash1) = identity.quickhash1.as_deref() {
            properties.insert("openslide.quickhash-1", quickhash1);
        }

        let dataset = Dataset {
            id: identity.dataset_id,
            scenes,
            associated_images,
            properties,
            icc_profiles: HashMap::new(),
        };

        Ok(DatasetLayout {
            dataset,
            tile_sources,
            associated_sources,
        })
    }
}

impl LeicaInterpreter {
    fn levels_for_image(
        &self,
        container: &TiffContainer,
        top_ifds: &[IfdId],
        image: &LeicaImageInfo,
    ) -> Result<(Vec<Level>, HashMap<u32, u32>), TiffParseError> {
        let mut by_r: BTreeMap<u32, Vec<&ParsedDimension>> = BTreeMap::new();
        for dim in &image.ifd_levels {
            by_r.entry(dim.r).or_default().push(dim);
        }
        let level0 = image.level0_dimension().ok_or_else(|| {
            TiffParseError::Structure("Leica SCN image is missing level 0".into())
        })?;
        let mut levels = Vec::with_capacity(by_r.len());
        let mut level_index_by_r = HashMap::with_capacity(by_r.len());

        for (level_index, (r, dims)) in by_r.iter().enumerate() {
            let representative = dims
                .iter()
                .copied()
                .max_by_key(|dim| dim.width * dim.height)
                .ok_or_else(|| {
                    TiffParseError::Structure(format!(
                        "Leica SCN image has empty resolution group {r}"
                    ))
                })?;
            let ifd_id = lookup_ifd(top_ifds, representative.ifd_index)?;
            let tile_width = container.get_u32(ifd_id, tags::TILE_WIDTH).map_err(|_| {
                TiffParseError::Structure(format!(
                    "Leica SCN main image IFD {} is not tiled",
                    representative.ifd_index
                ))
            })?;
            let tile_height = container.get_u32(ifd_id, tags::TILE_LENGTH).map_err(|_| {
                TiffParseError::Structure(format!(
                    "Leica SCN main image IFD {} is not tiled",
                    representative.ifd_index
                ))
            })?;
            if tile_width == 0 || tile_height == 0 {
                return Err(TiffParseError::Structure(format!(
                    "Leica SCN: invalid tile size {}x{} at level {}",
                    tile_width, tile_height, level_index
                )));
            }

            let width = dims.iter().map(|dim| dim.width).max().unwrap_or(0);
            let height = dims.iter().map(|dim| dim.height).max().unwrap_or(0);
            levels.push(Level {
                dimensions: (width, height),
                downsample: if level_index == 0 {
                    1.0
                } else {
                    level0.width as f64 / width as f64
                },
                tile_layout: TileLayout::Regular {
                    tile_width,
                    tile_height,
                    tiles_across: width.div_ceil(tile_width as u64),
                    tiles_down: height.div_ceil(tile_height as u64),
                },
            });
            level_index_by_r.insert(*r, level_index as u32);
        }

        Ok((levels, level_index_by_r))
    }

    fn tiled_source_for_ifd(
        &self,
        container: &TiffContainer,
        ifd_id: IfdId,
    ) -> Result<TileSource, TiffParseError> {
        container.get_u32(ifd_id, tags::TILE_WIDTH).map_err(|_| {
            TiffParseError::Structure(format!("Leica SCN IFD {:?} is not tiled", ifd_id))
        })?;
        Ok(TileSource::TiledIfd {
            ifd_id,
            jpeg_tables: container
                .get_bytes(ifd_id, tags::JPEG_TABLES)
                .ok()
                .map(|bytes| bytes.to_vec()),
            compression: container
                .get_u32(ifd_id, tags::COMPRESSION)
                .map(compression_from_tag)
                .unwrap_or(Compression::Jpeg),
        })
    }

    fn collect_dimensions(
        &self,
        image_node: &xml::XmlNode,
    ) -> Result<Vec<ParsedDimension>, TiffParseError> {
        let mut dimensions = Vec::new();
        let pixels_nodes = image_node.find_all("pixels");
        if !pixels_nodes.is_empty() {
            for pixels in pixels_nodes {
                for dim in pixels.find_all("dimension") {
                    if let Some(parsed) = self.parse_dimension(dim)? {
                        dimensions.push(parsed);
                    }
                }
            }
        }
        for dim in image_node.find_all("dimension") {
            if let Some(parsed) = self.parse_dimension(dim)? {
                dimensions.push(parsed);
            }
        }
        Ok(dimensions)
    }

    fn parse_dimension(
        &self,
        dim: &xml::XmlNode,
    ) -> Result<Option<ParsedDimension>, TiffParseError> {
        let Some(ifd_str) = dim.attr("ifd") else {
            return Ok(None);
        };
        let ifd_index = ifd_str.parse::<usize>().map_err(|_| {
            TiffParseError::Structure(format!(
                "Leica SCN: invalid ifd index '{}' in <dimension>",
                ifd_str
            ))
        })?;
        let width = dim.attr("sizeX").unwrap_or("0").parse::<u64>().unwrap_or(0);
        let height = dim.attr("sizeY").unwrap_or("0").parse::<u64>().unwrap_or(0);
        if width == 0 || height == 0 {
            return Ok(None);
        }
        let r = dim.attr("r").unwrap_or("0").parse::<u32>().unwrap_or(0);
        let z = dim.attr("z").unwrap_or("0").parse::<u32>().unwrap_or(0);
        let c = dim.attr("c").unwrap_or("0").parse::<u32>().unwrap_or(0);
        Ok(Some(ParsedDimension {
            ifd_index,
            width,
            height,
            r,
            z,
            c,
        }))
    }

    fn parse_image_info(
        &self,
        image_node: &xml::XmlNode,
        collection_size: (u64, u64),
    ) -> Result<Option<LeicaImageInfo>, TiffParseError> {
        let ifd_levels = self.collect_dimensions(image_node)?;
        if ifd_levels.is_empty() {
            return Ok(None);
        }

        let Some(view) = image_node.find("view") else {
            return Ok(None);
        };
        let view_size = (
            parse_required_u64_attr(view, "sizeX", "Leica SCN view")?,
            parse_required_u64_attr(view, "sizeY", "Leica SCN view")?,
        );
        let view_offset = (
            parse_optional_i64_attr(view, "offsetX").unwrap_or(0),
            parse_optional_i64_attr(view, "offsetY").unwrap_or(0),
        );
        let creation_date = image_node
            .find("creationDate")
            .and_then(|node| node.text.as_ref())
            .map(ToOwned::to_owned);
        let device_model = image_node
            .find("device")
            .and_then(|node| node.attr("model"))
            .map(ToOwned::to_owned);
        let device_version = image_node
            .find("device")
            .and_then(|node| node.attr("version"))
            .map(ToOwned::to_owned);
        let objective = image_node
            .find("scanSettings")
            .and_then(|node| node.find("objectiveSettings"))
            .and_then(|node| node.find("objective"))
            .and_then(|node| node.text.as_ref())
            .map(ToOwned::to_owned);
        let numerical_aperture = image_node
            .find("scanSettings")
            .and_then(|node| node.find("illuminationSettings"))
            .and_then(|node| node.find("numericalAperture"))
            .and_then(|node| node.text.as_ref())
            .map(ToOwned::to_owned);
        let illumination_source = image_node
            .find("scanSettings")
            .and_then(|node| node.find("illuminationSettings"))
            .and_then(|node| node.find("illuminationSource"))
            .and_then(|node| node.text.as_ref())
            .map(ToOwned::to_owned);

        let explicit_view_type = view.attr("type").map(|value| value.to_ascii_lowercase());
        let is_macro = explicit_view_type.as_deref() == Some("macro")
            || (view_offset == (0, 0) && view_size == collection_size);

        Ok(Some(LeicaImageInfo {
            name: image_node.attr("name").map(ToOwned::to_owned),
            ifd_levels,
            channels: parse_channel_settings(image_node),
            view_size,
            view_offset,
            creation_date,
            device_model,
            device_version,
            objective,
            numerical_aperture,
            illumination_source,
            is_macro,
        }))
    }

    fn associated_source_for_ifd(
        &self,
        container: &TiffContainer,
        ifd_id: IfdId,
    ) -> Result<TileSource, TiffParseError> {
        let compression = container
            .get_u32(ifd_id, tags::COMPRESSION)
            .map(compression_from_tag)
            .unwrap_or(Compression::Jpeg);
        if container.get_u32(ifd_id, tags::TILE_WIDTH).is_ok() {
            Ok(TileSource::TiledIfd {
                ifd_id,
                jpeg_tables: container
                    .get_bytes(ifd_id, tags::JPEG_TABLES)
                    .ok()
                    .map(|bytes| bytes.to_vec()),
                compression,
            })
        } else {
            Ok(TileSource::Stripped {
                ifd_id,
                jpeg_tables: container
                    .get_bytes(ifd_id, tags::JPEG_TABLES)
                    .ok()
                    .map(|bytes| bytes.to_vec()),
                compression,
                strip_offsets: container
                    .get_u64_array(ifd_id, tags::STRIP_OFFSETS)
                    .map(|values| values.to_vec())
                    .unwrap_or_default(),
                strip_byte_counts: container
                    .get_u64_array(ifd_id, tags::STRIP_BYTE_COUNTS)
                    .map(|values| values.to_vec())
                    .unwrap_or_default(),
            })
        }
    }

    fn parse_public_properties(
        &self,
        collection: &xml::XmlNode,
        main_images: &[LeicaImageInfo],
        level0_cpp_x: f64,
        level0_cpp_y: f64,
    ) -> Result<Properties, TiffParseError> {
        let representative = main_images.first().ok_or_else(|| {
            TiffParseError::Structure("Leica SCN: no main images for property parsing".into())
        })?;
        let mut properties = Properties::new();
        properties.insert("openslide.vendor", "leica");

        if let Some(barcode) = collection
            .find("barcode")
            .and_then(|node| node.text.as_ref())
        {
            let decoded = BASE64_STANDARD
                .decode(barcode)
                .ok()
                .and_then(|bytes| String::from_utf8(bytes).ok())
                .unwrap_or_else(|| barcode.clone());
            properties.insert("leica.barcode", decoded);
        }
        if let Some(value) = representative.creation_date.as_deref() {
            properties.insert("leica.creation-date", value);
        }
        if let Some(value) = representative.device_model.as_deref() {
            properties.insert("leica.device-model", value);
        }
        if let Some(value) = representative.device_version.as_deref() {
            properties.insert("leica.device-version", value);
        }
        if let Some(value) = representative.illumination_source.as_deref() {
            properties.insert("leica.illumination-source", value);
        }
        if let Some(value) = representative.numerical_aperture.as_deref() {
            properties.insert("leica.aperture", value);
        }
        if let Some(value) = representative.objective.as_deref() {
            properties.insert("leica.objective", value);
            properties.insert("openslide.objective-power", value);
        }
        properties.insert("openslide.mpp-x", (level0_cpp_x / 1000.0).to_string());
        properties.insert("openslide.mpp-y", (level0_cpp_y / 1000.0).to_string());

        let mut bounds_x = i64::MAX;
        let mut bounds_y = i64::MAX;
        let mut bounds_right = i64::MIN;
        let mut bounds_bottom = i64::MIN;
        for (idx, image) in main_images.iter().enumerate() {
            let level0 = image
                .ifd_levels
                .iter()
                .find(|level| level.z == 0 && level.c == 0 && level.r == 0)
                .or_else(|| image.level0_dimension())
                .ok_or_else(|| {
                    TiffParseError::Structure("Leica SCN image missing level 0 bounds".into())
                })?;
            let x = (image.view_offset.0 as f64 / level0_cpp_x).floor() as i64;
            let y = (image.view_offset.1 as f64 / level0_cpp_y).floor() as i64;
            let w = i64::try_from(level0.width).unwrap_or(i64::MAX);
            let h = i64::try_from(level0.height).unwrap_or(i64::MAX);
            properties.insert(format!("openslide.region[{idx}].x"), x.to_string());
            properties.insert(format!("openslide.region[{idx}].y"), y.to_string());
            properties.insert(format!("openslide.region[{idx}].width"), w.to_string());
            properties.insert(format!("openslide.region[{idx}].height"), h.to_string());
            bounds_x = bounds_x.min(x);
            bounds_y = bounds_y.min(y);
            bounds_right = bounds_right.max(x + w);
            bounds_bottom = bounds_bottom.max(y + h);
        }
        if bounds_x != i64::MAX && bounds_y != i64::MAX {
            properties.insert("openslide.bounds-x", bounds_x.to_string());
            properties.insert("openslide.bounds-y", bounds_y.to_string());
            properties.insert(
                "openslide.bounds-width",
                (bounds_right - bounds_x).to_string(),
            );
            properties.insert(
                "openslide.bounds-height",
                (bounds_bottom - bounds_y).to_string(),
            );
        }

        Ok(properties)
    }
}

impl LeicaImageInfo {
    fn level0_dimension(&self) -> Option<&ParsedDimension> {
        self.ifd_levels
            .iter()
            .find(|level| level.r == 0 && level.z == 0 && level.c == 0)
            .or_else(|| self.ifd_levels.iter().find(|level| level.r == 0))
            .or_else(|| self.ifd_levels.iter().min_by_key(|level| level.r))
    }

    fn axes_shape(&self) -> AxesShape {
        let z = self
            .ifd_levels
            .iter()
            .map(|level| level.z)
            .max()
            .unwrap_or(0)
            + 1;
        let c = self
            .ifd_levels
            .iter()
            .map(|level| level.c)
            .max()
            .unwrap_or(0)
            + 1;
        AxesShape { z, c, t: 1 }
    }

    fn channels_for_axes(&self, axes: AxesShape) -> Vec<ChannelInfo> {
        if axes.c == 1
            && self.channels.is_empty()
            && self.illumination_source.as_deref() == Some("brightfield")
        {
            return Vec::new();
        }

        let mut channels = (0..axes.c)
            .map(|idx| ChannelInfo {
                name: Some(format!("Channel {idx}")),
                color: None,
                excitation_nm: None,
                emission_nm: None,
            })
            .collect::<Vec<_>>();
        for channel in &self.channels {
            if let Some(slot) = channels.get_mut(channel.index as usize) {
                *slot = channel.info.clone();
            }
        }
        if axes.c == 1 && self.channels.is_empty() {
            if let Some(channel) = channels.first_mut() {
                channel.name = self
                    .name
                    .clone()
                    .or_else(|| self.illumination_source.clone())
                    .or_else(|| Some("Channel 0".into()));
            }
        }
        channels
    }
}

fn lookup_ifd(top_ifds: &[IfdId], ifd_index: usize) -> Result<IfdId, TiffParseError> {
    top_ifds.get(ifd_index).copied().ok_or_else(|| {
        TiffParseError::Structure(format!(
            "Leica SCN references out-of-range IFD index {}",
            ifd_index
        ))
    })
}

fn unique_associated_name(
    associated_images: &HashMap<String, AssociatedImage>,
    base: &str,
) -> String {
    if !associated_images.contains_key(base) {
        return base.into();
    }
    for index in 1.. {
        let candidate = format!("{base}-{index}");
        if !associated_images.contains_key(&candidate) {
            return candidate;
        }
    }
    unreachable!("unbounded suffix search must eventually return")
}

fn sample_type_for_ifd(container: &TiffContainer, ifd_id: IfdId) -> SampleType {
    let bits_per_sample = container
        .get_u32(ifd_id, tags::BITS_PER_SAMPLE)
        .unwrap_or(8);
    let sample_format = container.get_u32(ifd_id, 339).unwrap_or(1);
    match (bits_per_sample, sample_format) {
        (16, _) => SampleType::Uint16,
        (32, 3) => SampleType::Float32,
        _ => SampleType::Uint8,
    }
}

fn associated_channel_count(container: &TiffContainer, ifd_id: IfdId) -> u16 {
    container
        .get_u32(ifd_id, tags::SAMPLES_PER_PIXEL)
        .ok()
        .and_then(|value| u16::try_from(value).ok())
        .unwrap_or(3)
}

fn parse_channel_settings(image_node: &xml::XmlNode) -> Vec<ParsedChannel> {
    let Some(channel_settings) = image_node
        .find("scanSettings")
        .and_then(|scan| scan.find("channelSettings"))
    else {
        return Vec::new();
    };

    channel_settings
        .find_all("channel")
        .into_iter()
        .filter_map(|channel| {
            let index = channel.attr("index")?.parse::<u32>().ok()?;
            Some(ParsedChannel {
                index,
                info: ChannelInfo {
                    name: channel.attr("name").map(ToOwned::to_owned),
                    color: channel.attr("rgb").and_then(parse_rgb_hex),
                    excitation_nm: None,
                    emission_nm: None,
                },
            })
        })
        .collect()
}

fn parse_rgb_hex(value: &str) -> Option<[u8; 3]> {
    let value = value.strip_prefix('#').unwrap_or(value);
    if value.len() != 6 {
        return None;
    }
    Some([
        u8::from_str_radix(&value[0..2], 16).ok()?,
        u8::from_str_radix(&value[2..4], 16).ok()?,
        u8::from_str_radix(&value[4..6], 16).ok()?,
    ])
}

fn parse_required_u64_attr(
    node: &xml::XmlNode,
    attr: &str,
    context: &str,
) -> Result<u64, TiffParseError> {
    let value = node.attr(attr).ok_or_else(|| {
        TiffParseError::Structure(format!("{context}: missing required attribute '{attr}'"))
    })?;
    value.parse::<u64>().map_err(|_| {
        TiffParseError::Structure(format!(
            "{context}: invalid integer '{}' for attribute '{}'",
            value, attr
        ))
    })
}

fn parse_optional_i64_attr(node: &xml::XmlNode, attr: &str) -> Option<i64> {
    node.attr(attr).and_then(|value| value.parse::<i64>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::registry::Slide;
    use std::path::Path;

    #[test]
    fn detect_scn_with_namespace() {
        let xml = format!(
            r#"<scn xmlns="{}"><collection sizeX="1000" sizeY="1000"></collection></scn>"#,
            LEICA_NS_2010_10
        );
        let root = xml::parse_xml(&xml).unwrap();
        assert_eq!(root.find("collection").unwrap().attr("sizeX"), Some("1000"));
    }

    #[test]
    fn parse_dimension_extracts_resolution_index() {
        let interp = LeicaInterpreter;
        let node = xml::XmlNode {
            tag: "dimension".into(),
            attributes: HashMap::from([
                ("ifd".into(), "2".into()),
                ("sizeX".into(), "4096".into()),
                ("sizeY".into(), "3072".into()),
                ("r".into(), "3".into()),
                ("z".into(), "0".into()),
                ("c".into(), "2".into()),
            ]),
            text: None,
            children: vec![],
        };

        let parsed = interp.parse_dimension(&node).unwrap().unwrap();
        assert_eq!(parsed.ifd_index, 2);
        assert_eq!(parsed.width, 4096);
        assert_eq!(parsed.height, 3072);
        assert_eq!(parsed.r, 3);
        assert_eq!(parsed.z, 0);
        assert_eq!(parsed.c, 2);
    }

    #[test]
    fn parse_channel_settings_extracts_names_and_colors() {
        let image = xml::XmlNode {
            tag: "image".into(),
            attributes: HashMap::new(),
            text: None,
            children: vec![xml::XmlNode {
                tag: "scanSettings".into(),
                attributes: HashMap::new(),
                text: None,
                children: vec![xml::XmlNode {
                    tag: "channelSettings".into(),
                    attributes: HashMap::new(),
                    text: None,
                    children: vec![xml::XmlNode {
                        tag: "channel".into(),
                        attributes: HashMap::from([
                            ("index".into(), "2".into()),
                            ("name".into(), "TX2|Empty".into()),
                            ("rgb".into(), "#ff0000".into()),
                        ]),
                        text: None,
                        children: vec![],
                    }],
                }],
            }],
        };

        let channels = parse_channel_settings(&image);
        assert_eq!(channels.len(), 1);
        assert_eq!(channels[0].index, 2);
        assert_eq!(channels[0].info.name.as_deref(), Some("TX2|Empty"));
        assert_eq!(channels[0].info.color, Some([255, 0, 0]));
    }

    #[test]
    fn macro_detection_uses_collection_geometry() {
        let interp = LeicaInterpreter;
        let image = xml::XmlNode {
            tag: "image".into(),
            attributes: HashMap::new(),
            text: None,
            children: vec![
                xml::XmlNode {
                    tag: "pixels".into(),
                    attributes: HashMap::from([
                        ("sizeX".into(), "100".into()),
                        ("sizeY".into(), "200".into()),
                    ]),
                    text: None,
                    children: vec![xml::XmlNode {
                        tag: "dimension".into(),
                        attributes: HashMap::from([
                            ("ifd".into(), "0".into()),
                            ("sizeX".into(), "100".into()),
                            ("sizeY".into(), "200".into()),
                            ("r".into(), "0".into()),
                        ]),
                        text: None,
                        children: vec![],
                    }],
                },
                xml::XmlNode {
                    tag: "view".into(),
                    attributes: HashMap::from([
                        ("sizeX".into(), "1000".into()),
                        ("sizeY".into(), "2000".into()),
                        ("offsetX".into(), "0".into()),
                        ("offsetY".into(), "0".into()),
                    ]),
                    text: None,
                    children: vec![],
                },
                xml::XmlNode {
                    tag: "scanSettings".into(),
                    attributes: HashMap::new(),
                    text: None,
                    children: vec![xml::XmlNode {
                        tag: "illuminationSettings".into(),
                        attributes: HashMap::new(),
                        text: None,
                        children: vec![xml::XmlNode {
                            tag: "illuminationSource".into(),
                            attributes: HashMap::new(),
                            text: Some("brightfield".into()),
                            children: vec![],
                        }],
                    }],
                },
            ],
        };

        let parsed = interp
            .parse_image_info(&image, (1000, 2000))
            .unwrap()
            .unwrap();
        assert!(parsed.is_macro);
    }

    #[test]
    fn public_properties_use_axis_specific_cpp() {
        let interp = LeicaInterpreter;
        let collection = xml::XmlNode {
            tag: "collection".into(),
            attributes: HashMap::new(),
            text: None,
            children: vec![],
        };
        let main_images = vec![LeicaImageInfo {
            name: Some("main".into()),
            ifd_levels: vec![ParsedDimension {
                ifd_index: 0,
                width: 100,
                height: 50,
                r: 0,
                z: 0,
                c: 0,
            }],
            channels: vec![],
            view_size: (1000, 600),
            view_offset: (200, 150),
            creation_date: None,
            device_model: None,
            device_version: None,
            objective: None,
            numerical_aperture: None,
            illumination_source: Some("brightfield".into()),
            is_macro: false,
        }];

        let props = interp
            .parse_public_properties(&collection, &main_images, 10.0, 12.0)
            .unwrap();
        assert_eq!(props.get("openslide.mpp-x"), Some("0.01"));
        assert_eq!(props.get("openslide.mpp-y"), Some("0.012"));
    }

    #[test]
    fn public_geometry_uses_collection_nm_per_pixel_for_both_axes() {
        let _main_images = [LeicaImageInfo {
            name: Some("main".into()),
            ifd_levels: vec![
                ParsedDimension {
                    ifd_index: 0,
                    width: 36832,
                    height: 38432,
                    r: 0,
                    z: 0,
                    c: 0,
                },
                ParsedDimension {
                    ifd_index: 1,
                    width: 9208,
                    height: 9608,
                    r: 1,
                    z: 0,
                    c: 0,
                },
                ParsedDimension {
                    ifd_index: 2,
                    width: 2302,
                    height: 2402,
                    r: 2,
                    z: 0,
                    c: 0,
                },
                ParsedDimension {
                    ifd_index: 3,
                    width: 576,
                    height: 600,
                    r: 3,
                    z: 0,
                    c: 0,
                },
                ParsedDimension {
                    ifd_index: 4,
                    width: 144,
                    height: 150,
                    r: 4,
                    z: 0,
                    c: 0,
                },
            ],
            channels: vec![],
            view_size: (18416000, 19217000),
            view_offset: (5389341, 17548313),
            creation_date: None,
            device_model: None,
            device_version: None,
            objective: None,
            numerical_aperture: None,
            illumination_source: Some("brightfield".into()),
            is_macro: false,
        }];
        let level0_cpp_x = 18416000.0 / 36832.0;
        let level0_cpp_y = 19217000.0 / 38432.0;
        let level3_cpp: f64 = 18416000.0 / 576.0;
        let width = (26564529.0_f64 / level3_cpp).ceil() as u64;
        let height = (76734666.0_f64 / level3_cpp).ceil() as u64;
        assert_eq!(width, 831);
        assert_eq!(height, 2401);
        assert!(level0_cpp_y > level0_cpp_x);
    }

    #[test]
    fn opens_dissimilar_leica_scenes_when_corpus_is_available() {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
        let path = workspace_root.join("downloads/openslide-testdata/Leica/Leica-3.scn");
        if !path.exists() {
            return;
        }

        let slide = Slide::open(&path).expect("open Leica multi-scene SCN");
        let dataset = slide.dataset();
        assert!(dataset.scenes.len() > 1);
        assert!(dataset.associated_images.contains_key("macro"));
        let first_series = &dataset.scenes[0].series[0];
        assert_eq!(first_series.axes, AxesShape::default());
        assert!(!first_series.levels.is_empty());
    }

    #[test]
    fn opens_fluorescence_leica_channels_when_corpus_is_available() {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
        let path =
            workspace_root.join("downloads/openslide-testdata/Leica/Leica-Fluorescence-1.scn");
        if !path.exists() {
            return;
        }

        let slide = Slide::open(&path).expect("open Leica fluorescence SCN");
        let dataset = slide.dataset();
        let series = &dataset.scenes[0].series[0];
        assert_eq!(series.axes.c, 3);
        assert_eq!(series.channels.len(), 3);
        assert_eq!(series.channels[0].color, Some([0, 0, 255]));
        for c in 0..series.axes.c {
            let tile = slide
                .read_tile(
                    &TileRequest {
                        scene: 0,
                        series: 0,
                        level: 0,
                        plane: PlaneSelection { z: 0, c, t: 0 },
                        col: 0,
                        row: 0,
                    },
                    TileOutputPreference::cpu(),
                )
                .expect("read fluorescence channel tile");
            assert!(matches!(tile, TilePixels::Cpu(_)));
        }
    }
}
