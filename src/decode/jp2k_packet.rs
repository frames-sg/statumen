use crate::decode::jp2k_codestream::{
    Jp2kCodestreamInfo, Jp2kProgressionOrder, Jp2kQuantizationStyle, Jp2kTilePartInfo,
};
use crate::error::WsiError;

const DEFAULT_PRECINCT_EXPONENT: u8 = 15;
const DEFAULT_SEGMENT_MAX_PASSES: u32 = 109;
const MARKER_SOP: [u8; 2] = [0xFF, 0x91];
const MARKER_EPH: [u8; 2] = [0xFF, 0x92];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Jp2kPacketCoordinate {
    pub layer: u16,
    pub resolution: u8,
    pub component: u16,
    pub precinct: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Jp2kCodeBlockContribution {
    pub component: u16,
    pub resolution: u8,
    pub band: u8,
    pub precinct: u32,
    pub code_block: u32,
    pub zero_bit_planes: u32,
    pub pass_count: u32,
    pub length: usize,
    pub body_offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Jp2kPacket {
    pub coordinate: Jp2kPacketCoordinate,
    pub header_offset: usize,
    pub header_length: usize,
    pub body_offset: usize,
    pub body_length: usize,
    pub contributions: Vec<Jp2kCodeBlockContribution>,
}

#[derive(Debug, Clone)]
struct TileLayout {
    components: Vec<ComponentLayout>,
}

#[derive(Debug, Clone)]
struct ComponentLayout {
    resolutions: Vec<ResolutionLayout>,
}

#[derive(Debug, Clone)]
struct ResolutionLayout {
    precinct_width: u32,
    precinct_height: u32,
    bands: Vec<BandLayout>,
}

#[derive(Debug, Clone)]
struct BandLayout {
    bandno: u8,
    precincts: Vec<PrecinctLayout>,
}

#[derive(Debug, Clone)]
struct PrecinctLayout {
    code_block_width: u32,
    code_block_height: u32,
    code_block_count: usize,
}

#[derive(Debug, Clone)]
struct PacketParserState {
    components: Vec<ComponentState>,
}

#[derive(Debug, Clone)]
struct ComponentState {
    resolutions: Vec<ResolutionState>,
}

#[derive(Debug, Clone)]
struct ResolutionState {
    bands: Vec<BandState>,
}

#[derive(Debug, Clone)]
struct BandState {
    precincts: Vec<PrecinctState>,
}

#[derive(Debug, Clone)]
struct PrecinctState {
    inclusion_tree: TagTree,
    zero_bit_plane_tree: TagTree,
    code_blocks: Vec<CodeBlockState>,
}

#[derive(Debug, Clone, Default)]
struct CodeBlockState {
    included_before: bool,
    num_passes: u32,
    num_len_bits: u32,
    zero_bit_planes: u32,
}

#[derive(Debug, Clone)]
struct TagTree {
    nodes: Vec<TagTreeNode>,
}

#[derive(Debug, Clone)]
struct TagTreeNode {
    parent: Option<usize>,
    value: i32,
    low: i32,
}

struct PacketHeaderReader<'a> {
    data: &'a [u8],
    offset: usize,
    buf: u32,
    bit_count: u32,
}

impl<'a> PacketHeaderReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            offset: 0,
            buf: 0,
            bit_count: 0,
        }
    }

    fn read_bit(&mut self) -> Result<bool, WsiError> {
        if self.bit_count == 0 {
            self.byte_in()?;
        }
        self.bit_count -= 1;
        Ok(((self.buf >> self.bit_count) & 1) != 0)
    }

    fn read_bits(&mut self, bit_count: u32) -> Result<u32, WsiError> {
        if bit_count == 0 || bit_count > 32 {
            return Err(WsiError::Jp2k(format!(
                "invalid JP2K packet bit count: {}",
                bit_count
            )));
        }

        let mut value = 0u32;
        for shift in (0..bit_count).rev() {
            if self.read_bit()? {
                value |= 1 << shift;
            }
        }
        Ok(value)
    }

    fn align(&mut self) -> Result<(), WsiError> {
        if (self.buf & 0xFF) == 0xFF {
            self.byte_in()?;
        }
        self.bit_count = 0;
        Ok(())
    }

    fn bytes_read(&self) -> usize {
        self.offset
    }

    fn byte_in(&mut self) -> Result<(), WsiError> {
        self.buf = (self.buf << 8) & 0xFFFF;
        self.bit_count = if self.buf == 0xFF00 { 7 } else { 8 };
        let next = self
            .data
            .get(self.offset)
            .copied()
            .ok_or_else(|| WsiError::Jp2k("truncated JP2K packet header".into()))?;
        self.buf |= next as u32;
        self.offset += 1;
        Ok(())
    }
}

impl TagTree {
    fn new(width: usize, height: usize) -> Self {
        let mut level_widths = Vec::new();
        let mut level_heights = Vec::new();
        let mut w = width.max(1);
        let mut h = height.max(1);
        loop {
            level_widths.push(w);
            level_heights.push(h);
            if w == 1 && h == 1 {
                break;
            }
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }

        let mut level_offsets = Vec::with_capacity(level_widths.len());
        let mut node_count = 0usize;
        for (&level_width, &level_height) in level_widths.iter().zip(level_heights.iter()) {
            level_offsets.push(node_count);
            node_count += level_width * level_height;
        }

        let mut nodes = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            nodes.push(TagTreeNode {
                parent: None,
                value: i32::MAX / 4,
                low: 0,
            });
        }

        for level in 0..(level_widths.len().saturating_sub(1)) {
            let width = level_widths[level];
            let parent_width = level_widths[level + 1];
            let level_offset = level_offsets[level];
            let parent_offset = level_offsets[level + 1];
            for y in 0..level_heights[level] {
                for x in 0..width {
                    let index = level_offset + y * width + x;
                    let parent_index = parent_offset + (y / 2) * parent_width + (x / 2);
                    nodes[index].parent = Some(parent_index);
                }
            }
        }

        Self { nodes }
    }

    fn decode(
        &mut self,
        leaf_index: usize,
        threshold: i32,
        reader: &mut PacketHeaderReader<'_>,
    ) -> Result<bool, WsiError> {
        let mut stack = Vec::new();
        let mut node_index = leaf_index;
        while let Some(parent) = self.nodes[node_index].parent {
            stack.push(node_index);
            node_index = parent;
        }

        let mut low = 0i32;
        loop {
            if low > self.nodes[node_index].low {
                self.nodes[node_index].low = low;
            } else {
                low = self.nodes[node_index].low;
            }

            while low < threshold && low < self.nodes[node_index].value {
                if reader.read_bit()? {
                    self.nodes[node_index].value = low;
                } else {
                    low += 1;
                }
            }

            self.nodes[node_index].low = low;
            if let Some(child) = stack.pop() {
                node_index = child;
            } else {
                break;
            }
        }

        Ok(self.nodes[node_index].value < threshold)
    }
}

impl TileLayout {
    fn from_codestream(info: &Jp2kCodestreamInfo) -> Result<Self, WsiError> {
        if info.coding_style.custom_precincts {
            return Err(WsiError::Jp2k(
                "JP2K packet parser only supports default precinct partitions".into(),
            ));
        }
        if info.coding_style.code_block_style != 0 {
            return Err(WsiError::Jp2k(
                "JP2K packet parser only supports default code-block style".into(),
            ));
        }
        if info.quantization.style != Jp2kQuantizationStyle::ScalarExpounded {
            return Err(WsiError::Jp2k(
                "JP2K packet parser currently expects scalar-expounded quantization".into(),
            ));
        }

        let resolution_count = info.coding_style.resolution_count() as usize;
        let code_block_width_exponent = info.coding_style.code_block_width_exponent + 2;
        let code_block_height_exponent = info.coding_style.code_block_height_exponent + 2;
        let tile_x0 = info.image_origin_x.max(info.tile_origin_x) as i32;
        let tile_y0 = info.image_origin_y.max(info.tile_origin_y) as i32;
        let image_x1 = info
            .image_origin_x
            .checked_add(info.image_width)
            .ok_or_else(|| WsiError::Jp2k("JP2K image X bounds overflow".into()))?;
        let image_y1 = info
            .image_origin_y
            .checked_add(info.image_height)
            .ok_or_else(|| WsiError::Jp2k("JP2K image Y bounds overflow".into()))?;
        let tile_x1 = info
            .tile_origin_x
            .checked_add(info.tile_width)
            .ok_or_else(|| WsiError::Jp2k("JP2K tile X bounds overflow".into()))?
            .min(image_x1) as i32;
        let tile_y1 = info
            .tile_origin_y
            .checked_add(info.tile_height)
            .ok_or_else(|| WsiError::Jp2k("JP2K tile Y bounds overflow".into()))?
            .min(image_y1) as i32;

        let mut components = Vec::with_capacity(info.components.len());
        for component in &info.components {
            let mut step_index = 0usize;
            let tile_component_x0 =
                ceil_div(tile_x0, component.horizontal_sample_separation as i32);
            let tile_component_y0 = ceil_div(tile_y0, component.vertical_sample_separation as i32);
            let tile_component_x1 =
                ceil_div(tile_x1, component.horizontal_sample_separation as i32);
            let tile_component_y1 = ceil_div(tile_y1, component.vertical_sample_separation as i32);
            let mut resolutions = Vec::with_capacity(resolution_count);
            let mut level_no = resolution_count as i32;
            for resolution in 0..resolution_count {
                level_no -= 1;
                let resolution_x0 = ceil_div_pow2(tile_component_x0, level_no);
                let resolution_y0 = ceil_div_pow2(tile_component_y0, level_no);
                let resolution_x1 = ceil_div_pow2(tile_component_x1, level_no);
                let resolution_y1 = ceil_div_pow2(tile_component_y1, level_no);

                let precinct_exponent_x = DEFAULT_PRECINCT_EXPONENT;
                let precinct_exponent_y = DEFAULT_PRECINCT_EXPONENT;
                let precinct_x0 = floor_div_pow2(resolution_x0, precinct_exponent_x as i32)
                    << precinct_exponent_x;
                let precinct_y0 = floor_div_pow2(resolution_y0, precinct_exponent_y as i32)
                    << precinct_exponent_y;
                let precinct_x1 =
                    ceil_div_pow2(resolution_x1, precinct_exponent_x as i32) << precinct_exponent_x;
                let precinct_y1 =
                    ceil_div_pow2(resolution_y1, precinct_exponent_y as i32) << precinct_exponent_y;
                let precinct_width = if resolution_x0 == resolution_x1 {
                    0
                } else {
                    ((precinct_x1 - precinct_x0) >> precinct_exponent_x) as u32
                };
                let precinct_height = if resolution_y0 == resolution_y1 {
                    0
                } else {
                    ((precinct_y1 - precinct_y0) >> precinct_exponent_y) as u32
                };

                let (
                    code_block_group_x0,
                    code_block_group_y0,
                    code_block_group_width_exponent,
                    code_block_group_height_exponent,
                    band_count,
                ) = if resolution == 0 {
                    (
                        precinct_x0,
                        precinct_y0,
                        precinct_exponent_x,
                        precinct_exponent_y,
                        1usize,
                    )
                } else {
                    (
                        ceil_div_pow2(precinct_x0, 1),
                        ceil_div_pow2(precinct_y0, 1),
                        precinct_exponent_x - 1,
                        precinct_exponent_y - 1,
                        3usize,
                    )
                };

                let code_block_width =
                    code_block_width_exponent.min(code_block_group_width_exponent);
                let code_block_height =
                    code_block_height_exponent.min(code_block_group_height_exponent);
                let mut bands = Vec::with_capacity(band_count);
                for band_index in 0..band_count {
                    let (bandno, band_x0, band_y0, band_x1, band_y1) = if resolution == 0 {
                        (
                            0u8,
                            ceil_div_pow2(tile_component_x0, level_no),
                            ceil_div_pow2(tile_component_y0, level_no),
                            ceil_div_pow2(tile_component_x1, level_no),
                            ceil_div_pow2(tile_component_y1, level_no),
                        )
                    } else {
                        let bandno = (band_index + 1) as u8;
                        let x0b = (bandno & 1) as i64;
                        let y0b = (bandno >> 1) as i64;
                        (
                            bandno,
                            ceil_div_pow2_i64(
                                tile_component_x0 as i64 - (x0b << level_no),
                                level_no + 1,
                            ),
                            ceil_div_pow2_i64(
                                tile_component_y0 as i64 - (y0b << level_no),
                                level_no + 1,
                            ),
                            ceil_div_pow2_i64(
                                tile_component_x1 as i64 - (x0b << level_no),
                                level_no + 1,
                            ),
                            ceil_div_pow2_i64(
                                tile_component_y1 as i64 - (y0b << level_no),
                                level_no + 1,
                            ),
                        )
                    };

                    info.quantization.steps.get(step_index).ok_or_else(|| {
                        WsiError::Jp2k("JP2K quantization step table underflow".into())
                    })?;
                    step_index += 1;

                    let mut precincts =
                        Vec::with_capacity((precinct_width * precinct_height) as usize);
                    for precinct in 0..(precinct_width * precinct_height) {
                        let precinct_x = code_block_group_x0
                            + (precinct % precinct_width) as i32
                                * (1i32 << code_block_group_width_exponent);
                        let precinct_y = code_block_group_y0
                            + (precinct / precinct_width) as i32
                                * (1i32 << code_block_group_height_exponent);
                        let precinct_x_end = precinct_x + (1i32 << code_block_group_width_exponent);
                        let precinct_y_end =
                            precinct_y + (1i32 << code_block_group_height_exponent);
                        let clipped_x0 = precinct_x.max(band_x0);
                        let clipped_y0 = precinct_y.max(band_y0);
                        let clipped_x1 = precinct_x_end.min(band_x1);
                        let clipped_y1 = precinct_y_end.min(band_y1);
                        let code_block_x0 =
                            floor_div_pow2(clipped_x0, code_block_width as i32) << code_block_width;
                        let code_block_y0 = floor_div_pow2(clipped_y0, code_block_height as i32)
                            << code_block_height;
                        let code_block_x1 =
                            ceil_div_pow2(clipped_x1, code_block_width as i32) << code_block_width;
                        let code_block_y1 = ceil_div_pow2(clipped_y1, code_block_height as i32)
                            << code_block_height;
                        let code_blocks_w =
                            ((code_block_x1 - code_block_x0) >> code_block_width) as u32;
                        let code_blocks_h =
                            ((code_block_y1 - code_block_y0) >> code_block_height) as u32;
                        precincts.push(PrecinctLayout {
                            code_block_width: code_blocks_w,
                            code_block_height: code_blocks_h,
                            code_block_count: (code_blocks_w * code_blocks_h) as usize,
                        });
                    }

                    bands.push(BandLayout { bandno, precincts });
                }

                resolutions.push(ResolutionLayout {
                    precinct_width,
                    precinct_height,
                    bands,
                });
            }
            components.push(ComponentLayout { resolutions });
        }

        Ok(Self { components })
    }

    fn precinct_count(&self, component: usize, resolution: usize) -> u32 {
        let resolution = &self.components[component].resolutions[resolution];
        resolution.precinct_width * resolution.precinct_height
    }
}

impl PacketParserState {
    fn new(layout: &TileLayout) -> Self {
        let components = layout
            .components
            .iter()
            .map(|component| ComponentState {
                resolutions: component
                    .resolutions
                    .iter()
                    .map(|resolution| ResolutionState {
                        bands: resolution
                            .bands
                            .iter()
                            .map(|band| BandState {
                                precincts: band
                                    .precincts
                                    .iter()
                                    .map(|precinct| PrecinctState {
                                        inclusion_tree: TagTree::new(
                                            precinct.code_block_width.max(1) as usize,
                                            precinct.code_block_height.max(1) as usize,
                                        ),
                                        zero_bit_plane_tree: TagTree::new(
                                            precinct.code_block_width.max(1) as usize,
                                            precinct.code_block_height.max(1) as usize,
                                        ),
                                        code_blocks: vec![
                                            CodeBlockState::default();
                                            precinct.code_block_count
                                        ],
                                    })
                                    .collect(),
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect();

        Self { components }
    }
}

#[cfg(test)]
pub(crate) fn enumerate_packet_order(info: &Jp2kCodestreamInfo) -> Vec<Jp2kPacketCoordinate> {
    let Ok(layout) = TileLayout::from_codestream(info) else {
        return Vec::new();
    };
    enumerate_packet_order_with_layout(info, &layout)
}

#[cfg(test)]
pub(crate) fn tile_part_packet_coordinates(
    info: &Jp2kCodestreamInfo,
    tile_part: &Jp2kTilePartInfo,
) -> Vec<Jp2kPacketCoordinate> {
    let Ok(layout) = TileLayout::from_codestream(info) else {
        return Vec::new();
    };
    tile_part_packet_coordinates_with_layout(info, tile_part, &layout)
}

pub(crate) fn parse_tile_part_packets(
    data: &[u8],
    info: &Jp2kCodestreamInfo,
    tile_part: &Jp2kTilePartInfo,
) -> Result<Vec<Jp2kPacket>, WsiError> {
    let layout = TileLayout::from_codestream(info)?;
    let packet_coordinates = tile_part_packet_coordinates_with_layout(info, tile_part, &layout);
    let tile_part_end = tile_part
        .data_offset
        .checked_add(tile_part.data_length)
        .ok_or_else(|| WsiError::Jp2k("JP2K tile-part data range overflow".into()))?;
    let tile_part_data = data
        .get(tile_part.data_offset..tile_part_end)
        .ok_or_else(|| WsiError::Jp2k("JP2K tile-part payload range out of bounds".into()))?;

    let mut packets = Vec::with_capacity(packet_coordinates.len());
    let mut state = PacketParserState::new(&layout);
    let mut offset = 0usize;
    for coordinate in packet_coordinates {
        let packet_data = tile_part_data
            .get(offset..)
            .ok_or_else(|| WsiError::Jp2k("JP2K packet offset out of bounds".into()))?;
        let mut packet_header_prefix = 0usize;
        if info.coding_style.sop_markers && packet_data.starts_with(&MARKER_SOP) {
            if packet_data.len() < 6 {
                return Err(WsiError::Jp2k("truncated JP2K SOP marker".into()));
            }
            packet_header_prefix = 6;
        }

        let packet_payload = &packet_data[packet_header_prefix..];
        let (header_length, body_length, mut contributions) =
            parse_packet_payload(packet_payload, &coordinate, info, &layout, &mut state)?;
        let body_offset = offset + packet_header_prefix + header_length;
        let body_end = body_offset
            .checked_add(body_length)
            .ok_or_else(|| WsiError::Jp2k("JP2K packet body range overflow".into()))?;
        if body_end > tile_part_data.len() {
            return Err(WsiError::Jp2k("truncated JP2K packet body".into()));
        }

        let mut contribution_body_offset = tile_part.data_offset + body_offset;
        for contribution in &mut contributions {
            contribution.body_offset = contribution_body_offset;
            contribution_body_offset += contribution.length;
        }

        packets.push(Jp2kPacket {
            coordinate,
            header_offset: tile_part.data_offset + offset,
            header_length: packet_header_prefix + header_length,
            body_offset: tile_part.data_offset + body_offset,
            body_length,
            contributions,
        });
        offset = body_end;
    }

    Ok(packets)
}

fn enumerate_packet_order_with_layout(
    info: &Jp2kCodestreamInfo,
    layout: &TileLayout,
) -> Vec<Jp2kPacketCoordinate> {
    let layers = info.coding_style.layers;
    let resolution_count = info.coding_style.resolution_count();
    let component_count = info.components.len() as u16;
    let max_precincts = (0..component_count as usize)
        .flat_map(|component| {
            (0..resolution_count as usize)
                .map(move |resolution| layout.precinct_count(component, resolution))
        })
        .max()
        .unwrap_or(0);
    let mut packets = Vec::new();

    let mut push_if_present = |layer: u16, resolution: u8, component: u16, precinct: u32| {
        if precinct < layout.precinct_count(component as usize, resolution as usize) {
            packets.push(Jp2kPacketCoordinate {
                layer,
                resolution,
                component,
                precinct,
            });
        }
    };

    match info.coding_style.progression_order {
        Jp2kProgressionOrder::Lrcp => {
            for layer in 0..layers {
                for resolution in 0..resolution_count {
                    for component in 0..component_count {
                        let precinct_count =
                            layout.precinct_count(component as usize, resolution as usize);
                        for precinct in 0..precinct_count {
                            push_if_present(layer, resolution, component, precinct);
                        }
                    }
                }
            }
        }
        Jp2kProgressionOrder::Rlcp => {
            for resolution in 0..resolution_count {
                for layer in 0..layers {
                    for component in 0..component_count {
                        let precinct_count =
                            layout.precinct_count(component as usize, resolution as usize);
                        for precinct in 0..precinct_count {
                            push_if_present(layer, resolution, component, precinct);
                        }
                    }
                }
            }
        }
        Jp2kProgressionOrder::Rpcl => {
            for resolution in 0..resolution_count {
                for precinct in 0..max_precincts {
                    for component in 0..component_count {
                        for layer in 0..layers {
                            push_if_present(layer, resolution, component, precinct);
                        }
                    }
                }
            }
        }
        Jp2kProgressionOrder::Pcrl => {
            for precinct in 0..max_precincts {
                for component in 0..component_count {
                    for resolution in 0..resolution_count {
                        for layer in 0..layers {
                            push_if_present(layer, resolution, component, precinct);
                        }
                    }
                }
            }
        }
        Jp2kProgressionOrder::Cprl => {
            for component in 0..component_count {
                let component_max_precincts = (0..resolution_count as usize)
                    .map(|resolution| layout.precinct_count(component as usize, resolution))
                    .max()
                    .unwrap_or(0);
                for precinct in 0..component_max_precincts {
                    for resolution in 0..resolution_count {
                        for layer in 0..layers {
                            push_if_present(layer, resolution, component, precinct);
                        }
                    }
                }
            }
        }
        Jp2kProgressionOrder::Unknown(_) => {}
    }

    packets
}

fn tile_part_packet_coordinates_with_layout(
    info: &Jp2kCodestreamInfo,
    tile_part: &Jp2kTilePartInfo,
    layout: &TileLayout,
) -> Vec<Jp2kPacketCoordinate> {
    let packets = enumerate_packet_order_with_layout(info, layout);
    if tile_part.header.tile_part_count <= 1 {
        return packets;
    }

    packets
        .into_iter()
        .skip(tile_part.header.tile_part_index as usize)
        .step_by(tile_part.header.tile_part_count as usize)
        .collect()
}

fn parse_packet_payload(
    data: &[u8],
    coordinate: &Jp2kPacketCoordinate,
    info: &Jp2kCodestreamInfo,
    layout: &TileLayout,
    state: &mut PacketParserState,
) -> Result<(usize, usize, Vec<Jp2kCodeBlockContribution>), WsiError> {
    let component_index = coordinate.component as usize;
    let resolution_index = coordinate.resolution as usize;
    let precinct_index = coordinate.precinct as usize;
    let resolution_layout = &layout.components[component_index].resolutions[resolution_index];
    let resolution_state = &mut state.components[component_index].resolutions[resolution_index];
    let mut reader = PacketHeaderReader::new(data);
    let packet_present = reader.read_bit()?;
    if !packet_present {
        reader.align()?;
        let mut header_length = reader.bytes_read();
        if info.coding_style.eph_markers {
            expect_marker(data, header_length, &MARKER_EPH, "EPH")?;
            header_length += MARKER_EPH.len();
        }
        return Ok((header_length, 0, Vec::new()));
    }

    let mut contributions = Vec::new();
    for (band_index, band_layout) in resolution_layout.bands.iter().enumerate() {
        let precinct_layout = band_layout
            .precincts
            .get(precinct_index)
            .ok_or_else(|| WsiError::Jp2k("JP2K packet precinct index out of range".into()))?;
        let precinct_state = resolution_state.bands[band_index]
            .precincts
            .get_mut(precinct_index)
            .ok_or_else(|| WsiError::Jp2k("JP2K packet precinct state out of range".into()))?;

        for code_block_index in 0..precinct_layout.code_block_count {
            let code_block_state = precinct_state
                .code_blocks
                .get_mut(code_block_index)
                .ok_or_else(|| {
                    WsiError::Jp2k("JP2K packet code-block state out of range".into())
                })?;
            let included = if code_block_state.included_before {
                reader.read_bit()?
            } else {
                precinct_state.inclusion_tree.decode(
                    code_block_index,
                    coordinate.layer as i32 + 1,
                    &mut reader,
                )?
            };
            if !included {
                continue;
            }

            if !code_block_state.included_before {
                let mut zero_bit_plane_threshold = 0i32;
                while !precinct_state.zero_bit_plane_tree.decode(
                    code_block_index,
                    zero_bit_plane_threshold,
                    &mut reader,
                )? {
                    zero_bit_plane_threshold += 1;
                }
                code_block_state.zero_bit_planes = zero_bit_plane_threshold as u32;
                code_block_state.num_len_bits = 3;
                code_block_state.included_before = true;
            }

            let new_passes = read_num_passes(&mut reader)?;
            let len_increment = read_comma_code(&mut reader)?;
            code_block_state.num_len_bits = code_block_state
                .num_len_bits
                .checked_add(len_increment)
                .ok_or_else(|| WsiError::Jp2k("JP2K code-block length-bit overflow".into()))?;
            if code_block_state.num_passes + new_passes > DEFAULT_SEGMENT_MAX_PASSES {
                return Err(WsiError::Jp2k(format!(
                    "unsupported JP2K code-block pass count {}",
                    code_block_state.num_passes + new_passes
                )));
            }
            let bit_count = code_block_state.num_len_bits + floor_log2_u32(new_passes.max(1));
            let length = reader.read_bits(bit_count)? as usize;
            code_block_state.num_passes += new_passes;

            contributions.push(Jp2kCodeBlockContribution {
                component: coordinate.component,
                resolution: coordinate.resolution,
                band: band_layout.bandno,
                precinct: coordinate.precinct,
                code_block: code_block_index as u32,
                zero_bit_planes: code_block_state.zero_bit_planes,
                pass_count: new_passes,
                length,
                body_offset: 0,
            });
        }
    }

    reader.align()?;
    let mut header_length = reader.bytes_read();
    if info.coding_style.eph_markers {
        expect_marker(data, header_length, &MARKER_EPH, "EPH")?;
        header_length += MARKER_EPH.len();
    }
    let body_length = contributions
        .iter()
        .map(|contribution| contribution.length)
        .sum();
    Ok((header_length, body_length, contributions))
}

fn read_num_passes(reader: &mut PacketHeaderReader<'_>) -> Result<u32, WsiError> {
    if !reader.read_bit()? {
        return Ok(1);
    }
    if !reader.read_bit()? {
        return Ok(2);
    }
    let value = reader.read_bits(2)?;
    if value != 3 {
        return Ok(3 + value);
    }
    let value = reader.read_bits(5)?;
    if value != 31 {
        return Ok(6 + value);
    }
    Ok(37 + reader.read_bits(7)?)
}

fn read_comma_code(reader: &mut PacketHeaderReader<'_>) -> Result<u32, WsiError> {
    let mut value = 0u32;
    while reader.read_bit()? {
        value += 1;
    }
    Ok(value)
}

fn expect_marker(data: &[u8], offset: usize, marker: &[u8], name: &str) -> Result<(), WsiError> {
    let actual = data
        .get(offset..offset + marker.len())
        .ok_or_else(|| WsiError::Jp2k(format!("truncated JP2K {} marker", name)))?;
    if actual != marker {
        return Err(WsiError::Jp2k(format!("expected JP2K {} marker", name)));
    }
    Ok(())
}

fn ceil_div(value: i32, divisor: i32) -> i32 {
    (value + divisor - 1) / divisor
}

fn ceil_div_pow2(value: i32, exponent: i32) -> i32 {
    (value + (1i64 << exponent) as i32 - 1) >> exponent
}

fn ceil_div_pow2_i64(value: i64, exponent: i32) -> i32 {
    ((value + (1i64 << exponent) - 1) >> exponent) as i32
}

fn floor_div_pow2(value: i32, exponent: i32) -> i32 {
    value >> exponent
}

fn floor_log2_u32(value: u32) -> u32 {
    u32::BITS - 1 - value.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::jp2k_codestream::{
        Jp2kCodestreamInfo, Jp2kCodingStyleInfo, Jp2kComponentInfo, Jp2kQuantStep,
        Jp2kQuantizationInfo, Jp2kTilePartHeader, Jp2kWaveletTransform,
    };

    struct PacketHeaderWriter {
        bytes: Vec<u8>,
        buf: u32,
        bit_count: u32,
    }

    impl PacketHeaderWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                buf: 0,
                bit_count: 8,
            }
        }

        fn write_bit(&mut self, bit: bool) {
            if self.bit_count == 0 {
                self.byte_out();
            }
            self.bit_count -= 1;
            if bit {
                self.buf |= 1 << self.bit_count;
            }
        }

        fn write_bits(&mut self, value: u32, bit_count: u32) {
            for shift in (0..bit_count).rev() {
                self.write_bit(((value >> shift) & 1) != 0);
            }
        }

        fn write_num_passes(&mut self, passes: u32) {
            match passes {
                1 => self.write_bit(false),
                2 => self.write_bits(0b10, 2),
                3..=5 => self.write_bits(0b1100 | (passes - 3), 4),
                6..=36 => self.write_bits(0b111100000 | (passes - 6), 9),
                37..=164 => self.write_bits(0b1111111110000000 | (passes - 37), 16),
                _ => panic!("unsupported test pass count"),
            }
        }

        fn write_comma_code(&mut self, value: u32) {
            for _ in 0..value {
                self.write_bit(true);
            }
            self.write_bit(false);
        }

        fn finish(mut self) -> Vec<u8> {
            self.in_align();
            self.bytes
        }

        fn in_align(&mut self) {
            self.byte_out();
            if self.bit_count == 7 {
                self.byte_out();
            }
        }

        fn byte_out(&mut self) {
            self.buf = (self.buf << 8) & 0xFFFF;
            self.bit_count = if self.buf == 0xFF00 { 7 } else { 8 };
            self.bytes.push((self.buf >> 8) as u8);
        }
    }

    fn test_info(
        order: Jp2kProgressionOrder,
        layers: u16,
        width: u32,
        height: u32,
        decomposition_levels: u8,
    ) -> Jp2kCodestreamInfo {
        let mut steps = Vec::new();
        for _ in 0..(3 * decomposition_levels as usize + 1) {
            steps.push(Jp2kQuantStep {
                exponent: 8,
                mantissa: 0,
            });
        }
        Jp2kCodestreamInfo {
            image_origin_x: 0,
            image_origin_y: 0,
            image_width: width,
            image_height: height,
            tile_width: width,
            tile_height: height,
            tile_origin_x: 0,
            tile_origin_y: 0,
            tile_count_x: 1,
            tile_count_y: 1,
            components: vec![
                Jp2kComponentInfo {
                    precision_bits: 8,
                    is_signed: false,
                    horizontal_sample_separation: 1,
                    vertical_sample_separation: 1,
                };
                3
            ],
            coding_style: Jp2kCodingStyleInfo {
                progression_order: order,
                layers,
                multiple_component_transform: false,
                decomposition_levels,
                code_block_width_exponent: 4,
                code_block_height_exponent: 4,
                code_block_style: 0,
                transform: Jp2kWaveletTransform::Irreversible9x7,
                custom_precincts: false,
                sop_markers: false,
                eph_markers: false,
            },
            quantization: Jp2kQuantizationInfo {
                style: Jp2kQuantizationStyle::ScalarExpounded,
                guard_bits: 2,
                steps,
            },
            main_header_length: 0,
            tile_parts: vec![Jp2kTilePartInfo {
                header: Jp2kTilePartHeader {
                    tile_index: 0,
                    tile_part_length: 0,
                    tile_part_index: 0,
                    tile_part_count: 1,
                },
                data_offset: 0,
                data_length: 0,
            }],
            seen_markers: vec![],
        }
    }

    fn empty_packet(eph_markers: bool) -> Vec<u8> {
        let mut writer = PacketHeaderWriter::new();
        writer.write_bit(false);
        let mut packet = writer.finish();
        if eph_markers {
            packet.extend_from_slice(&MARKER_EPH);
        }
        packet
    }

    fn single_contribution_packet(eph_markers: bool, body: &[u8]) -> Vec<u8> {
        let mut writer = PacketHeaderWriter::new();
        writer.write_bit(true);
        writer.write_bit(true);
        writer.write_bit(true);
        writer.write_num_passes(1);
        writer.write_comma_code(0);
        writer.write_bits(body.len() as u32, 3);
        let mut packet = writer.finish();
        if eph_markers {
            packet.extend_from_slice(&MARKER_EPH);
        }
        packet.extend_from_slice(body);
        packet
    }

    #[test]
    fn enumerate_lrcp_packets() {
        let packets =
            enumerate_packet_order(&test_info(Jp2kProgressionOrder::Lrcp, 2, 512, 256, 2));
        assert_eq!(packets.len(), 18);
        assert_eq!(
            packets[0],
            Jp2kPacketCoordinate {
                layer: 0,
                resolution: 0,
                component: 0,
                precinct: 0,
            }
        );
        assert_eq!(
            packets.last().copied().unwrap(),
            Jp2kPacketCoordinate {
                layer: 1,
                resolution: 2,
                component: 2,
                precinct: 0,
            }
        );
    }

    #[test]
    fn enumerate_multiple_precincts_for_large_resolutions() {
        let packets =
            enumerate_packet_order(&test_info(Jp2kProgressionOrder::Rlcp, 1, 70000, 40000, 0));
        assert!(packets.iter().any(|packet| packet.precinct > 0));
    }

    #[test]
    fn tile_part_packet_coordinates_split_round_robin() {
        let info = test_info(Jp2kProgressionOrder::Lrcp, 2, 512, 256, 2);
        let tile_part = Jp2kTilePartInfo {
            header: Jp2kTilePartHeader {
                tile_index: 0,
                tile_part_length: 0,
                tile_part_index: 1,
                tile_part_count: 2,
            },
            data_offset: 0,
            data_length: 0,
        };

        let packets = tile_part_packet_coordinates(&info, &tile_part);
        assert_eq!(packets.len(), 9);
        assert_eq!(
            packets[0],
            Jp2kPacketCoordinate {
                layer: 0,
                resolution: 0,
                component: 1,
                precinct: 0,
            }
        );
    }

    #[test]
    fn parse_single_packet_header_and_body_ranges() {
        let info = test_info(Jp2kProgressionOrder::Lrcp, 1, 8, 8, 0);
        let mut codestream = Vec::new();
        codestream.extend_from_slice(&single_contribution_packet(
            false,
            &[0x11, 0x22, 0x33, 0x44],
        ));
        codestream.extend_from_slice(&empty_packet(false));
        codestream.extend_from_slice(&empty_packet(false));

        let tile_part = Jp2kTilePartInfo {
            header: Jp2kTilePartHeader {
                tile_index: 0,
                tile_part_length: 0,
                tile_part_index: 0,
                tile_part_count: 1,
            },
            data_offset: 0,
            data_length: codestream.len(),
        };

        let packets = parse_tile_part_packets(&codestream, &info, &tile_part).unwrap();
        assert_eq!(packets.len(), 3);
        let packet = &packets[0];
        assert_eq!(packet.coordinate.component, 0);
        assert_eq!(packet.header_offset, 0);
        assert_eq!(packet.body_length, 4);
        assert_eq!(packet.contributions.len(), 1);
        assert_eq!(packet.contributions[0].length, 4);
        assert_eq!(packet.contributions[0].body_offset, packet.body_offset);
        assert_eq!(packets[1].body_length, 0);
        assert_eq!(packets[2].body_length, 0);
    }

    #[test]
    fn parse_empty_packet_with_eph_marker() {
        let mut info = test_info(Jp2kProgressionOrder::Lrcp, 1, 8, 8, 0);
        info.coding_style.eph_markers = true;
        let mut codestream = Vec::new();
        codestream.extend_from_slice(&empty_packet(true));
        codestream.extend_from_slice(&empty_packet(true));
        codestream.extend_from_slice(&empty_packet(true));

        let tile_part = Jp2kTilePartInfo {
            header: Jp2kTilePartHeader {
                tile_index: 0,
                tile_part_length: 0,
                tile_part_index: 0,
                tile_part_count: 1,
            },
            data_offset: 0,
            data_length: codestream.len(),
        };

        let packets = parse_tile_part_packets(&codestream, &info, &tile_part).unwrap();
        assert_eq!(packets[0].body_length, 0);
        assert_eq!(packets[0].header_length, empty_packet(true).len());
        assert_eq!(packets[1].body_length, 0);
        assert_eq!(packets[2].body_length, 0);
    }
}
