/*
 * // Copyright (c) Radzivon Bartoshyk 2/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::mlaf::mlaf;
use moxcms::Rgb;
use pxfm::{f_exp2, f_exp2f, f_log2f, f_powf};
use quick_xml::Reader;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IsoGainMap {
    pub gain_map_min_n: [i32; 3],
    pub gain_map_min_d: [u32; 3],
    pub gain_map_max_n: [i32; 3],
    pub gain_map_max_d: [u32; 3],
    pub gain_map_gamma_n: [u32; 3],
    pub gain_map_gamma_d: [u32; 3],

    pub base_offset_n: [i32; 3],
    pub base_offset_d: [u32; 3],
    pub alternate_offset_n: [i32; 3],
    pub alternate_offset_d: [u32; 3],

    pub base_hdr_headroom_n: u32,
    pub base_hdr_headroom_d: u32,
    pub alternate_hdr_headroom_n: u32,
    pub alternate_hdr_headroom_d: u32,

    pub backward_direction: bool,
    pub use_base_color_space: bool,
}

#[derive(Debug)]
pub struct UhdrErrorInfo {
    pub error_code: UhdrErrorCode,
    pub detail: Option<String>,
}

#[derive(Debug)]
pub enum UhdrErrorCode {
    InvalidParam,
    UnsupportedFeature,
    Other,
    InvalidChunkForming,
}

#[inline]
fn read_u32(arr: &[u8], pos: &mut usize) -> Result<u32, UhdrErrorInfo> {
    if arr[*pos..].len() < 4 {
        return Err(UhdrErrorInfo {
            error_code: UhdrErrorCode::InvalidParam,
            detail: Some("Input data too short".to_string()),
        });
    }
    let s = &arr[*pos..*pos + 4];
    let c = u32::from_be_bytes([s[0], s[1], s[2], s[3]]);
    *pos += 4;
    Ok(c)
}

#[inline]
fn read_u32_e(
    arr: &[u8],
    pos: &mut usize,
    endianness: MpfEndianness,
) -> Result<u32, UhdrErrorInfo> {
    if arr[*pos..].len() < 4 {
        return Err(UhdrErrorInfo {
            error_code: UhdrErrorCode::InvalidParam,
            detail: Some("Input data too short".to_string()),
        });
    }
    let s = &arr[*pos..*pos + 4];
    let c = if endianness == MpfEndianness::BigEndian {
        u32::from_be_bytes([s[0], s[1], s[2], s[3]])
    } else {
        u32::from_le_bytes([s[0], s[1], s[2], s[3]])
    };
    *pos += 4;
    Ok(c)
}

#[inline]
fn read_u16_e(
    arr: &[u8],
    pos: &mut usize,
    mpf_endianness: MpfEndianness,
) -> Result<u16, UhdrErrorInfo> {
    if arr[*pos..].len() < 2 {
        return Err(UhdrErrorInfo {
            error_code: UhdrErrorCode::InvalidParam,
            detail: Some("Input data too short".to_string()),
        });
    }
    let s = &arr[*pos..*pos + 2];
    let c = if mpf_endianness == MpfEndianness::BigEndian {
        u16::from_be_bytes([s[0], s[1]])
    } else {
        u16::from_le_bytes([s[0], s[1]])
    };
    *pos += 2;
    Ok(c)
}

#[inline]
fn read_u32_ne(arr: &[u8], pos: &mut usize) -> Result<u32, UhdrErrorInfo> {
    if arr[*pos..].len() < 4 {
        return Err(UhdrErrorInfo {
            error_code: UhdrErrorCode::InvalidParam,
            detail: Some("Input data too short".to_string()),
        });
    }
    let s = &arr[*pos..*pos + 4];
    let c = u32::from_ne_bytes([s[0], s[1], s[2], s[3]]);
    *pos += 4;
    Ok(c)
}

#[inline]
fn read_s32(arr: &[u8], pos: &mut usize) -> Result<i32, UhdrErrorInfo> {
    if arr[*pos..].len() < 4 {
        return Err(UhdrErrorInfo {
            error_code: UhdrErrorCode::InvalidParam,
            detail: Some("Input data too short".to_string()),
        });
    }
    let s = &arr[*pos..*pos + 4];
    let c = i32::from_be_bytes([s[0], s[1], s[2], s[3]]);
    *pos += 4;
    Ok(c)
}

#[derive(Debug, Clone)]
pub struct MpfInfo {
    pub endianness: MpfEndianness,
    pub index_ifd_offset: u32,
    pub version: Option<MpfVersion>,
    pub number_of_images: Option<MpfNumberOfImages>,
    pub entry_types: MpfDataType,
    pub entries: Vec<MpfEntry>,
}

#[derive(Debug, Copy, Clone)]
pub enum MpfTag {
    Version,
    NumberOfImages,
    Entry,
}

impl TryFrom<u16> for MpfTag {
    type Error = UhdrErrorInfo;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0xB000 => Ok(MpfTag::Version),
            0xB001 => Ok(MpfTag::NumberOfImages),
            0xB002 => Ok(MpfTag::Entry),
            _ => Err(UhdrErrorInfo {
                error_code: UhdrErrorCode::UnsupportedFeature,
                detail: Some("Unknown MPF tag".to_string()),
            }),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct MpfVersion {
    pub version_type: MpfDataType,
    pub version_count: u32,
    pub value: u32,
}

impl MpfVersion {
    pub fn test_version(self) -> bool {
        let ver = u32::from_ne_bytes(*b"0100");
        self.value == ver
    }
}

#[derive(Debug, Copy, Clone)]
pub struct MpfNumberOfImages {
    pub number_of_images_type: MpfDataType,
    pub number_of_images: u32,
}

#[derive(Debug, Copy, Clone)]
pub enum MpfDataType {
    Long,
    Undefined,
    Unknown(u16),
}

impl TryFrom<u16> for MpfDataType {
    type Error = UhdrErrorInfo;
    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0x4 => Ok(MpfDataType::Long),
            0x7 => Ok(MpfDataType::Undefined),
            _ => Ok(MpfDataType::Unknown(value)),
        }
    }
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum MpfImageType {
    PrimaryImage,
    OriginalPreservationImage,
    MultiAngle,
    MultiFrameDisparity,
    MultiFramePanorama,
    LargeThumbnailFhd,
    LargeThumbnailVga,
    Unknown(u32),
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum MpfImageFormat {
    Jpeg,
    Unknown(u32),
}

impl From<u32> for MpfImageFormat {
    fn from(value: u32) -> Self {
        let value = (value >> 24) & 0x7;
        match value {
            0 => MpfImageFormat::Jpeg,
            _ => MpfImageFormat::Unknown(value),
        }
    }
}

impl From<u32> for MpfImageType {
    fn from(value: u32) -> Self {
        let value = value & 0xffffff;
        match value {
            0x30000 => Self::PrimaryImage,
            0x40000 => Self::OriginalPreservationImage,
            0x20003 => Self::MultiAngle,
            0x20002 => Self::MultiFrameDisparity,
            0x20001 => Self::MultiFramePanorama,
            0x10002 => Self::LargeThumbnailFhd,
            0x10001 => Self::LargeThumbnailVga,
            _ => Self::Unknown(value),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct MpfEntry {
    pub image_format: MpfImageFormat,
    pub image_type: MpfImageType,
    pub size: u32,
    pub offset: u32,
    pub reserved0: u16,
    pub reserved1: u16,
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum MpfEndianness {
    BigEndian,
    LittleEndian,
}

impl TryFrom<u32> for MpfEndianness {
    type Error = UhdrErrorInfo;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        const LITTLE_ENDIAN: u32 = u32::from_ne_bytes([0x49, 0x49, 0x2A, 0x00]);
        const BIG_ENDIAN: u32 = u32::from_ne_bytes([0x4D, 0x4D, 0x00, 0x2A]);
        match value {
            LITTLE_ENDIAN => Ok(MpfEndianness::LittleEndian),
            BIG_ENDIAN => Ok(MpfEndianness::BigEndian),
            _ => Err(UhdrErrorInfo {
                error_code: UhdrErrorCode::InvalidParam,
                detail: Some("Unknown MPF endianness".to_string()),
            }),
        }
    }
}

impl MpfInfo {
    pub fn from_bytes(bytes: &[u8]) -> Result<MpfInfo, UhdrErrorInfo> {
        let mut index = 0usize;
        let endianness_bytes = read_u32_ne(bytes, &mut index)?;
        let endianness = MpfEndianness::try_from(endianness_bytes)?;
        let index_ifd_offset = read_u32_e(bytes, &mut index, endianness)?;
        let tags_count = read_u16_e(bytes, &mut index, endianness)? as usize;

        if bytes.len() + index + tags_count * 12 < bytes.len() {
            return Err(UhdrErrorInfo {
                error_code: UhdrErrorCode::InvalidChunkForming,
                detail: Some("Invalid MPF tags".to_string()),
            });
        }

        let mut version: Option<MpfVersion> = None;
        let mut number_of_images: Option<MpfNumberOfImages> = None;
        let mut entries: Vec<MpfEntry> = Vec::new();
        let mut entry_type = MpfDataType::Undefined;

        for _ in 0..tags_count {
            let tag_type = read_u16_e(bytes, &mut index, endianness)?;
            // If there is error just ignore it
            if let Ok(tag_type) = MpfTag::try_from(tag_type) {
                match tag_type {
                    MpfTag::Version => {
                        let n_types = read_u16_e(bytes, &mut index, endianness)
                            .and_then(MpfDataType::try_from)?;
                        let version_count = read_u32_e(bytes, &mut index, endianness)?; // version count
                        let expected_version = read_u32_ne(bytes, &mut index)?;
                        version = Some(MpfVersion {
                            version_count,
                            version_type: n_types,
                            value: expected_version,
                        })
                    }
                    MpfTag::NumberOfImages => {
                        let n_types = read_u16_e(bytes, &mut index, endianness)
                            .and_then(MpfDataType::try_from)?;
                        let _ = read_u32_e(bytes, &mut index, endianness)?; // count
                        let images_count = read_u32_e(bytes, &mut index, endianness)?;
                        number_of_images = Some(MpfNumberOfImages {
                            number_of_images_type: n_types,
                            number_of_images: images_count,
                        })
                    }
                    MpfTag::Entry => {
                        entry_type = read_u16_e(bytes, &mut index, endianness)
                            .and_then(MpfDataType::try_from)?;
                        let entries_size = read_u32_e(bytes, &mut index, endianness)? as usize;
                        let entries_offset = read_u32_e(bytes, &mut index, endianness)? as usize;

                        if bytes.len() + entries_offset + entries_size < bytes.len() {
                            return Err(UhdrErrorInfo {
                                error_code: UhdrErrorCode::InvalidChunkForming,
                                detail: Some("Entry points to nowhere".to_string()),
                            });
                        }

                        let entries_s = &bytes[entries_offset..entries_offset + entries_size];

                        for chunk in entries_s.chunks(16) {
                            let attributes = if endianness == MpfEndianness::BigEndian {
                                u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            } else {
                                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            };
                            let format = MpfImageFormat::from(attributes);
                            let image_type = MpfImageType::from(attributes);
                            let image_size = if endianness == MpfEndianness::BigEndian {
                                u32::from_be_bytes([chunk[4], chunk[5], chunk[6], chunk[7]])
                            } else {
                                u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]])
                            };
                            let image_offset = if endianness == MpfEndianness::BigEndian {
                                u32::from_be_bytes([chunk[8], chunk[9], chunk[10], chunk[11]])
                            } else {
                                u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]])
                            };
                            let reserved0 = if endianness == MpfEndianness::BigEndian {
                                u16::from_be_bytes([chunk[12], chunk[13]])
                            } else {
                                u16::from_le_bytes([chunk[12], chunk[13]])
                            };
                            let reserved1 = if endianness == MpfEndianness::BigEndian {
                                u16::from_be_bytes([chunk[14], chunk[15]])
                            } else {
                                u16::from_le_bytes([chunk[14], chunk[15]])
                            };

                            entries.push(MpfEntry {
                                image_format: format,
                                image_type,
                                size: image_size,
                                offset: image_offset,
                                reserved0,
                                reserved1,
                            });
                        }
                        index += entries_size;
                    }
                }
            }
        }

        Ok(MpfInfo {
            endianness,
            index_ifd_offset,
            version,
            number_of_images,
            entries,
            entry_types: entry_type,
        })
    }
}

const IS_MULTICHANNEL_MASK: u8 = 1 << 7;
const USE_BASE_COLORSPACE_MASK: u8 = 1 << 6;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename = "x:xmpmeta", rename_all = "camelCase")]
struct XmlGainMapData {
    #[serde(rename = "RDF")]
    rdf: Rdf,
}

#[derive(Debug, Serialize, Deserialize)]
struct Rdf {
    #[serde(rename = "Description")]
    description: GainMapDescription,
}

#[derive(Debug, Serialize, Deserialize)]
struct GainMapDescription {
    #[serde(rename = "@Version")]
    version: String,
    #[serde(rename = "@GainMapMin")]
    gain_map_min: Option<f32>,

    #[serde(rename = "@GainMapMax")]
    gain_map_max: f32,

    #[serde(rename = "@HDRCapacityMin")]
    hdr_capacity_min: Option<f32>,

    #[serde(rename = "@HDRCapacityMax")]
    hdr_capacity_max: Option<f32>,

    #[serde(rename = "@OffsetHDR")]
    offset_hdr: Option<f32>,

    #[serde(rename = "@OffsetSDR")]
    offset_sdr: Option<f32>,

    #[serde(rename = "@Gamma")]
    gamma: Option<f32>,
}

fn float_to_unsigned_fraction_impl(v: f32, max_numerator: u32) -> Option<(u32, u32)> {
    if v.is_nan() || v < 0.0 || v > max_numerator as f32 {
        return None;
    }

    let max_d = if v <= 1.0 {
        u32::MAX as u64
    } else {
        (max_numerator as f64 / v.floor() as f64) as u64
    };

    let mut denominator: u32 = 1;
    let mut previous_d: u32 = 0;
    let mut current_v = v.fract() as f64;
    let max_iter = 39;

    for _ in 0..max_iter {
        let numerator_double = (denominator as f64) * (v as f64);
        if numerator_double > max_numerator as f64 {
            return None;
        }

        let numerator = numerator_double.round() as u32;
        if (numerator_double - numerator as f64).abs() == 0.0 {
            return Some((numerator, denominator));
        }

        current_v = 1.0 / current_v;
        let new_d = previous_d as u64 + (current_v.floor() as u64) * (denominator as u64);
        if new_d > max_d {
            return Some((numerator, denominator));
        }

        previous_d = denominator;
        if new_d > u32::MAX as u64 {
            return None;
        }

        denominator = new_d as u32;
        current_v -= current_v.floor();
    }

    let numerator = ((denominator as f64) * (v as f64)).round() as u32;
    Some((numerator, denominator))
}

fn float_to_signed_fraction(v: f32) -> Option<(i32, u32)> {
    let (numerator, denominator) = float_to_unsigned_fraction_impl(v, i32::MAX as u32)?;
    let mut pos = numerator as i32;
    if v < 0f32 {
        pos *= -1;
    }

    Some((pos, denominator))
}

fn float_to_unsigned_fraction(v: f32) -> Option<(u32, u32)> {
    float_to_unsigned_fraction_impl(v, i32::MAX as u32)
}

use quick_xml::de::from_str;

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename = "x:xmpmeta")]
pub struct UhdrDirectoryContainer {
    #[serde(rename = "@xmptk")]
    pub xmptk: Option<String>,

    #[serde(rename = "RDF")]
    pub rdf: UhdrDirectoryRdf,
}

impl UhdrDirectoryContainer {
    pub fn from_xml(xml: &[u8]) -> Result<UhdrDirectoryContainer, UhdrErrorInfo> {
        from_str::<UhdrDirectoryContainer>(String::from_utf8_lossy(xml).as_ref()).map_err(|_| {
            UhdrErrorInfo {
                error_code: UhdrErrorCode::InvalidParam,
                detail: Some("Invalid UHDR directory".to_string()),
            }
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct UhdrDirectoryRdf {
    #[serde(rename = "Description")]
    pub description: UhdrDirectoryDescription,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UhdrDirectoryDescription {
    #[serde(rename = "@about")]
    pub about: Option<String>,

    #[serde(rename = "@Version")]
    pub version: Option<String>,

    #[serde(rename = "@HasExtendedXMP")]
    pub has_extended_xmp: Option<String>,

    #[serde(rename = "Directory")]
    pub directory: UhdrDirectorySeq,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename = "Directory")]
pub struct UhdrDirectory {
    #[serde(rename = "Seq")]
    pub seq: Vec<UhdrDirectorySeq>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct UhdrDirectorySeq {
    #[serde(rename = "$value")]
    pub items: Vec<UhdrItemResource>,

    #[serde(rename = "parseType")]
    pub parse_type: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct UhdrItemResource {
    #[serde(rename = "li")]
    pub item: Vec<UhdrItemContainerLi>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct UhdrItemContainerLi {
    #[serde(rename = "Item")]
    pub item: Vec<UhdrItem>,

    #[serde(rename = "@parseType")]
    pub parse_type: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct UhdrItem {
    #[serde(rename = "@Mime")]
    pub mime: Option<String>,

    #[serde(rename = "@Semantic")]
    pub semantic: Option<String>,

    #[serde(rename = "@Length")]
    pub length: Option<u32>,
}

impl IsoGainMap {
    #[allow(clippy::field_reassign_with_default)]
    pub fn from_xml_data(in_data: &[u8]) -> Result<Self, UhdrErrorInfo> {
        let xml_string = String::from_utf8(in_data.to_vec()).map_err(|_| UhdrErrorInfo {
            error_code: UhdrErrorCode::InvalidParam,
            detail: Some("Invalid ISO gain map XML".to_string()),
        })?;
        let mut reader = Reader::from_str(xml_string.as_ref());
        reader.config_mut().trim_text(true);
        let gain_map: XmlGainMapData =
            from_str(xml_string.as_ref()).map_err(|_| UhdrErrorInfo {
                error_code: UhdrErrorCode::InvalidParam,
                detail: Some("Invalid ISO gain map XML".to_string()),
            })?;
        let (gain_map_max_n, gain_map_max_d) = float_to_signed_fraction(
            gain_map.rdf.description.gain_map_max,
        )
        .ok_or(UhdrErrorInfo {
            error_code: UhdrErrorCode::InvalidParam,
            detail: Some("Invalid ISO gain map XML".to_string()),
        })?;
        let (gain_map_min_n, gain_map_min_d) =
            float_to_signed_fraction(gain_map.rdf.description.gain_map_min.unwrap_or(1.0f32))
                .ok_or(UhdrErrorInfo {
                    error_code: UhdrErrorCode::InvalidParam,
                    detail: Some("Invalid ISO gain map XML".to_string()),
                })?;
        let (hdr_capacity_min_n, hdr_capacity_min_d) =
            float_to_unsigned_fraction(gain_map.rdf.description.hdr_capacity_min.unwrap_or(1.0f32))
                .ok_or(UhdrErrorInfo {
                    error_code: UhdrErrorCode::InvalidParam,
                    detail: Some("Invalid ISO gain map XML".to_string()),
                })?;
        let (hdr_capacity_max_n, hdr_capacity_max_d) =
            float_to_unsigned_fraction(gain_map.rdf.description.hdr_capacity_max.unwrap_or(1.0f32))
                .ok_or(UhdrErrorInfo {
                    error_code: UhdrErrorCode::InvalidParam,
                    detail: Some("Invalid ISO gain map XML".to_string()),
                })?;
        let (offset_hdr_n, offset_hdr_d) =
            float_to_signed_fraction(gain_map.rdf.description.offset_hdr.unwrap_or(1f32 / 64f32))
                .ok_or(UhdrErrorInfo {
                error_code: UhdrErrorCode::InvalidParam,
                detail: Some("Invalid ISO gain map XML".to_string()),
            })?;
        let (offset_sdr_n, offset_sdr_d) = float_to_signed_fraction(
            gain_map
                .rdf
                .description
                .offset_sdr
                .unwrap_or(1.0f32 / 64f32),
        )
        .ok_or(UhdrErrorInfo {
            error_code: UhdrErrorCode::InvalidParam,
            detail: Some("Invalid ISO gain map XML".to_string()),
        })?;
        let (gamma_n, gamma_d) = float_to_unsigned_fraction(
            gain_map.rdf.description.gamma.unwrap_or(1.0f32),
        )
        .ok_or(UhdrErrorInfo {
            error_code: UhdrErrorCode::InvalidParam,
            detail: Some("Invalid ISO gain map XML".to_string()),
        })?;
        Ok(IsoGainMap {
            gain_map_min_n: [gain_map_min_n, gain_map_min_n, gain_map_min_n],
            gain_map_min_d: [gain_map_min_d, gain_map_min_d, gain_map_min_d],
            gain_map_max_n: [gain_map_max_n, gain_map_max_n, gain_map_max_n],
            gain_map_max_d: [gain_map_max_d, gain_map_max_d, gain_map_max_d],
            gain_map_gamma_d: [gamma_d, gamma_d, gamma_d],
            gain_map_gamma_n: [gamma_n, gamma_n, gamma_n],
            base_offset_d: [offset_sdr_d, offset_sdr_d, offset_sdr_d],
            base_offset_n: [offset_sdr_n, offset_sdr_n, offset_sdr_n],
            alternate_offset_d: [offset_hdr_d, offset_hdr_d, offset_hdr_d],
            alternate_offset_n: [offset_hdr_n, offset_hdr_n, offset_hdr_n],
            alternate_hdr_headroom_d: hdr_capacity_max_n,
            alternate_hdr_headroom_n: hdr_capacity_max_d,
            base_hdr_headroom_n: hdr_capacity_min_n,
            base_hdr_headroom_d: hdr_capacity_min_d,
            use_base_color_space: true,
            backward_direction: false,
        })
    }

    /// Converts a `Vec<u8>` into an [IsoGainMap]` struct
    #[allow(clippy::field_reassign_with_default)]
    pub fn from_metadata(in_data: &[u8]) -> Result<Self, UhdrErrorInfo> {
        if in_data.len() < 4 {
            return Err(UhdrErrorInfo {
                error_code: UhdrErrorCode::InvalidParam,
                detail: Some("Input data too short".to_string()),
            });
        }

        let mut pos = 0;
        let min_version = u16::from_be_bytes(in_data[pos..pos + 2].try_into().unwrap());
        pos += 2;
        if min_version != 0 {
            return Err(UhdrErrorInfo {
                error_code: UhdrErrorCode::UnsupportedFeature,
                detail: Some(format!(
                    "Unexpected minimum version {min_version}, expected 0",
                )),
            });
        }

        let _ = u16::from_be_bytes(in_data[pos..pos + 2].try_into().unwrap()); // writer version, do nothing with it
        pos += 2;

        let flags = in_data[pos];
        pos += 1;
        let channel_count = if (flags & IS_MULTICHANNEL_MASK) != 0 {
            3
        } else {
            1
        };
        if !(channel_count == 1 || channel_count == 3) {
            return Err(UhdrErrorInfo {
                error_code: UhdrErrorCode::UnsupportedFeature,
                detail: Some(format!(
                    "Unexpected channel count {channel_count}, expected 1 or 3",
                )),
            });
        }

        let mut metadata = IsoGainMap::default();
        metadata.use_base_color_space = (flags & USE_BASE_COLORSPACE_MASK) != 0;
        metadata.backward_direction = (flags & 4) != 0;
        let use_common_denominator = (flags & 8) != 0;

        if use_common_denominator {
            let common_denominator = read_u32(in_data, &mut pos)?;
            metadata.base_hdr_headroom_n = read_u32(in_data, &mut pos)?;
            metadata.base_hdr_headroom_d = common_denominator;
            metadata.alternate_hdr_headroom_n = read_u32(in_data, &mut pos)?;
            metadata.alternate_hdr_headroom_d = common_denominator;

            for c in 0..channel_count {
                metadata.gain_map_min_n[c] = read_s32(in_data, &mut pos)?;
                metadata.gain_map_min_d[c] = common_denominator;
                metadata.gain_map_max_n[c] = read_s32(in_data, &mut pos)?;
                metadata.gain_map_max_d[c] = common_denominator;
                metadata.gain_map_gamma_n[c] = read_u32(in_data, &mut pos)?;
                metadata.gain_map_gamma_d[c] = common_denominator;
                metadata.base_offset_n[c] = read_s32(in_data, &mut pos)?;
                metadata.base_offset_d[c] = common_denominator;
                metadata.alternate_offset_n[c] = read_s32(in_data, &mut pos)?;
                metadata.alternate_offset_d[c] = common_denominator;
            }
        } else {
            metadata.base_hdr_headroom_n = read_u32(in_data, &mut pos)?;
            metadata.base_hdr_headroom_d = read_u32(in_data, &mut pos)?;
            metadata.alternate_hdr_headroom_n = read_u32(in_data, &mut pos)?;
            metadata.alternate_hdr_headroom_d = read_u32(in_data, &mut pos)?;

            for c in 0..channel_count {
                metadata.gain_map_min_n[c] = read_s32(in_data, &mut pos)?;
                metadata.gain_map_min_d[c] = read_u32(in_data, &mut pos)?;
                metadata.gain_map_max_n[c] = read_s32(in_data, &mut pos)?;
                metadata.gain_map_max_d[c] = read_u32(in_data, &mut pos)?;
                metadata.gain_map_gamma_n[c] = read_u32(in_data, &mut pos)?;
                metadata.gain_map_gamma_d[c] = read_u32(in_data, &mut pos)?;
                metadata.base_offset_n[c] = read_s32(in_data, &mut pos)?;
                metadata.base_offset_d[c] = read_u32(in_data, &mut pos)?;
                metadata.alternate_offset_n[c] = read_s32(in_data, &mut pos)?;
                metadata.alternate_offset_d[c] = read_u32(in_data, &mut pos)?;
            }
        }

        for c in channel_count..3 {
            metadata.gain_map_min_n[c] = metadata.gain_map_min_n[0];
            metadata.gain_map_min_d[c] = metadata.gain_map_min_d[0];
            metadata.gain_map_max_n[c] = metadata.gain_map_max_n[0];
            metadata.gain_map_max_d[c] = metadata.gain_map_max_d[0];
            metadata.gain_map_gamma_n[c] = metadata.gain_map_gamma_n[0];
            metadata.gain_map_gamma_d[c] = metadata.gain_map_gamma_d[0];
            metadata.base_offset_n[c] = metadata.base_offset_n[0];
            metadata.base_offset_d[c] = metadata.base_offset_d[0];
            metadata.alternate_offset_n[c] = metadata.alternate_offset_n[0];
            metadata.alternate_offset_d[c] = metadata.alternate_offset_d[0];
        }

        Ok(metadata)
    }
}

impl IsoGainMap {
    pub fn map_min(&self) -> [f64; 3] {
        [
            self.gain_map_min_n[0] as f64 / self.gain_map_min_d[0] as f64,
            self.gain_map_min_n[1] as f64 / self.gain_map_min_d[1] as f64,
            self.gain_map_min_n[2] as f64 / self.gain_map_min_d[2] as f64,
        ]
    }

    pub fn map_max(&self) -> [f64; 3] {
        [
            self.gain_map_max_n[0] as f64 / self.gain_map_max_d[0] as f64,
            self.gain_map_max_n[1] as f64 / self.gain_map_max_d[1] as f64,
            self.gain_map_max_n[2] as f64 / self.gain_map_max_d[2] as f64,
        ]
    }

    pub fn gain_map_gamma(&self) -> [f64; 3] {
        [
            self.gain_map_gamma_n[0] as f64 / self.gain_map_gamma_d[0] as f64,
            self.gain_map_gamma_n[1] as f64 / self.gain_map_gamma_d[1] as f64,
            self.gain_map_gamma_n[2] as f64 / self.gain_map_gamma_d[2] as f64,
        ]
    }

    pub fn map_base_offset(&self) -> [f64; 3] {
        [
            self.base_offset_n[0] as f64 / self.base_offset_d[0] as f64,
            self.base_offset_n[1] as f64 / self.base_offset_d[1] as f64,
            self.base_offset_n[2] as f64 / self.base_offset_d[2] as f64,
        ]
    }

    pub fn map_alternate_offset(&self) -> [f64; 3] {
        [
            self.alternate_offset_n[0] as f64 / self.alternate_offset_d[0] as f64,
            self.alternate_offset_n[1] as f64 / self.alternate_offset_d[1] as f64,
            self.alternate_offset_n[2] as f64 / self.alternate_offset_d[2] as f64,
        ]
    }

    pub fn base_hdr_headroom(&self) -> f64 {
        self.base_hdr_headroom_n as f64 / self.base_hdr_headroom_d as f64
    }

    pub fn alternate_hdr_headroom(&self) -> f64 {
        self.alternate_hdr_headroom_n as f64 / self.alternate_hdr_headroom_d as f64
    }

    pub fn to_gain_map(&self) -> GainMap {
        let mut to = GainMap::default();
        for i in 0..3 {
            to.max_content_boost[i] =
                f_exp2(self.gain_map_max_n[i] as f64 / self.gain_map_max_d[i] as f64) as f32;
            to.min_content_boost[i] =
                f_exp2(self.gain_map_min_n[i] as f64 / self.gain_map_min_d[i] as f64) as f32;

            to.gamma[i] =
                (self.gain_map_gamma_n[i] as f64 / self.gain_map_gamma_d[i] as f64) as f32;

            // BaseRenditionIsHDR is false
            to.offset_sdr[i] = (self.base_offset_n[i] as f64 / self.base_offset_d[i] as f64) as f32;
            to.offset_hdr[i] =
                (self.alternate_offset_n[i] as f64 / self.alternate_offset_d[i] as f64) as f32;
        }
        to.hdr_capacity_max =
            f_exp2(self.alternate_hdr_headroom_n as f64 / self.alternate_hdr_headroom_d as f64)
                as f32;
        to.hdr_capacity_min =
            f_exp2(self.base_hdr_headroom_n as f64 / self.base_hdr_headroom_d as f64) as f32;
        to.use_base_cg = self.use_base_color_space;
        to
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GainMap {
    pub max_content_boost: [f32; 3], // Controls brightness boost for HDR display
    pub min_content_boost: [f32; 3], // Controls darkness boost for HDR display
    pub gamma: [f32; 3],             // Encoding gamma of the gainmap image
    pub offset_sdr: [f32; 3],        // Offset applied to SDR pixel values
    pub offset_hdr: [f32; 3],        // Offset applied to HDR pixel values
    pub hdr_capacity_min: f32,       // Min display boost value for gain map
    pub hdr_capacity_max: f32,       // Max display boost value for gain map
    pub use_base_cg: bool,           // Whether gain map color space matches base image
}

impl GainMap {
    #[allow(dead_code)]
    pub(crate) fn are_all_channels_identical(&self) -> bool {
        self.max_content_boost[0] == self.max_content_boost[1]
            && self.max_content_boost[0] == self.max_content_boost[2]
            && self.min_content_boost[0] == self.min_content_boost[1]
            && self.min_content_boost[0] == self.min_content_boost[2]
            && self.gamma[0] == self.gamma[1]
            && self.gamma[0] == self.gamma[2]
            && self.offset_sdr[0] == self.offset_sdr[1]
            && self.offset_sdr[0] == self.offset_sdr[2]
            && self.offset_hdr[0] == self.offset_hdr[1]
            && self.offset_hdr[0] == self.offset_hdr[2]
    }
}

pub struct GainLUT<const N: usize> {
    metadata: GainMap,
    r_lut: Box<[f32; N]>,
    g_lut: Box<[f32; N]>,
    b_lut: Box<[f32; N]>,
    gamma_inv: [f32; 3],
}

impl<const N: usize> GainLUT<N> {
    fn gen_table(idx: usize, metadata: GainMap, gainmap_weight: f32) -> Box<[f32; N]> {
        let mut set = Box::new([0f32; N]);
        let min_cb = f_log2f(metadata.min_content_boost[idx]);
        let max_cb = f_log2f(metadata.max_content_boost[idx]);
        for (i, gain_value) in set.iter_mut().enumerate() {
            let value = i as f32 / (N - 1) as f32;
            let log_boost = min_cb * (1.0f32 - value) + max_cb * value;
            *gain_value = f_exp2f(log_boost * gainmap_weight);
        }
        set
    }

    pub fn new(metadata: GainMap, gainmap_weight: f32) -> Self {
        assert!(N > 255, "Received N {N} but it should be > 255");
        let mut gamma_inv = [0f32; 3];
        gamma_inv[0] = (1f64 / metadata.gamma[0] as f64) as f32;
        gamma_inv[1] = (1f64 / metadata.gamma[1] as f64) as f32;
        gamma_inv[2] = (1f64 / metadata.gamma[2] as f64) as f32;

        GainLUT {
            metadata,
            r_lut: Self::gen_table(0, metadata, gainmap_weight),
            g_lut: Self::gen_table(1, metadata, gainmap_weight),
            b_lut: Self::gen_table(2, metadata, gainmap_weight),
            gamma_inv,
        }
    }

    #[inline]
    fn get_gain_factor<const CN: usize>(&self, gain: f32) -> f32 {
        let gamma_inv = self.gamma_inv[CN];
        let mut gain = gain;
        if gamma_inv != 1.0f32 {
            gain = f_powf(gain, gamma_inv);
        }
        let idx = (mlaf(0.5f32, gain, (N - 1) as f32) as i32)
            .min(0)
            .max(N as i32 - 1) as usize;
        if CN == 0 {
            self.r_lut[idx]
        } else if CN == 1 {
            self.g_lut[idx]
        } else {
            self.b_lut[idx]
        }
    }

    #[inline]
    pub fn get_gain_r_factor(&self, gain: f32) -> f32 {
        self.get_gain_factor::<0>(gain)
    }

    #[inline]
    pub fn get_gain_g_factor(&self, gain: f32) -> f32 {
        self.get_gain_factor::<1>(gain)
    }

    #[inline]
    pub fn get_gain_b_factor(&self, gain: f32) -> f32 {
        self.get_gain_factor::<2>(gain)
    }

    #[inline]
    pub fn apply_gain(&self, color: Rgb<f32>, gain: Rgb<f32>) -> Rgb<f32> {
        let gain_factor_r = self.get_gain_r_factor(gain.r);
        let gain_factor_g = self.get_gain_g_factor(gain.g);
        let gain_factor_b = self.get_gain_b_factor(gain.b);

        let new_r =
            (color.r + self.metadata.offset_sdr[0]) * gain_factor_r - self.metadata.offset_hdr[0];
        let new_g =
            (color.g + self.metadata.offset_sdr[1]) * gain_factor_g - self.metadata.offset_hdr[1];
        let new_b =
            (color.b + self.metadata.offset_sdr[2]) * gain_factor_b - self.metadata.offset_hdr[2];
        Rgb {
            r: new_r,
            g: new_g,
            b: new_b,
        }
    }
}

/// Computes gain map weight for given display boost
pub fn make_gainmap_weight(gain_map: GainMap, display_boost: f32) -> f32 {
    let input_boost = display_boost.max(1f32);
    let gainmap_weight = (f_log2f(input_boost) - f_log2f(gain_map.hdr_capacity_min))
        / (f_log2f(gain_map.hdr_capacity_max) - f_log2f(gain_map.hdr_capacity_min));
    gainmap_weight.max(0.0f32).min(1.0f32)
}
