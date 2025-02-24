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
use crate::cms::cicp::{
    ChromacityTriple, ColorPrimaries, MatrixCoefficients, TransferCharacteristics,
};
use crate::cms::gamut::gamut_to_xyz;
use crate::cms::matrix::{Matrix3f, BT2020_MATRIX, DISPLAY_P3_MATRIX, SRGB_MATRIX};
use crate::cms::transform::{adapt_matrix_to_d50, adapt_matrix_to_illuminant_xyz, D50_XYZ};
use crate::cms::trc::Trc;
use crate::cms::{GamutColorSpace, XyY};
use crate::{ForgeError, Xyz};
use std::io::Read;

/// Constants for color profiles
const DISPLAY_PROFILE: u32 = 0x6D6E7472; // Example constant for profile class

pub const RGB_SIGNATURE: u32 = 0x52474220;
pub const GRAY_SIGNATURE: u32 = 0x47524159;
#[allow(unused)]
pub const XYZ_SIGNATURE: u32 = 0x58595A20;
#[allow(unused)]
pub const LAB_SIGNATURE: u32 = 0x4C616220;
#[allow(unused)]
pub const CMYK_SIGNATURE: u32 = 0x434D594B; // 'CMYK'

const XYZ_PCS_SPACE: u32 = 0x58595A20; // Example constant for XYZ PCS space
const ACSP_SIGNATURE: u32 = u32::from_ne_bytes(*b"acsp").to_be(); // 'acsp' signature for ICC

/// Constants representing the min and max values that fit in a signed 32-bit integer as a float
const MAX_S32_FITS_IN_FLOAT: f32 = 2_147_483_647.0; // i32::MAX as f32
const MIN_S32_FITS_IN_FLOAT: f32 = -2_147_483_648.0; // i32::MIN as f32

/// Fixed-point scaling factor (assuming Fixed1 = 65536 like in ICC profiles)
const FIXED1: f32 = 65536.0;
const MAX_PROFILE_SIZE: usize = 1024 * 1024 * 3;
const TAG_SIZE: usize = 12;
const MARK_TRC_CURV: u32 = u32::from_ne_bytes(*b"curv").to_be();
const MARK_TRC_PARAM: u32 = u32::from_ne_bytes(*b"para").to_be();

const R_TAG_XYZ: u32 = u32::from_ne_bytes(*b"rXYZ").to_be();
const G_TAG_XYZ: u32 = u32::from_ne_bytes(*b"gXYZ").to_be();
const B_TAG_XYZ: u32 = u32::from_ne_bytes(*b"bXYZ").to_be();
const R_TAG_TRC: u32 = u32::from_ne_bytes(*b"rTRC").to_be();
const G_TAG_TRC: u32 = u32::from_ne_bytes(*b"gTRC").to_be();
const B_TAG_TRC: u32 = u32::from_ne_bytes(*b"bTRC").to_be();
const K_TAG_TRC: u32 = u32::from_ne_bytes(*b"kTRC").to_be();
const WT_PT_TAG: u32 = u32::from_ne_bytes(*b"wtpt").to_be();
const CICP_TAG: u32 = u32::from_ne_bytes(*b"cicp").to_be();
const CHAD_TAG: u32 = u32::from_ne_bytes(*b"chad").to_be();
const CHROMATIC_TYPE: u32 = u32::from_ne_bytes(*b"sf32").to_be();
const XYZ_TYPE: u32 = u32::from_ne_bytes(*b"XYZ ").to_be();
const DISPLAY_DEVICE_PROFILE: u32 = u32::from_ne_bytes(*b"mntr").to_be();
const INPUT_DEVICE_PROFILE: u32 = u32::from_ne_bytes(*b"scnr").to_be();
const OUTPUT_DEVICE_PROFILE: u32 = u32::from_ne_bytes(*b"prtr").to_be();
#[allow(unused)]
const DEVICE_LINK_PROFILE: u32 = u32::from_ne_bytes(*b"link").to_be();
const COLOR_SPACE_PROFILE: u32 = u32::from_ne_bytes(*b"spac").to_be();

/// Clamps the float value within the range of an `i32`
/// Returns `i32::MAX` for NaN values.
#[inline]
const fn float_saturate2int(x: f32) -> i32 {
    if x.is_nan() {
        return i32::MAX;
    }
    x.clamp(MIN_S32_FITS_IN_FLOAT, MAX_S32_FITS_IN_FLOAT) as i32
}

/// Converts a float to a fixed-point integer representation
#[inline]
const fn float_round_to_fixed(x: f32) -> i32 {
    float_saturate2int((x as f64 * FIXED1 as f64 + 0.5) as f32)
}

#[derive(Clone, Debug, Copy)]
pub struct Chromacity {
    pub x: f32,
    pub y: f32,
}

impl Chromacity {
    pub const fn to_xyz(&self) -> Xyz {
        Xyz {
            x: self.x / self.y,
            y: 1f32,
            z: (1f32 - self.x - self.y) / self.y,
        }
    }

    pub const D65: Chromacity = Chromacity {
        x: 0.31272,
        y: 0.32903,
    };

    pub const D50: Chromacity = Chromacity {
        x: 0.34567,
        y: 0.35850,
    };
}

impl TryFrom<Xyz> for Chromacity {
    type Error = ForgeError;
    fn try_from(xyz: Xyz) -> Result<Self, Self::Error> {
        let sum = xyz.x + xyz.y + xyz.z;

        // Avoid division by zero or invalid XYZ values
        if sum == 0.0 {
            return Err(ForgeError::DivisionByZero);
        }

        let chromacity_x = xyz.x / sum;
        let chromacity_y = xyz.y / sum;

        Ok(Chromacity {
            x: chromacity_x,
            y: chromacity_y,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub enum RenderingIntent {
    AbsoluteColorimetric = 3,
    Saturation = 2,
    RelativeColorimetric = 1,
    #[default]
    Perceptual = 0,
}

impl TryFrom<u32> for RenderingIntent {
    type Error = ForgeError;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(RenderingIntent::Perceptual),
            1 => Ok(RenderingIntent::RelativeColorimetric),
            2 => Ok(RenderingIntent::Saturation),
            3 => Ok(RenderingIntent::AbsoluteColorimetric),
            _ => Err(ForgeError::InvalidRenderingIntent),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IccHeader {
    pub size: u32,                    // Size of the profile (computed)
    pub cmm_type: u32,                // Preferred CMM type (ignored)
    pub version: u32,                 // Version (4.3 or 4.4 if CICP is included)
    pub profile_class: u32,           // Display device profile
    pub data_color_space: u32,        // RGB input color space
    pub pcs: u32,                     // Profile connection space
    pub creation_date_time: [u8; 12], // Date and time (ignored)
    pub signature: u32,               // Profile signature
    pub platform: u32,                // Platform target (ignored)
    pub flags: u32,                   // Flags (not embedded, can be used independently)
    pub device_manufacturer: u32,     // Device manufacturer (ignored)
    pub device_model: u32,            // Device model (ignored)
    pub device_attributes: [u8; 8],   // Device attributes (ignored)
    pub rendering_intent: u32,        // Relative colorimetric rendering intent
    pub illuminant_x: i32,            // D50 standard illuminant X
    pub illuminant_y: i32,            // D50 standard illuminant Y
    pub illuminant_z: i32,            // D50 standard illuminant Z
    pub creator: u32,                 // Profile creator (ignored)
    pub profile_id: [u8; 16],         // Profile id checksum (ignored)
    pub reserved: [u8; 28],           // Reserved (ignored)
    pub tag_count: u32,               // Technically not part of header, but required
}

impl IccHeader {
    pub fn new(size: u32) -> Self {
        Self {
            size,
            cmm_type: 0,
            version: 0x04300000u32.to_be(),
            profile_class: DISPLAY_PROFILE.to_be(),
            data_color_space: RGB_SIGNATURE.to_be(),
            pcs: XYZ_PCS_SPACE.to_be(),
            creation_date_time: [0; 12],
            signature: ACSP_SIGNATURE.to_be(),
            platform: 0,
            flags: 0x00000000,
            device_manufacturer: 0,
            device_model: 0,
            device_attributes: [0; 8],
            rendering_intent: 1u32.to_be(),
            illuminant_x: float_round_to_fixed(D50_XYZ.x).to_be(),
            illuminant_y: float_round_to_fixed(D50_XYZ.y).to_be(),
            illuminant_z: float_round_to_fixed(D50_XYZ.z).to_be(),
            creator: 0,
            profile_id: [0; 16],
            reserved: [0; 28],
            tag_count: 0,
        }
    }

    pub fn new_from_slice(slice: &[u8]) -> Result<Self, ForgeError> {
        if slice.len() < size_of::<IccHeader>() {
            return Err(ForgeError::InvalidIcc);
        }
        let mut cursor = std::io::Cursor::new(slice);
        let mut buffer = [0u8; size_of::<IccHeader>()];
        cursor
            .read_exact(&mut buffer)
            .map_err(|_| ForgeError::InvalidIcc)?;

        let header = Self {
            size: u32::from_be_bytes(buffer[0..4].try_into().unwrap()),
            cmm_type: u32::from_be_bytes(buffer[4..8].try_into().unwrap()),
            version: u32::from_be_bytes(buffer[8..12].try_into().unwrap()),
            profile_class: u32::from_be_bytes(buffer[12..16].try_into().unwrap()),
            data_color_space: u32::from_be_bytes(buffer[16..20].try_into().unwrap()),
            pcs: u32::from_be_bytes(buffer[20..24].try_into().unwrap()),
            creation_date_time: buffer[24..36].try_into().unwrap(),
            signature: u32::from_be_bytes(buffer[36..40].try_into().unwrap()),
            platform: u32::from_be_bytes(buffer[40..44].try_into().unwrap()),
            flags: u32::from_be_bytes(buffer[44..48].try_into().unwrap()),
            device_manufacturer: u32::from_be_bytes(buffer[48..52].try_into().unwrap()),
            device_model: u32::from_be_bytes(buffer[52..56].try_into().unwrap()),
            device_attributes: buffer[56..64].try_into().unwrap(),
            rendering_intent: u32::from_be_bytes(buffer[64..68].try_into().unwrap()),
            illuminant_x: i32::from_be_bytes(buffer[68..72].try_into().unwrap()),
            illuminant_y: i32::from_be_bytes(buffer[72..76].try_into().unwrap()),
            illuminant_z: i32::from_be_bytes(buffer[76..80].try_into().unwrap()),
            creator: u32::from_be_bytes(buffer[80..84].try_into().unwrap()),
            profile_id: buffer[84..100].try_into().unwrap(),
            reserved: buffer[100..128].try_into().unwrap(),
            tag_count: u32::from_be_bytes(buffer[128..132].try_into().unwrap()),
        };

        if header.signature != ACSP_SIGNATURE {
            return Err(ForgeError::InvalidIcc);
        }
        Ok(header)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CicpProfile {
    pub color_primaries: ColorPrimaries,
    pub transfer_characteristics: TransferCharacteristics,
    pub matrix_coefficients: MatrixCoefficients,
}

#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct ColorProfile {
    pub pcs: u32,
    pub color_space: u32,
    pub profile_class: u32,
    pub rendering_intent: RenderingIntent,
    pub red_colorant: Xyz,
    pub green_colorant: Xyz,
    pub blue_colorant: Xyz,
    pub white_point: Option<Xyz>,
    pub red_trc: Option<Trc>,
    pub green_trc: Option<Trc>,
    pub blue_trc: Option<Trc>,
    pub gray_trc: Option<Trc>,
    pub cicp: Option<CicpProfile>,
    pub chromatic_adaptation: Option<Matrix3f>,
}

/* produces the nearest float to 'a' with a maximum error
 * of 1/1024 which happens for large values like 0x40000040 */
#[inline]
pub(crate) const fn s15_fixed16_number_to_float(a: i32) -> f32 {
    a as f32 / 65536.
}

#[inline]
const fn u16_number_to_float(a: u32) -> f32 {
    a as f32 / 65536.
}

impl ColorProfile {
    #[inline]
    fn read_trc_tag(slice: &[u8], entry: usize, tag_size: usize) -> Result<Trc, ForgeError> {
        let tag_size = if tag_size == 0 { TAG_SIZE } else { tag_size };
        let last_tag_offset = tag_size + entry;
        if last_tag_offset > slice.len() {
            return Err(ForgeError::InvalidIcc);
        }
        let tag = &slice[entry..last_tag_offset];
        if tag.len() < 12 {
            return Err(ForgeError::InvalidIcc);
        }

        let curve_type = u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]);
        if curve_type == MARK_TRC_CURV {
            let entry_count = u32::from_be_bytes([tag[8], tag[9], tag[10], tag[11]]) as usize;
            if entry_count == 0 {
                return Ok(Trc::Lut(vec![]));
            }
            if entry_count > 40000 {
                return Err(ForgeError::CurveLutIsTooLarge);
            }
            if tag.len() < 12 + entry_count * size_of::<u16>() {
                return Err(ForgeError::InvalidIcc);
            }
            let curve_sliced = &tag[12..entry_count * size_of::<u16>()];
            let mut curve_values = vec![0u16; entry_count];
            for (value, curve_value) in curve_sliced.chunks_exact(2).zip(curve_values.iter_mut()) {
                let gamma_s15 = u16::from_be_bytes([value[0], value[1]]);
                *curve_value = gamma_s15;
            }
            Ok(Trc::Lut(curve_values))
        } else if curve_type == MARK_TRC_PARAM {
            let entry_count = u16::from_be_bytes([tag[8], tag[9]]) as usize;
            if entry_count > 4 {
                return Err(ForgeError::InvalidIcc);
            }

            const COUNT_TO_LENGTH: [usize; 5] = [1, 3, 4, 5, 7]; //PARAMETRIC_CURVE_TYPE

            if tag.len() < 12 + COUNT_TO_LENGTH[entry_count] * size_of::<u32>() {
                return Err(ForgeError::InvalidIcc);
            }
            let curve_sliced = &tag[12..12 + COUNT_TO_LENGTH[entry_count] * size_of::<u32>()];
            let mut params = vec![0f32; COUNT_TO_LENGTH[entry_count]];
            for (value, param_value) in curve_sliced.chunks_exact(4).zip(params.iter_mut()) {
                let parametric_value = u32::from_be_bytes([value[0], value[1], value[2], value[3]]);
                *param_value = u16_number_to_float(parametric_value);
            }
            if entry_count == 1 || entry_count == 2 {
                /* we have a type 1 or type 2 function that has a division by 'a' */
                let a: f32 = params[1];
                if a == 0.0 {
                    return Err(ForgeError::ParametricCurveZeroDivision);
                }
            }
            return Ok(Trc::Parametric(params));
        } else {
            return Err(ForgeError::InvalidIcc);
        }
    }

    #[inline]
    fn read_chad_tag(slice: &[u8], entry: usize, tag_size: usize) -> Result<Matrix3f, ForgeError> {
        let tag_size = if tag_size == 0 { TAG_SIZE } else { tag_size };
        let last_tag_offset = tag_size + entry;
        if last_tag_offset > slice.len() {
            return Err(ForgeError::InvalidIcc);
        }
        if slice[entry..].len() < 8 {
            return Err(ForgeError::InvalidIcc);
        }
        let tag0 = &slice[entry..entry + 8];
        let c_type = u32::from_be_bytes([tag0[0], tag0[1], tag0[2], tag0[3]]);
        if c_type != CHROMATIC_TYPE {
            return Err(ForgeError::InvalidIcc);
        }
        if slice.len() < 9 * size_of::<u32>() + 8 {
            return Err(ForgeError::InvalidIcc);
        }
        let tag = &slice[entry + 8..last_tag_offset];
        if tag.len() != size_of::<Matrix3f>() {
            return Err(ForgeError::InvalidIcc);
        }
        let mut matrix = Matrix3f::default();
        for (i, chunk) in tag.chunks_exact(4).enumerate() {
            let q15_16_x = i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            matrix.v[i / 3][i % 3] = s15_fixed16_number_to_float(q15_16_x);
        }
        Ok(matrix)
    }

    #[inline]
    fn read_xyz_tag(slice: &[u8], entry: usize, tag_size: usize) -> Result<Xyz, ForgeError> {
        let tag_size = if tag_size == 0 { TAG_SIZE } else { tag_size };
        let last_tag_offset = tag_size + entry;
        if last_tag_offset > slice.len() {
            return Err(ForgeError::InvalidIcc);
        }
        let tag = &slice[entry..last_tag_offset];
        if tag.len() < 20 {
            return Err(ForgeError::InvalidIcc);
        }
        let q15_16_x = i32::from_be_bytes([tag[8], tag[9], tag[10], tag[11]]);
        let q15_16_y = i32::from_be_bytes([tag[12], tag[13], tag[14], tag[15]]);
        let q15_16_z = i32::from_be_bytes([tag[16], tag[17], tag[18], tag[19]]);
        let x = s15_fixed16_number_to_float(q15_16_x);
        let y = s15_fixed16_number_to_float(q15_16_y);
        let z = s15_fixed16_number_to_float(q15_16_z);
        Ok(Xyz { x, y, z })
    }

    #[inline]
    fn read_cicp_tag(
        slice: &[u8],
        entry: usize,
        tag_size: usize,
    ) -> Result<CicpProfile, ForgeError> {
        let tag_size = if tag_size == 0 { TAG_SIZE } else { tag_size };
        let last_tag_offset = tag_size + entry;
        if last_tag_offset > slice.len() {
            return Err(ForgeError::InvalidIcc);
        }
        let tag = &slice[entry..last_tag_offset];
        if tag.len() < 12 {
            return Err(ForgeError::InvalidIcc);
        }
        let primaries = ColorPrimaries::try_from(tag[8])?;
        let transfer_characteristics = TransferCharacteristics::try_from(tag[9])?;
        let matrix_coefficients = MatrixCoefficients::try_from(tag[10])?;
        Ok(CicpProfile {
            color_primaries: primaries,
            transfer_characteristics,
            matrix_coefficients,
        })
    }

    #[allow(clippy::field_reassign_with_default)]
    pub fn new_from_slice(slice: &[u8]) -> Result<Self, ForgeError> {
        let header = IccHeader::new_from_slice(slice)?;
        let tags_count = header.tag_count as usize;
        if slice.len() >= MAX_PROFILE_SIZE {
            return Err(ForgeError::InvalidIcc);
        }
        if slice.len() < tags_count * TAG_SIZE + size_of::<IccHeader>() {
            return Err(ForgeError::InvalidIcc);
        }
        let tags_slice =
            &slice[size_of::<IccHeader>()..size_of::<IccHeader>() + tags_count * TAG_SIZE];
        let mut profile = ColorProfile::default();
        profile.rendering_intent = RenderingIntent::try_from(header.rendering_intent)?;
        profile.pcs = header.pcs;
        profile.profile_class = header.profile_class;
        profile.color_space = header.data_color_space;
        let color_space = profile.color_space;
        let known_profile_class = profile.profile_class == DISPLAY_DEVICE_PROFILE
            || profile.profile_class == INPUT_DEVICE_PROFILE
            || profile.profile_class == OUTPUT_DEVICE_PROFILE
            || profile.profile_class == COLOR_SPACE_PROFILE;
        if known_profile_class {
            for tag in tags_slice.chunks_exact(TAG_SIZE) {
                let tag_value = u32::from_be_bytes([tag[0], tag[1], tag[2], tag[3]]);
                let tag_entry = u32::from_be_bytes([tag[4], tag[5], tag[6], tag[7]]);
                let tag_size = u32::from_be_bytes([tag[8], tag[9], tag[10], tag[11]]) as usize;
                if tag_value == R_TAG_XYZ && color_space == RGB_SIGNATURE {
                    profile.red_colorant = Self::read_xyz_tag(slice, tag_entry as usize, tag_size)?;
                } else if tag_value == G_TAG_XYZ && color_space == RGB_SIGNATURE {
                    profile.green_colorant =
                        Self::read_xyz_tag(slice, tag_entry as usize, tag_size)?;
                } else if tag_value == B_TAG_XYZ && color_space == RGB_SIGNATURE {
                    profile.blue_colorant =
                        Self::read_xyz_tag(slice, tag_entry as usize, tag_size)?;
                } else if tag_value == CICP_TAG {
                    profile.cicp = Some(Self::read_cicp_tag(slice, tag_entry as usize, tag_size)?);
                } else if tag_value == R_TAG_TRC && color_space == RGB_SIGNATURE {
                    match Self::read_trc_tag(slice, tag_entry as usize, tag_size) {
                        Ok(trc) => profile.red_trc = Some(trc),
                        Err(err) => return Err(err),
                    }
                } else if tag_value == G_TAG_TRC && color_space == RGB_SIGNATURE {
                    match Self::read_trc_tag(slice, tag_entry as usize, tag_size) {
                        Ok(trc) => profile.green_trc = Some(trc),
                        Err(err) => return Err(err),
                    }
                } else if tag_value == B_TAG_TRC && color_space == RGB_SIGNATURE {
                    match Self::read_trc_tag(slice, tag_entry as usize, tag_size) {
                        Ok(trc) => profile.blue_trc = Some(trc),
                        Err(err) => return Err(err),
                    }
                } else if tag_value == K_TAG_TRC && color_space == GRAY_SIGNATURE {
                    match Self::read_trc_tag(slice, tag_entry as usize, tag_size) {
                        Ok(trc) => profile.gray_trc = Some(trc),
                        Err(err) => return Err(err),
                    }
                } else if tag_value == WT_PT_TAG {
                    match Self::read_xyz_tag(slice, tag_entry as usize, tag_size) {
                        Ok(wt) => profile.white_point = Some(wt),
                        Err(err) => return Err(err),
                    }
                } else if tag_value == CHAD_TAG {
                    profile.chromatic_adaptation =
                        Some(Self::read_chad_tag(slice, tag_entry as usize, tag_size)?);
                }
            }
        }

        // if CICP present better check that other values is present

        if let Some(cicp) = profile.cicp {
            if cicp.color_primaries.has_chromacity() {
                let primaries_xy: ChromacityTriple = cicp.color_primaries.try_into()?;
                profile.red_colorant = primaries_xy.red.to_xyz();
                profile.green_colorant = primaries_xy.green.to_xyz();
                profile.blue_colorant = primaries_xy.blue.to_xyz();
                let white_point: Chromacity = cicp.color_primaries.white_point()?;
                profile.white_point = Some(white_point.to_xyz());
            }

            if cicp.transfer_characteristics.has_transfer_curve() {
                if profile.red_trc.is_none() {
                    profile.red_trc = Some(cicp.transfer_characteristics.try_into()?);
                }
                if profile.blue_trc.is_none() {
                    profile.blue_trc = Some(cicp.transfer_characteristics.try_into()?);
                }
                if profile.green_trc.is_none() {
                    profile.green_trc = Some(cicp.transfer_characteristics.try_into()?);
                }
            }
        }

        Ok(profile)
    }

    pub fn build_gray_linearize_table<const N: usize>(&self) -> Option<Box<[f32; N]>> {
        if let Some(trc) = &self.gray_trc {
            return trc.build_linearize_table::<N>();
        }
        None
    }

    pub fn build_r_linearize_table<const N: usize>(&self) -> Option<Box<[f32; N]>> {
        if let Some(trc) = &self.red_trc {
            return trc.build_linearize_table::<N>();
        }
        None
    }

    pub fn build_g_linearize_table<const N: usize>(&self) -> Option<Box<[f32; N]>> {
        if let Some(trc) = &self.green_trc {
            return trc.build_linearize_table::<N>();
        }
        None
    }

    pub fn build_b_linearize_table<const N: usize>(&self) -> Option<Box<[f32; N]>> {
        if let Some(trc) = &self.blue_trc {
            return trc.build_linearize_table::<N>();
        }
        None
    }
}

impl ColorProfile {
    #[inline]
    pub fn colorant_matrix(&self) -> Matrix3f {
        if let Some(cicp) = self.cicp {
            if let ColorPrimaries::Bt709 = cicp.color_primaries {
                return SRGB_MATRIX;
            } else if let ColorPrimaries::Bt2020 = cicp.color_primaries {
                return BT2020_MATRIX;
            } else if let ColorPrimaries::Smpte432 = cicp.color_primaries {
                return DISPLAY_P3_MATRIX;
            }
        }
        Matrix3f {
            v: [
                [
                    self.red_colorant.x,
                    self.green_colorant.x,
                    self.blue_colorant.x,
                ],
                [
                    self.red_colorant.y,
                    self.green_colorant.y,
                    self.blue_colorant.y,
                ],
                [
                    self.red_colorant.z,
                    self.green_colorant.z,
                    self.blue_colorant.z,
                ],
            ],
        }
    }

    pub fn matches_default(&self) -> Option<GamutColorSpace> {
        let colorant = self.colorant_matrix();
        if colorant.test_equality(SRGB_MATRIX) {
            return Some(GamutColorSpace::Srgb);
        } else if colorant.test_equality(BT2020_MATRIX) {
            return Some(GamutColorSpace::Bt2020);
        } else if colorant.test_equality(DISPLAY_P3_MATRIX) {
            return Some(GamutColorSpace::DisplayP3);
        }
        None
    }

    pub(crate) fn update_rgb_colorimetry(
        &mut self,
        white_point: XyY,
        primaries: ChromacityTriple,
    ) -> bool {
        let red_xyz = primaries.red.to_xyz();
        let green_xyz = primaries.green.to_xyz();
        let blue_xyz = primaries.blue.to_xyz();
        let xyz_matrix = Matrix3f {
            v: [
                [red_xyz.x, green_xyz.x, blue_xyz.x],
                [red_xyz.y, green_xyz.y, blue_xyz.y],
                [red_xyz.z, green_xyz.z, blue_xyz.z],
            ],
        };
        let colorants = match self.rgb_to_xyz(xyz_matrix, white_point.to_xyz()) {
            None => return false,
            Some(v) => v,
        };
        let colorants = match adapt_matrix_to_d50(Some(colorants), white_point) {
            Some(colorants) => colorants,
            None => return false,
        };

        /* note: there's a transpose type of operation going on here */
        self.red_colorant.x = colorants.v[0][0];
        self.red_colorant.y = colorants.v[1][0];
        self.red_colorant.z = colorants.v[2][0];
        self.green_colorant.x = colorants.v[0][1];
        self.green_colorant.y = colorants.v[1][1];
        self.green_colorant.z = colorants.v[2][1];
        self.blue_colorant.x = colorants.v[0][2];
        self.blue_colorant.y = colorants.v[1][2];
        self.blue_colorant.z = colorants.v[2][2];
        true
    }

    pub fn rgb_to_xyz(&self, xyz_matrix: Matrix3f, wp: Xyz) -> Option<Matrix3f> {
        let xyz_inverse = xyz_matrix.inverse()?;
        let s = xyz_inverse.mul_vector(wp.to_vector());
        let mut v = xyz_matrix.mul_row_vector::<0>(s);
        v = v.mul_row_vector::<1>(s);
        v = v.mul_row_vector::<2>(s);
        Some(v)
    }

    pub fn rgb_to_xyz_matrix(&self) -> Option<Matrix3f> {
        let xyz_matrix = self.colorant_matrix();
        let white_point = self.white_point.unwrap_or(D50_XYZ);
        self.rgb_to_xyz(xyz_matrix, white_point)
    }

    /// Computes transform matrix RGB -> XYZ -> RGB
    /// Current profile is used as source, other as destination
    pub fn rgb_to_xyz_other_xyz_rgb(&self, dest: &ColorProfile) -> Option<Matrix3f> {
        let mut source = self.rgb_to_xyz_matrix()?;
        let dst = dest.rgb_to_xyz_matrix()?;
        let source_wp = self.white_point.unwrap_or(D50_XYZ);
        let dest_wp = dest.white_point.unwrap_or(D50_XYZ);
        if source_wp != dest_wp {
            source = adapt_matrix_to_illuminant_xyz(Some(source), source_wp, dest_wp)?;
        }
        let dest_inverse = dst.inverse()?;
        Some(dest_inverse.mat_mul(source))
    }
}

/* from lcms: cmsWhitePointFromTemp */
/* tempK must be >= 4000. and <= 25000.
 * Invalid values of tempK will return
 * (x,y,Y) = (-1.0, -1.0, -1.0)
 * similar to argyll: icx_DTEMP2XYZ() */
fn white_point_from_temperature(temp_k: i32) -> XyY {
    let mut white_point = XyY {
        x: 0.,
        y: 0.,
        yb: 0.,
    };
    // No optimization provided.
    let temp_k = temp_k as f64; // Square
    let temp_k2 = temp_k * temp_k; // Cube
    let temp_k3 = temp_k2 * temp_k;
    // For correlated color temperature (T) between 4000K and 7000K:
    let x = if (4000.0..=7000.0).contains(&temp_k) {
        -4.6070 * (1E9 / temp_k3) + 2.9678 * (1E6 / temp_k2) + 0.09911 * (1E3 / temp_k) + 0.244063
    } else if temp_k > 7000.0 && temp_k <= 25000.0 {
        -2.0064 * (1E9 / temp_k3) + 1.9018 * (1E6 / temp_k2) + 0.24748 * (1E3 / temp_k) + 0.237040
    } else {
        // or for correlated color temperature (T) between 7000K and 25000K:
        // Invalid tempK
        white_point.x = -1.0;
        white_point.y = -1.0;
        white_point.yb = -1.0;
        debug_assert!(false, "invalid temp");
        return white_point;
    };
    // Obtain y(x)
    let y = -3.000 * (x * x) + 2.870 * x - 0.275;
    // wave factors (not used, but here for futures extensions)
    // let M1 = (-1.3515 - 1.7703*x + 5.9114 *y)/(0.0241 + 0.2562*x - 0.7341*y);
    // let M2 = (0.0300 - 31.4424*x + 30.0717*y)/(0.0241 + 0.2562*x - 0.7341*y);
    // Fill white_point struct
    white_point.x = x as f32;
    white_point.y = y as f32;
    white_point.yb = 1.0;
    white_point
}

pub fn white_point_srgb() -> XyY {
    white_point_from_temperature(6504)
}

impl ColorProfile {
    pub fn new_srgb() -> ColorProfile {
        let primaries = ChromacityTriple::try_from(ColorPrimaries::Bt709).unwrap();
        let white_point = white_point_srgb();
        let mut profile = ColorProfile::default();
        profile.update_rgb_colorimetry(white_point, primaries);

        let curve = Trc::Parametric(vec![2.4, 1. / 1.055, 0.055 / 1.055, 1. / 12.92, 0.04045]);
        profile.red_trc = Some(curve.clone());
        profile.blue_trc = Some(curve.clone());
        profile.green_trc = Some(curve);
        profile.profile_class = DISPLAY_DEVICE_PROFILE;
        profile.rendering_intent = RenderingIntent::Perceptual;
        profile.color_space = RGB_SIGNATURE;
        profile.pcs = XYZ_TYPE;
        profile
    }

    pub fn new_display_p3() -> ColorProfile {
        let primaries = ChromacityTriple::try_from(ColorPrimaries::Smpte432).unwrap();
        let white_point = white_point_srgb();
        let mut profile = ColorProfile::default();
        profile.update_rgb_colorimetry(white_point, primaries);

        let curve = Trc::Parametric(vec![2.4, 1. / 1.055, 0.055 / 1.055, 1. / 12.92, 0.04045]);
        profile.red_trc = Some(curve.clone());
        profile.blue_trc = Some(curve.clone());
        profile.green_trc = Some(curve);
        profile.profile_class = DISPLAY_DEVICE_PROFILE;
        profile.rendering_intent = RenderingIntent::Perceptual;
        profile.color_space = RGB_SIGNATURE;
        profile.pcs = XYZ_TYPE;
        profile
    }

    pub fn new_bt2020() -> ColorProfile {
        let primaries = ChromacityTriple::try_from(ColorPrimaries::Bt2020).unwrap();
        let white_point = white_point_srgb();
        let mut profile = ColorProfile::default();
        profile.update_rgb_colorimetry(white_point, primaries);

        let curve = Trc::Parametric(vec![2.4, 1. / 1.055, 0.055 / 1.055, 1. / 12.92, 0.04045]);
        profile.red_trc = Some(curve.clone());
        profile.blue_trc = Some(curve.clone());
        profile.green_trc = Some(curve);
        profile.profile_class = DISPLAY_DEVICE_PROFILE;
        profile.rendering_intent = RenderingIntent::Perceptual;
        profile.color_space = RGB_SIGNATURE;
        profile.pcs = XYZ_TYPE;
        profile
    }
}

pub(crate) fn make_icc_transform(
    input_color_space: GamutColorSpace,
    output_color_space: GamutColorSpace,
) -> Option<Matrix3f> {
    let d0 = gamut_to_xyz(output_color_space.primaries_xy(), Chromacity::D65)?;
    let dest = d0.inverse()?;
    let src = gamut_to_xyz(input_color_space.primaries_xy(), Chromacity::D65)?;
    let product = dest.mat_mul(src);
    Some(product)
}
