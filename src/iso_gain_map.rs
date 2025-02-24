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
use crate::mappers::Rgb;
use crate::mlaf::mlaf;

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

impl IsoGainMap {
    /// Converts a `Vec<u8>` into an `IsoGainMap` struct
    #[allow(clippy::field_reassign_with_default)]
    pub fn from_bytes(in_data: &[u8]) -> Result<Self, UhdrErrorInfo> {
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
                    "Unexpected minimum version {}, expected 0",
                    min_version
                )),
            });
        }

        let _ = u16::from_be_bytes(in_data[pos..pos + 2].try_into().unwrap()); // writer version, do nothing with it
        pos += 2;

        let flags = in_data[pos];
        pos += 1;
        let channel_count = if (flags & 0x01) != 0 { 3 } else { 1 };
        if !(channel_count == 1 || channel_count == 3) {
            return Err(UhdrErrorInfo {
                error_code: UhdrErrorCode::UnsupportedFeature,
                detail: Some(format!(
                    "Unexpected channel count {}, expected 1 or 3",
                    channel_count
                )),
            });
        }

        let mut metadata = IsoGainMap::default();
        metadata.use_base_color_space = (flags & 0x02) != 0;
        metadata.backward_direction = (flags & 0x04) != 0;
        let use_common_denominator = (flags & 0x08) != 0;

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
                (self.gain_map_max_n[i] as f64 / self.gain_map_max_d[i] as f64).exp2() as f32;
            to.min_content_boost[i] =
                (self.gain_map_min_n[i] as f64 / self.gain_map_min_d[i] as f64).exp2() as f32;

            to.gamma[i] =
                (self.gain_map_gamma_n[i] as f64 / self.gain_map_gamma_d[i] as f64) as f32;

            // BaseRenditionIsHDR is false
            to.offset_sdr[i] = (self.base_offset_n[i] as f64 / self.base_offset_d[i] as f64) as f32;
            to.offset_hdr[i] =
                (self.alternate_offset_n[i] as f64 / self.alternate_offset_d[i] as f64) as f32;
        }
        to.hdr_capacity_max = (self.alternate_hdr_headroom_n as f64
            / self.alternate_hdr_headroom_d as f64)
            .exp2() as f32;
        to.hdr_capacity_min =
            (self.base_hdr_headroom_n as f64 / self.base_hdr_headroom_d as f64).exp2() as f32;
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
    pub(crate) fn all_channels_are_identical(&self) -> bool {
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
        let min_cb = metadata.min_content_boost[idx].log2();
        let max_cb = metadata.max_content_boost[idx].log2();
        for (i, gain_value) in set.iter_mut().enumerate() {
            let value = i as f32 / (N - 1) as f32;
            let log_boost = min_cb * (1.0f32 - value) + max_cb * value;
            *gain_value = (log_boost * gainmap_weight).exp2();
        }
        set
    }

    pub fn new(metadata: GainMap, gainmap_weight: f32) -> Self {
        assert!(N >= 255, "Received N");
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
            gain = gain.powf(gamma_inv);
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
    pub fn apply_gain(&self, color: Rgb, gain: Rgb) -> Rgb {
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

pub fn make_gainmap_weight(gain_map: GainMap, display_boost: f32) -> f32 {
    let input_boost = display_boost.max(1f32);
    let gainmap_weight = (input_boost.log2() - gain_map.hdr_capacity_min.log2())
        / (gain_map.hdr_capacity_max.log2() - gain_map.hdr_capacity_min.log2());
    gainmap_weight.max(0.0f32).min(1.0f32)
}
