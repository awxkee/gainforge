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
use crate::err::ForgeError;
use crate::gamma::trc_from_cicp;
use crate::mappers::{
    AcesToneMapper, ClampToneMapper, ExtendedReinhardToneMapper, FilmicToneMapper,
    Rec2408ToneMapper, ReinhardJodieToneMapper, ReinhardToneMapper, ToneMap,
};
use crate::mlaf::mlaf;
use crate::spline::{create_spline, SplineToneMapper};
use crate::{m_clamp, ToneMappingMethod};
use moxcms::{
    gamut_clip_preserve_chroma, CmsError, ColorProfile, InPlaceStage, Jzazbz, Matrix3f, Oklab, Rgb,
    Yrg,
};
use num_traits::AsPrimitive;
use std::fmt::Debug;

/// Defines gamut clipping mode
#[derive(Debug, Default, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum GamutClipping {
    #[default]
    NoClip,
    Clip,
}

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub enum MappingColorSpace {
    /// Scene linear light RGB colorspace.
    ///
    /// Some description what to expect:
    /// - Does not produce colours outside the destination gamut.
    /// - Generally tends to produce “natural” looking colours, although saturation of extreme colours
    ///   is reduced substantially and some colours may be changed in hue.
    /// - Problematic when it is desired to retain bright saturated colours, such as coloured lights at
    ///   night
    /// - Depending on the amount of compression, the saturation decrease may be excessive, and
    ///   occasionally hue changes can be objectionable.
    Rgb(RgbToneMapperParameters),
    /// Yrg filmic colorspace.
    ///
    /// Slightly better results as in linear RGB, for a little computational cost.
    ///
    /// Some description what to expect:
    /// - Has the potential to produce colours outside the destination gamut, which then require gamut
    ///   mapping
    /// - Preserves chromaticity except for where gamut mapping is applied. Does not produce a
    ///   “natural” looking desaturation of tonally compressed colours.
    /// - Problems can be avoided by using in combination with a variable desaturation and gamut
    ///   mapping algorithm, although such algorithms generally perform best in hue, saturation and
    ///   lightness colour spaces (requiring a colour space change).
    ///
    /// Algorithm here do not perform gamut mapping.
    YRgb(CommonToneMapperParameters),
    /// Oklab perceptual colorspace.
    ///
    /// It exists more for *experiments* how does it look like.
    /// Results are often strange and might not be acceptable.
    /// *Oklab is not really were created for HDR*.
    ///
    /// Some description of what to expect:
    /// - Provides perceptually uniform lightness adjustments.
    /// - Preserves hue better than RGB but can lead to color shifts due to applying aggressive compression.
    /// - Suitable for applications where perceptual lightness control is more important than strict colorimetric accuracy.
    Oklab(CommonToneMapperParameters),
    /// JzAzBz perceptual HDR colorspace.
    ///
    /// Some description of what to expect:
    /// - Designed for HDR workflows, explicitly modeling display-referred luminance.
    /// - Thus, as at the first point often produces the best results.
    /// - Provides better perceptual uniformity than Oklab, particularly in high dynamic range content.
    /// - Preserves hue well, but chroma adjustments may be necessary when mapping between displays with different peak brightness.
    /// - Can help avoid color distortions and oversaturation when adapting HDR content to lower-luminance displays.
    ///
    /// This is very slow computational method.
    /// Content brightness should be specified.
    Jzazbz(JzazbzToneMapperParameters),
}

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct JzazbzToneMapperParameters {
    pub content_brightness: f32,
    pub exposure: f32,
}

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct CommonToneMapperParameters {
    pub exposure: f32,
}

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct RgbToneMapperParameters {
    pub exposure: f32,
    pub gamut_clipping: GamutClipping,
}

impl Default for JzazbzToneMapperParameters {
    fn default() -> Self {
        Self {
            content_brightness: 1000f32,
            exposure: 1.0f32,
        }
    }
}

impl Default for CommonToneMapperParameters {
    fn default() -> Self {
        Self { exposure: 1.0f32 }
    }
}

impl Default for RgbToneMapperParameters {
    fn default() -> Self {
        Self {
            exposure: 1.0f32,
            gamut_clipping: GamutClipping::default(),
        }
    }
}

pub type SyncToneMapper8Bit = dyn ToneMapper<u8> + Send + Sync;
pub type SyncToneMapper16Bit = dyn ToneMapper<u16> + Send + Sync;

type SyncToneMap = dyn ToneMap + Send + Sync;

struct MatrixStage<const CN: usize> {
    pub(crate) gamut_color_conversion: Matrix3f,
}

impl<const CN: usize> InPlaceStage for MatrixStage<CN> {
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        let c = self.gamut_color_conversion;
        for chunk in dst.chunks_exact_mut(CN) {
            let r = mlaf(
                mlaf(chunk[0] * c.v[0][0], chunk[1], c.v[0][1]),
                chunk[2],
                c.v[0][2],
            );
            let g = mlaf(
                mlaf(chunk[0] * c.v[1][0], chunk[1], c.v[1][1]),
                chunk[2],
                c.v[1][2],
            );
            let b = mlaf(
                mlaf(chunk[0] * c.v[2][0], chunk[1], c.v[2][1]),
                chunk[2],
                c.v[2][2],
            );

            chunk[0] = m_clamp(r, 0.0, 1.0);
            chunk[1] = m_clamp(g, 0.0, 1.0);
            chunk[2] = m_clamp(b, 0.0, 1.0);
        }
        Ok(())
    }
}

struct MatrixGamutClipping<const CN: usize> {
    pub(crate) gamut_color_conversion: Matrix3f,
}

impl<const CN: usize> InPlaceStage for MatrixGamutClipping<CN> {
    fn transform(&self, dst: &mut [f32]) -> Result<(), CmsError> {
        let c = self.gamut_color_conversion;
        for chunk in dst.chunks_exact_mut(CN) {
            let r = mlaf(
                mlaf(chunk[0] * c.v[0][0], chunk[1], c.v[0][1]),
                chunk[2],
                c.v[0][2],
            );
            let g = mlaf(
                mlaf(chunk[0] * c.v[1][0], chunk[1], c.v[1][1]),
                chunk[2],
                c.v[1][2],
            );
            let b = mlaf(
                mlaf(chunk[0] * c.v[2][0], chunk[1], c.v[2][1]),
                chunk[2],
                c.v[2][2],
            );

            let mut rgb = Rgb::new(r, g, b);
            if rgb.is_out_of_gamut() {
                rgb = gamut_clip_preserve_chroma(rgb);
                chunk[0] = m_clamp(rgb.r, 0.0, 1.0);
                chunk[1] = m_clamp(rgb.g, 0.0, 1.0);
                chunk[2] = m_clamp(rgb.b, 0.0, 1.0);
            } else {
                chunk[0] = m_clamp(r, 0.0, 1.0);
                chunk[1] = m_clamp(g, 0.0, 1.0);
                chunk[2] = m_clamp(b, 0.0, 1.0);
            }
        }
        Ok(())
    }
}

pub(crate) struct ToneMapperImpl<T: Copy, const N: usize, const CN: usize, const GAMMA_SIZE: usize>
{
    pub(crate) linear_map_r: Box<[f32; N]>,
    pub(crate) linear_map_g: Box<[f32; N]>,
    pub(crate) linear_map_b: Box<[f32; N]>,
    pub(crate) gamma_map_r: Box<[T; 65536]>,
    pub(crate) gamma_map_g: Box<[T; 65536]>,
    pub(crate) gamma_map_b: Box<[T; 65536]>,
    pub(crate) im_stage: Option<Box<dyn InPlaceStage + Sync + Send>>,
    tone_map: Box<SyncToneMap>,
    params: RgbToneMapperParameters,
}

pub(crate) struct ToneMapperImplYrg<
    T: Copy,
    const N: usize,
    const CN: usize,
    const GAMMA_SIZE: usize,
> {
    pub(crate) linear_map_r: Box<[f32; N]>,
    pub(crate) linear_map_g: Box<[f32; N]>,
    pub(crate) linear_map_b: Box<[f32; N]>,
    pub(crate) gamma_map_r: Box<[T; 65536]>,
    pub(crate) gamma_map_g: Box<[T; 65536]>,
    pub(crate) gamma_map_b: Box<[T; 65536]>,
    pub(crate) to_xyz: Matrix3f,
    pub(crate) to_rgb: Matrix3f,
    tone_map: Box<SyncToneMap>,
    parameters: CommonToneMapperParameters,
}

pub(crate) struct ToneMapperImplOklab<
    T: Copy,
    const N: usize,
    const CN: usize,
    const GAMMA_SIZE: usize,
> {
    pub(crate) linear_map_r: Box<[f32; N]>,
    pub(crate) linear_map_g: Box<[f32; N]>,
    pub(crate) linear_map_b: Box<[f32; N]>,
    pub(crate) gamma_map_r: Box<[T; 65536]>,
    pub(crate) gamma_map_g: Box<[T; 65536]>,
    pub(crate) gamma_map_b: Box<[T; 65536]>,
    tone_map: Box<SyncToneMap>,
    parameters: CommonToneMapperParameters,
}

pub(crate) struct ToneMapperImplJzazbz<
    T: Copy,
    const N: usize,
    const CN: usize,
    const GAMMA_SIZE: usize,
> {
    pub(crate) linear_map_r: Box<[f32; N]>,
    pub(crate) linear_map_g: Box<[f32; N]>,
    pub(crate) linear_map_b: Box<[f32; N]>,
    pub(crate) gamma_map_r: Box<[T; 65536]>,
    pub(crate) gamma_map_g: Box<[T; 65536]>,
    pub(crate) gamma_map_b: Box<[T; 65536]>,
    pub(crate) to_xyz: Matrix3f,
    pub(crate) to_rgb: Matrix3f,
    tone_map: Box<SyncToneMap>,
    parameters: JzazbzToneMapperParameters,
}

pub trait ToneMapper<T: Copy + Default + Debug> {
    /// Tone map image lane.
    ///
    /// Lane length must be multiple of channels.
    /// Lane length must match.
    fn tonemap_lane(&self, src: &[T], dst: &mut [T]) -> Result<(), ForgeError>;

    /// Tone map lane whereas content been linearized.
    ///
    /// Lane length must be multiple of channels.
    fn tonemap_linearized_lane(&self, in_place: &mut [f32]) -> Result<(), ForgeError>;
}

impl<
        T: Copy + AsPrimitive<usize> + Clone + Default + Debug,
        const N: usize,
        const CN: usize,
        const GAMMA_SIZE: usize,
    > ToneMapper<T> for ToneMapperImpl<T, N, CN, GAMMA_SIZE>
where
    u32: AsPrimitive<T>,
{
    fn tonemap_lane(&self, src: &[T], dst: &mut [T]) -> Result<(), ForgeError> {
        assert!(CN == 3 || CN == 4);
        if src.len() != dst.len() {
            return Err(ForgeError::LaneSizeMismatch);
        }
        if src.len() % CN != 0 {
            return Err(ForgeError::LaneMultipleOfChannels);
        }
        assert_eq!(src.len(), dst.len());
        let mut linearized_content = vec![0f32; src.len()];
        for (src, dst) in src
            .chunks_exact(CN)
            .zip(linearized_content.chunks_exact_mut(CN))
        {
            dst[0] = self.linear_map_r[src[0].as_()] * self.params.exposure;
            dst[1] = self.linear_map_g[src[1].as_()] * self.params.exposure;
            dst[2] = self.linear_map_b[src[2].as_()] * self.params.exposure;
            if CN == 4 {
                dst[3] = f32::from_bits(src[3].as_() as u32);
            }
        }

        self.tonemap_linearized_lane(&mut linearized_content)?;

        if let Some(c) = &self.im_stage {
            c.transform(&mut linearized_content)
                .map_err(|_| ForgeError::UnknownError)?;
        } else {
            for chunk in linearized_content.chunks_exact_mut(CN) {
                let rgb = Rgb::new(chunk[0], chunk[1], chunk[2]);
                chunk[0] = m_clamp(rgb.r, 0.0, 1.0);
                chunk[1] = m_clamp(rgb.g, 0.0, 1.0);
                chunk[2] = m_clamp(rgb.b, 0.0, 1.0);
            }
        }

        let scale_value = (GAMMA_SIZE - 1) as f32;

        for (dst, src) in dst
            .chunks_exact_mut(CN)
            .zip(linearized_content.chunks_exact(CN))
        {
            let r = mlaf(0.5f32, src[0], scale_value) as u16;
            let g = mlaf(0.5f32, src[1], scale_value) as u16;
            let b = mlaf(0.5f32, src[2], scale_value) as u16;
            dst[0] = self.gamma_map_r[r as usize];
            dst[1] = self.gamma_map_g[g as usize];
            dst[2] = self.gamma_map_b[b as usize];
            if CN == 4 {
                dst[3] = src[3].to_bits().as_();
            }
        }

        Ok(())
    }

    fn tonemap_linearized_lane(&self, in_place: &mut [f32]) -> Result<(), ForgeError> {
        assert!(CN == 3 || CN == 4);
        if in_place.len() % CN != 0 {
            return Err(ForgeError::LaneMultipleOfChannels);
        }
        self.tone_map.process_lane(in_place);
        Ok(())
    }
}

impl<
        T: Copy + AsPrimitive<usize> + Clone + Default + Debug,
        const N: usize,
        const CN: usize,
        const GAMMA_SIZE: usize,
    > ToneMapper<T> for ToneMapperImplYrg<T, N, CN, GAMMA_SIZE>
where
    u32: AsPrimitive<T>,
{
    fn tonemap_lane(&self, src: &[T], dst: &mut [T]) -> Result<(), ForgeError> {
        assert!(CN == 3 || CN == 4);
        if src.len() != dst.len() {
            return Err(ForgeError::LaneSizeMismatch);
        }
        if src.len() % CN != 0 {
            return Err(ForgeError::LaneMultipleOfChannels);
        }
        assert_eq!(src.len(), dst.len());
        let mut linearized_content = vec![0f32; src.len()];
        for (src, dst) in src
            .chunks_exact(CN)
            .zip(linearized_content.chunks_exact_mut(CN))
        {
            let xyz = (Rgb::new(
                self.linear_map_r[src[0].as_()],
                self.linear_map_g[src[1].as_()],
                self.linear_map_b[src[2].as_()],
            ) * self.parameters.exposure)
                .to_xyz(self.to_xyz);
            let yrg = Yrg::from_xyz(xyz);
            dst[0] = yrg.y;
            dst[1] = yrg.r;
            dst[2] = yrg.g;
            if CN == 4 {
                dst[3] = f32::from_bits(src[3].as_() as u32);
            }
        }

        self.tonemap_linearized_lane(&mut linearized_content)?;

        for dst in linearized_content.chunks_exact_mut(CN) {
            let yrg = Yrg::new(dst[0], dst[1], dst[2]);
            let xyz = yrg.to_xyz();
            let rgb = xyz.to_linear_rgb(self.to_rgb);
            dst[0] = m_clamp(rgb.r, 0.0, 1.0);
            dst[1] = m_clamp(rgb.g, 0.0, 1.0);
            dst[2] = m_clamp(rgb.b, 0.0, 1.0);
        }

        let scale_value = (GAMMA_SIZE - 1) as f32;

        for (dst, src) in dst
            .chunks_exact_mut(CN)
            .zip(linearized_content.chunks_exact(CN))
        {
            let r = mlaf(0.5f32, src[0], scale_value) as u16;
            let g = mlaf(0.5f32, src[1], scale_value) as u16;
            let b = mlaf(0.5f32, src[2], scale_value) as u16;
            dst[0] = self.gamma_map_r[r as usize];
            dst[1] = self.gamma_map_g[g as usize];
            dst[2] = self.gamma_map_b[b as usize];
            if CN == 4 {
                dst[3] = src[3].to_bits().as_();
            }
        }

        Ok(())
    }

    fn tonemap_linearized_lane(&self, in_place: &mut [f32]) -> Result<(), ForgeError> {
        assert!(CN == 3 || CN == 4);
        if in_place.len() % CN != 0 {
            return Err(ForgeError::LaneMultipleOfChannels);
        }
        self.tone_map.process_luma_lane(in_place);
        Ok(())
    }
}

impl<
        T: Copy + AsPrimitive<usize> + Clone + Default + Debug,
        const N: usize,
        const CN: usize,
        const GAMMA_SIZE: usize,
    > ToneMapper<T> for ToneMapperImplOklab<T, N, CN, GAMMA_SIZE>
where
    u32: AsPrimitive<T>,
{
    fn tonemap_lane(&self, src: &[T], dst: &mut [T]) -> Result<(), ForgeError> {
        assert!(CN == 3 || CN == 4);
        if src.len() != dst.len() {
            return Err(ForgeError::LaneSizeMismatch);
        }
        if src.len() % CN != 0 {
            return Err(ForgeError::LaneMultipleOfChannels);
        }
        assert_eq!(src.len(), dst.len());
        let mut linearized_content = vec![0f32; src.len()];
        for (src, dst) in src
            .chunks_exact(CN)
            .zip(linearized_content.chunks_exact_mut(CN))
        {
            let xyz = Rgb::new(
                self.linear_map_r[src[0].as_()],
                self.linear_map_g[src[1].as_()],
                self.linear_map_b[src[2].as_()],
            ) * self.parameters.exposure;
            let yrg = Oklab::from_linear_rgb(xyz);
            dst[0] = yrg.l;
            dst[1] = yrg.a;
            dst[2] = yrg.b;
            if CN == 4 {
                dst[3] = f32::from_bits(src[3].as_() as u32);
            }
        }

        self.tonemap_linearized_lane(&mut linearized_content)?;

        for dst in linearized_content.chunks_exact_mut(CN) {
            let yrg = Oklab::new(dst[0], dst[1], dst[2]);
            let rgb = yrg.to_linear_rgb();
            dst[0] = rgb.r;
            dst[1] = rgb.g;
            dst[2] = rgb.b;
        }

        for chunk in linearized_content.chunks_exact_mut(CN) {
            let rgb = Rgb::new(chunk[0], chunk[1], chunk[2]);
            chunk[0] = m_clamp(rgb.r, 0.0, 1.0);
            chunk[1] = m_clamp(rgb.g, 0.0, 1.0);
            chunk[2] = m_clamp(rgb.b, 0.0, 1.0);
        }

        let scale_value = (GAMMA_SIZE - 1) as f32;

        for (dst, src) in dst
            .chunks_exact_mut(CN)
            .zip(linearized_content.chunks_exact(CN))
        {
            let r = mlaf(0.5f32, src[0], scale_value) as u16;
            let g = mlaf(0.5f32, src[1], scale_value) as u16;
            let b = mlaf(0.5f32, src[2], scale_value) as u16;
            dst[0] = self.gamma_map_r[r as usize];
            dst[1] = self.gamma_map_g[g as usize];
            dst[2] = self.gamma_map_b[b as usize];
            if CN == 4 {
                dst[3] = src[3].to_bits().as_();
            }
        }

        Ok(())
    }

    fn tonemap_linearized_lane(&self, in_place: &mut [f32]) -> Result<(), ForgeError> {
        assert!(CN == 3 || CN == 4);
        if in_place.len() % CN != 0 {
            return Err(ForgeError::LaneMultipleOfChannels);
        }
        self.tone_map.process_luma_lane(in_place);
        Ok(())
    }
}

impl<
        T: Copy + AsPrimitive<usize> + Clone + Default + Debug,
        const N: usize,
        const CN: usize,
        const GAMMA_SIZE: usize,
    > ToneMapper<T> for ToneMapperImplJzazbz<T, N, CN, GAMMA_SIZE>
where
    u32: AsPrimitive<T>,
{
    fn tonemap_lane(&self, src: &[T], dst: &mut [T]) -> Result<(), ForgeError> {
        assert!(CN == 3 || CN == 4);
        if src.len() != dst.len() {
            return Err(ForgeError::LaneSizeMismatch);
        }
        if src.len() % CN != 0 {
            return Err(ForgeError::LaneMultipleOfChannels);
        }
        assert_eq!(src.len(), dst.len());
        let mut linearized_content = vec![0f32; src.len()];

        for (src, dst) in src
            .chunks_exact(CN)
            .zip(linearized_content.chunks_exact_mut(CN))
        {
            let xyz = (Rgb::new(
                self.linear_map_r[src[0].as_()],
                self.linear_map_g[src[1].as_()],
                self.linear_map_b[src[2].as_()],
            ) * self.parameters.exposure)
                .to_xyz(self.to_xyz);
            let jab =
                Jzazbz::from_xyz_with_display_luminance(xyz, self.parameters.content_brightness);
            dst[0] = jab.jz;
            dst[1] = jab.az;
            dst[2] = jab.bz;
            if CN == 4 {
                dst[3] = f32::from_bits(src[3].as_() as u32);
            }
        }

        self.tonemap_linearized_lane(&mut linearized_content)?;

        for dst in linearized_content.chunks_exact_mut(CN) {
            let jab = Jzazbz::new(dst[0], dst[1], dst[2]);
            let xyz = jab.to_xyz(self.parameters.content_brightness);
            let rgb = xyz.to_linear_rgb(self.to_rgb);
            dst[0] = m_clamp(rgb.r, 0.0, 1.0);
            dst[1] = m_clamp(rgb.g, 0.0, 1.0);
            dst[2] = m_clamp(rgb.b, 0.0, 1.0);
        }

        let scale_value = (GAMMA_SIZE - 1) as f32;

        for (dst, src) in dst
            .chunks_exact_mut(CN)
            .zip(linearized_content.chunks_exact(CN))
        {
            let r = mlaf(0.5f32, src[0], scale_value) as u16;
            let g = mlaf(0.5f32, src[1], scale_value) as u16;
            let b = mlaf(0.5f32, src[2], scale_value) as u16;
            dst[0] = self.gamma_map_r[r as usize];
            dst[1] = self.gamma_map_g[g as usize];
            dst[2] = self.gamma_map_b[b as usize];
            if CN == 4 {
                dst[3] = src[3].to_bits().as_();
            }
        }

        Ok(())
    }

    fn tonemap_linearized_lane(&self, in_place: &mut [f32]) -> Result<(), ForgeError> {
        assert!(CN == 3 || CN == 4);
        if in_place.len() % CN != 0 {
            return Err(ForgeError::LaneMultipleOfChannels);
        }
        self.tone_map.process_luma_lane(in_place);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialOrd, PartialEq)]
pub struct GainHdrMetadata {
    pub content_max_brightness: f32,
    pub display_max_brightness: f32,
}

impl Default for GainHdrMetadata {
    fn default() -> Self {
        Self {
            content_max_brightness: 1000f32,
            display_max_brightness: 250f32,
        }
    }
}

impl GainHdrMetadata {
    pub fn new(content_max_brightness: f32, display_max_brightness: f32) -> Self {
        Self {
            content_max_brightness,
            display_max_brightness,
        }
    }
}

fn make_icc_transform(
    input_color_space: &ColorProfile,
    output_color_space: &ColorProfile,
) -> Matrix3f {
    input_color_space
        .transform_matrix(output_color_space)
        .unwrap_or(Matrix3f::IDENTITY)
}

fn create_tone_mapper_u8<const CN: usize>(
    input_color_space: &ColorProfile,
    output_color_space: &ColorProfile,
    method: ToneMappingMethod,
    working_color_space: MappingColorSpace,
) -> Result<Box<SyncToneMapper8Bit>, ForgeError> {
    let (linear_table_r, linear_table_g, linear_table_b);
    if let Some(trc) = input_color_space
        .cicp
        .and_then(|x| trc_from_cicp(x.transfer_characteristics))
    {
        linear_table_r = trc.generate_linear_table_u8();
        linear_table_g = linear_table_r.clone();
        linear_table_b = linear_table_g.clone();
    } else {
        linear_table_r = input_color_space
            .build_r_linearize_table::<256, 8>(true)
            .map_err(|_| ForgeError::InvalidTrcCurve)?;
        linear_table_g = input_color_space
            .build_g_linearize_table::<256, 8>(true)
            .map_err(|_| ForgeError::InvalidTrcCurve)?;
        linear_table_b = input_color_space
            .build_b_linearize_table::<256, 8>(true)
            .map_err(|_| ForgeError::InvalidTrcCurve)?;
    }
    let (gamma_table_r, gamma_table_g, gamma_table_b);
    if let Some(trc) = output_color_space
        .cicp
        .and_then(|x| trc_from_cicp(x.transfer_characteristics))
    {
        gamma_table_r = trc.generate_gamma_table_u8();
        gamma_table_g = gamma_table_r.clone();
        gamma_table_b = gamma_table_g.clone();
    } else {
        gamma_table_r = output_color_space
            .build_gamma_table::<u8, 65536, 8192, 8>(&output_color_space.red_trc, true)
            .unwrap();
        gamma_table_g = output_color_space
            .build_gamma_table::<u8, 65536, 8192, 8>(&output_color_space.green_trc, true)
            .unwrap();
        gamma_table_b = output_color_space
            .build_gamma_table::<u8, 65536, 8192, 8>(&output_color_space.blue_trc, true)
            .unwrap();
    }
    let conversion = make_icc_transform(input_color_space, output_color_space);

    let tone_map = make_mapper::<CN>(input_color_space, method);

    match working_color_space {
        MappingColorSpace::Rgb(params) => {
            let im_stage: Box<dyn InPlaceStage + Send + Sync> =
                if params.gamut_clipping == GamutClipping::Clip {
                    Box::new(MatrixGamutClipping::<CN> {
                        gamut_color_conversion: conversion,
                    })
                } else {
                    Box::new(MatrixStage::<CN> {
                        gamut_color_conversion: conversion,
                    })
                };
            Ok(Box::new(ToneMapperImpl::<u8, 256, CN, 8192> {
                linear_map_r: linear_table_r,
                linear_map_g: linear_table_g,
                linear_map_b: linear_table_b,
                gamma_map_r: gamma_table_r,
                gamma_map_b: gamma_table_g,
                gamma_map_g: gamma_table_b,
                im_stage: Some(im_stage),
                tone_map,
                params,
            }))
        }
        MappingColorSpace::YRgb(params) => Ok(Box::new(ToneMapperImplYrg::<u8, 256, CN, 8192> {
            linear_map_r: linear_table_r,
            linear_map_g: linear_table_g,
            linear_map_b: linear_table_b,
            gamma_map_r: gamma_table_r,
            gamma_map_b: gamma_table_g,
            gamma_map_g: gamma_table_b,
            to_xyz: input_color_space
                .rgb_to_xyz_matrix()
                .unwrap_or(Matrix3f::IDENTITY),
            to_rgb: output_color_space
                .rgb_to_xyz_matrix()
                .and_then(|x| x.inverse())
                .unwrap_or(Matrix3f::IDENTITY),
            tone_map,
            parameters: params,
        })),
        MappingColorSpace::Oklab(params) => {
            Ok(Box::new(ToneMapperImplOklab::<u8, 256, CN, 8192> {
                linear_map_r: linear_table_r,
                linear_map_g: linear_table_g,
                linear_map_b: linear_table_b,
                gamma_map_r: gamma_table_r,
                gamma_map_b: gamma_table_g,
                gamma_map_g: gamma_table_b,
                tone_map,
                parameters: params,
            }))
        }
        MappingColorSpace::Jzazbz(brightness) => {
            Ok(Box::new(ToneMapperImplJzazbz::<u8, 256, CN, 8192> {
                linear_map_r: linear_table_r,
                linear_map_g: linear_table_g,
                linear_map_b: linear_table_b,
                gamma_map_r: gamma_table_r,
                gamma_map_b: gamma_table_g,
                gamma_map_g: gamma_table_b,
                to_xyz: input_color_space
                    .rgb_to_xyz_matrix()
                    .unwrap_or(Matrix3f::IDENTITY),
                to_rgb: output_color_space
                    .rgb_to_xyz_matrix()
                    .and_then(|x| x.inverse())
                    .unwrap_or(Matrix3f::IDENTITY),
                tone_map,
                parameters: brightness,
            }))
        }
    }
}

fn create_tone_mapper_u16<const CN: usize, const BIT_DEPTH: usize>(
    input_color_space: &ColorProfile,
    output_color_space: &ColorProfile,
    method: ToneMappingMethod,
    working_color_space: MappingColorSpace,
) -> Result<Box<SyncToneMapper16Bit>, ForgeError> {
    assert!((8..=16).contains(&BIT_DEPTH));
    let (linear_table_r, linear_table_g, linear_table_b);
    if let Some(trc) = input_color_space
        .cicp
        .and_then(|x| trc_from_cicp(x.transfer_characteristics))
    {
        linear_table_r = trc.generate_linear_table_u16(BIT_DEPTH);
        linear_table_g = linear_table_r.clone();
        linear_table_b = linear_table_g.clone();
    } else {
        linear_table_r = input_color_space
            .build_r_linearize_table::<65536, BIT_DEPTH>(true)
            .map_err(|_| ForgeError::InvalidTrcCurve)?;
        linear_table_g = input_color_space
            .build_g_linearize_table::<65536, BIT_DEPTH>(true)
            .map_err(|_| ForgeError::InvalidTrcCurve)?;
        linear_table_b = input_color_space
            .build_b_linearize_table::<65536, BIT_DEPTH>(true)
            .map_err(|_| ForgeError::InvalidTrcCurve)?;
    }
    let (gamma_table_r, gamma_table_g, gamma_table_b);
    if let Some(trc) = output_color_space
        .cicp
        .and_then(|x| trc_from_cicp(x.transfer_characteristics))
    {
        gamma_table_r = trc.generate_gamma_table_u16(BIT_DEPTH);
        gamma_table_g = gamma_table_r.clone();
        gamma_table_b = gamma_table_g.clone();
    } else {
        gamma_table_r = output_color_space
            .build_gamma_table::<u16, 65536, 65536, BIT_DEPTH>(&output_color_space.red_trc, true)
            .unwrap();
        gamma_table_g = output_color_space
            .build_gamma_table::<u16, 65536, 65536, BIT_DEPTH>(&output_color_space.green_trc, true)
            .unwrap();
        gamma_table_b = output_color_space
            .build_gamma_table::<u16, 65536, 65536, BIT_DEPTH>(&output_color_space.blue_trc, true)
            .unwrap();
    }
    let conversion = make_icc_transform(input_color_space, output_color_space);

    let tone_map = make_mapper::<CN>(input_color_space, method);

    match working_color_space {
        MappingColorSpace::Rgb(params) => {
            let im_stage: Box<dyn InPlaceStage + Send + Sync> =
                if params.gamut_clipping == GamutClipping::Clip {
                    Box::new(MatrixGamutClipping::<CN> {
                        gamut_color_conversion: conversion,
                    })
                } else {
                    Box::new(MatrixStage::<CN> {
                        gamut_color_conversion: conversion,
                    })
                };
            Ok(Box::new(ToneMapperImpl::<u16, 65536, CN, 65536> {
                linear_map_r: linear_table_r,
                linear_map_g: linear_table_g,
                linear_map_b: linear_table_b,
                gamma_map_r: gamma_table_r,
                gamma_map_b: gamma_table_g,
                gamma_map_g: gamma_table_b,
                im_stage: Some(im_stage),
                tone_map,
                params,
            }))
        }
        MappingColorSpace::YRgb(params) => {
            Ok(Box::new(ToneMapperImplYrg::<u16, 65536, CN, 65536> {
                linear_map_r: linear_table_r,
                linear_map_g: linear_table_g,
                linear_map_b: linear_table_b,
                gamma_map_r: gamma_table_r,
                gamma_map_b: gamma_table_g,
                gamma_map_g: gamma_table_b,
                to_xyz: input_color_space
                    .rgb_to_xyz_matrix()
                    .unwrap_or(Matrix3f::IDENTITY),
                to_rgb: output_color_space
                    .rgb_to_xyz_matrix()
                    .and_then(|x| x.inverse())
                    .unwrap_or(Matrix3f::IDENTITY),
                tone_map,
                parameters: params,
            }))
        }
        MappingColorSpace::Oklab(params) => {
            Ok(Box::new(ToneMapperImplOklab::<u16, 65536, CN, 65536> {
                linear_map_r: linear_table_r,
                linear_map_g: linear_table_g,
                linear_map_b: linear_table_b,
                gamma_map_r: gamma_table_r,
                gamma_map_b: gamma_table_g,
                gamma_map_g: gamma_table_b,
                tone_map,
                parameters: params,
            }))
        }
        MappingColorSpace::Jzazbz(brightness) => {
            Ok(Box::new(ToneMapperImplJzazbz::<u16, 65536, CN, 65536> {
                linear_map_r: linear_table_r,
                linear_map_g: linear_table_g,
                linear_map_b: linear_table_b,
                gamma_map_r: gamma_table_r,
                gamma_map_b: gamma_table_g,
                gamma_map_g: gamma_table_b,
                to_xyz: input_color_space
                    .rgb_to_xyz_matrix()
                    .unwrap_or(Matrix3f::IDENTITY),
                to_rgb: output_color_space
                    .rgb_to_xyz_matrix()
                    .and_then(|x| x.inverse())
                    .unwrap_or(Matrix3f::IDENTITY),
                tone_map,
                parameters: brightness,
            }))
        }
    }
}

fn make_mapper<const CN: usize>(
    input_color_space: &ColorProfile,
    method: ToneMappingMethod,
) -> Box<SyncToneMap> {
    let primaries = input_color_space
        .rgb_to_xyz_matrix()
        .unwrap_or(Matrix3f::IDENTITY);
    let luma_primaries: [f32; 3] = primaries.v[1];
    let tone_map: Box<SyncToneMap> = match method {
        ToneMappingMethod::Rec2408(data) => Box::new(Rec2408ToneMapper::<CN>::new(
            data.content_max_brightness,
            data.display_max_brightness,
            203f32,
            luma_primaries,
        )),
        ToneMappingMethod::Filmic => Box::new(FilmicToneMapper::<CN>::default()),
        ToneMappingMethod::Aces => Box::new(AcesToneMapper::<CN>::default()),
        ToneMappingMethod::ExtendedReinhard => Box::new(ExtendedReinhardToneMapper::<CN> {
            primaries: luma_primaries,
        }),
        ToneMappingMethod::ReinhardJodie => Box::new(ReinhardJodieToneMapper::<CN> {
            primaries: luma_primaries,
        }),
        ToneMappingMethod::Reinhard => Box::new(ReinhardToneMapper::<CN>::default()),
        ToneMappingMethod::Clamp => Box::new(ClampToneMapper::<CN>::default()),
        ToneMappingMethod::FilmicSpline(params) => {
            let spline = create_spline(params);
            Box::new(SplineToneMapper::<CN> {
                spline,
                primaries: luma_primaries,
            })
        }
    };
    tone_map
}

macro_rules! define8 {
    ($method: ident, $cn: expr, $name: expr) => {
        #[doc = concat!("Creates an ", $name," tone mapper. \
        \
        ICC profile do expect that for HDR tone management `CICP` tag will be used. \
        Tone mapper will search for `CICP` in [ColorProfile] and if there is some value, \
        then transfer function from `CICP` will be used. \
        Otherwise, we will interpolate ICC tone reproduction LUT tables.")]
        pub fn $method(
            input_color_space: &ColorProfile,
            output_color_space: &ColorProfile,
            method: ToneMappingMethod,
            working_color_space: MappingColorSpace,
        ) -> Result<Box<SyncToneMapper8Bit>, ForgeError> {
            create_tone_mapper_u8::<$cn>(
                input_color_space,
                output_color_space,
                method,
                working_color_space,
            )
        }
    };
}

define8!(create_tone_mapper_rgb, 3, "RGB8");
define8!(create_tone_mapper_rgba, 4, "RGBA8");

macro_rules! define16 {
    ($method: ident, $cn: expr, $bp: expr, $name: expr) => {
        #[doc = concat!("Creates an ", $name," tone mapper. \
        \
        ICC profile do expect that for HDR tone management `CICP` tag will be used. \
        Tone mapper will search for `CICP` in [ColorProfile] and if there is some value, \
        then transfer function from `CICP` will be used. \
        Otherwise, we will interpolate ICC tone reproduction LUT tables.")]
        pub fn $method(
            input_color_space: &ColorProfile,
            output_color_space: &ColorProfile,
            method: ToneMappingMethod,
            working_color_space: MappingColorSpace,
        ) -> Result<Box<SyncToneMapper16Bit>, ForgeError> {
            create_tone_mapper_u16::<$cn, $bp>(
                input_color_space,
                output_color_space,
                method,
                working_color_space,
            )
        }
    };
}

define16!(create_tone_mapper_rgb10, 3, 10, "RGB10");
define16!(create_tone_mapper_rgba10, 4, 10, "RGBA10");

define16!(create_tone_mapper_rgb12, 3, 12, "RGB12");
define16!(create_tone_mapper_rgba12, 4, 12, "RGBA12");

define16!(create_tone_mapper_rgb14, 3, 14, "RGB14");
define16!(create_tone_mapper_rgba14, 4, 14, "RGBA14");

define16!(create_tone_mapper_rgb16, 3, 16, "RGB16");
define16!(create_tone_mapper_rgba16, 4, 16, "RGBA16");
