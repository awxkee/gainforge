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
use crate::cms::GamutColorSpace;
use crate::err::ForgeError;
use crate::gamma::HdrTransferFunction;
use crate::mappers::{
    AcesToneMapper, AluToneMapper, ClampToneMapper, DragoToneMapper, ExtendedReinhardToneMapper,
    FilmicToneMapper, Rec2408ToneMapper, ReinhardJodieToneMapper, ReinhardToneMapper, ToneMap,
};
use crate::mlaf::mlaf;
use crate::spline::{create_spline, SplineToneMapper};
use crate::{m_clamp, GainImage, GainImageMut, ToneMappingMethod, TransferFunction};
use moxcms::{gamut_clip_preserve_chroma, CmsError, ColorProfile, InPlaceStage, Matrix3f, Rgb};
use num_traits::AsPrimitive;
use std::fmt::Debug;

/// Defines gamut clipping mode
#[derive(Debug, Default, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum GamutClipping {
    #[default]
    NoClip,
    Clip,
}

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
    pub(crate) linear_map: Box<[f32; N]>,
    pub(crate) gamma_map: Box<[T; 65636]>,
    pub(crate) im_stage: Option<Box<dyn InPlaceStage + Sync + Send>>,
    tone_map: Box<SyncToneMap>,
}

pub trait ToneMapper<T: Copy + Default + Debug, const N: usize> {
    /// Tone map image lane.
    ///
    /// Lane length must be multiple of channels.
    /// Lane length must match.
    fn tonemap_lane(&self, src: &[T], dst: &mut [T]) -> Result<(), ForgeError>;

    /// Tone map lane whereas content been linearized.
    ///
    /// Lane length must be multiple of channels.
    fn tonemap_linearized_lane(&self, in_place: &mut [f32]) -> Result<(), ForgeError>;

    /// Tone map whole image.
    ///
    /// For local tone mapper it is just passthrough to [ToneMapper::tonemap_lane].
    /// For global tone mappers this call is required.
    fn tonemap_image(
        &self,
        src: &GainImage<'_, T, N>,
        dst: &mut GainImageMut<'_, T, N>,
    ) -> Result<(), ForgeError>;
}

impl<
        T: Copy + AsPrimitive<usize> + Clone + Default + Debug,
        const N: usize,
        const CN: usize,
        const GAMMA_SIZE: usize,
    > ToneMapper<T, CN> for ToneMapperImpl<T, N, CN, GAMMA_SIZE>
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
            dst[0] = self.linear_map[src[0].as_()];
            dst[1] = self.linear_map[src[1].as_()];
            dst[2] = self.linear_map[src[2].as_()];
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
            dst[0] = self.gamma_map[r as usize];
            dst[1] = self.gamma_map[g as usize];
            dst[2] = self.gamma_map[b as usize];
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

    fn tonemap_image(
        &self,
        src: &GainImage<'_, T, CN>,
        dst: &mut GainImageMut<'_, T, CN>,
    ) -> Result<(), ForgeError> {
        assert!(CN == 3 || CN == 4);
        src.check_layout()?;
        dst.check_layout()?;
        src.size_matches_mut(dst)?;
        let dst_stride = dst.row_stride();
        let dst_width = dst.width;
        for (src_row, dst_row) in src
            .data
            .as_ref()
            .chunks_exact(src.row_stride())
            .zip(dst.data.borrow_mut().chunks_exact_mut(dst_stride))
        {
            let src = &src_row[..src.width * CN];
            let dst = &mut dst_row[..dst_width * CN];
            self.tonemap_lane(src, dst)?;
        }
        Ok(())
    }
}

pub(crate) struct DragoToneMapperImpl<
    T: Copy,
    const N: usize,
    const CN: usize,
    const GAMMA_SIZE: usize,
> {
    pub(crate) linear_map: Box<[f32; N]>,
    pub(crate) gamma_map: Box<[T; 65636]>,
    pub(crate) gamut_clipping: GamutClipping,
    tone_map: DragoToneMapper<CN>,
    primaries: [f32; 3],
}

impl<
        T: Copy + AsPrimitive<usize> + Clone + Default + Debug,
        const N: usize,
        const CN: usize,
        const GAMMA_SIZE: usize,
    > DragoToneMapperImpl<T, N, CN, GAMMA_SIZE>
where
    u32: AsPrimitive<T>,
{
    fn process_image(
        &self,
        src: &[T],
        src_stride: usize,
        dst: &mut [T],
        dst_stride: usize,
        width: usize,
        height: usize,
    ) -> Result<(), ForgeError> {
        let mut linearized_content = vec![0f32; width * height * CN];
        for (src_row, dst_row) in src
            .as_ref()
            .chunks_exact(src_stride)
            .zip(linearized_content.chunks_exact_mut(width * CN))
        {
            let src = &src_row[..width * CN];
            for (chunk, dst) in src.chunks_exact(CN).zip(dst_row.chunks_exact_mut(CN)) {
                dst[0] = self.linear_map[chunk[0].as_()];
                dst[1] = self.linear_map[chunk[1].as_()];
                dst[2] = self.linear_map[chunk[2].as_()];
                if CN == 4 {
                    dst[3] = f32::from_bits(src[3].as_() as u32);
                }
            }
        }
        let mut intensity_map = vec![0f32; width * height];
        let mut scene_max = 0f32;
        for (chunk, dst) in linearized_content
            .chunks_exact(CN)
            .zip(intensity_map.iter_mut())
        {
            let l = chunk[0] * self.primaries[0]
                + chunk[1] * self.primaries[1]
                + chunk[2] * self.primaries[2];
            *dst = l;
            scene_max = l.max(scene_max);
        }

        if scene_max == 0f32 {
            scene_max = 1f32;
        }

        let common_den = (scene_max + 1f32).log10();
        let l_scale = self.tone_map.j_num / common_den;

        if self.gamut_clipping == GamutClipping::Clip {
            for (dst_row, &luma) in linearized_content
                .chunks_exact_mut(CN)
                .zip(intensity_map.iter())
            {
                let luma = self.tone_map.exposure * luma;
                let n1 = (luma + 1f32).log2();
                let d1 = (2f32 + (luma / scene_max).powf(self.tone_map.asymp_power) * 8f32).log2();
                let ld = l_scale * n1 / d1;
                if ld == 0f32 {
                    continue;
                }
                let new_rgb = Rgb::new(
                    dst_row[0] * self.tone_map.exposure / luma * ld,
                    dst_row[1] * self.tone_map.exposure / luma * ld,
                    dst_row[2] * self.tone_map.exposure / luma * ld,
                );
                let clipped_rgb = gamut_clip_preserve_chroma(new_rgb).clamp(0.0, 1.0);
                dst_row[0] = clipped_rgb.r;
                dst_row[1] = clipped_rgb.g;
                dst_row[2] = clipped_rgb.b;
            }
        } else {
            for (dst_row, &luma) in linearized_content
                .chunks_exact_mut(CN)
                .zip(intensity_map.iter())
            {
                let luma = self.tone_map.exposure * luma;
                let n1 = (luma + 1f32).log2();
                let d1 = (2f32 + (luma / scene_max).powf(self.tone_map.asymp_power) * 8f32).log2();
                let ld = l_scale * n1 / d1;
                if ld == 0f32 {
                    continue;
                }
                dst_row[0] = m_clamp(dst_row[0] * self.tone_map.exposure / luma * ld, 0.0, 1.0);
                dst_row[1] = m_clamp(dst_row[1] * self.tone_map.exposure / luma * ld, 0.0, 1.0);
                dst_row[2] = m_clamp(dst_row[2] * self.tone_map.exposure / luma * ld, 0.0, 1.0);
            }
        }

        let scale_value = (GAMMA_SIZE - 1) as f32;

        for (dst, lin) in dst
            .chunks_exact_mut(dst_stride)
            .zip(linearized_content.chunks_exact_mut(width * CN))
        {
            for (dst, src) in dst.chunks_exact_mut(CN).zip(lin.chunks_exact(CN)) {
                let r = mlaf(0.5f32, src[0], scale_value) as u16;
                let g = mlaf(0.5f32, src[1], scale_value) as u16;
                let b = mlaf(0.5f32, src[2], scale_value) as u16;
                dst[0] = self.gamma_map[r as usize];
                dst[1] = self.gamma_map[g as usize];
                dst[2] = self.gamma_map[b as usize];
                if CN == 4 {
                    dst[3] = src[3].to_bits().as_();
                }
            }
        }

        Ok(())
    }
}

impl<
        T: Copy + AsPrimitive<usize> + Clone + Default + Debug,
        const N: usize,
        const CN: usize,
        const GAMMA_SIZE: usize,
    > ToneMapper<T, CN> for DragoToneMapperImpl<T, N, CN, GAMMA_SIZE>
where
    u32: AsPrimitive<T>,
{
    fn tonemap_lane(&self, _: &[T], _: &mut [T]) -> Result<(), ForgeError> {
        unreachable!("You must not use tonemap lane on global tone mapper")
    }

    fn tonemap_linearized_lane(&self, _: &mut [f32]) -> Result<(), ForgeError> {
        unreachable!("You must not use tonemap lane on global tone mapper")
    }

    fn tonemap_image(
        &self,
        src: &GainImage<'_, T, CN>,
        dst: &mut GainImageMut<'_, T, CN>,
    ) -> Result<(), ForgeError> {
        assert!(CN == 3 || CN == 4);
        src.check_layout()?;
        dst.check_layout()?;
        src.size_matches_mut(dst)?;
        let dst_stride = dst.row_stride();
        self.process_image(
            src.data.as_ref(),
            src.row_stride(),
            dst.data.borrow_mut(),
            dst_stride,
            src.width,
            src.height,
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialOrd, PartialEq)]
pub struct GainHDRMetadata {
    pub content_max_brightness: f32,
    pub display_max_brightness: f32,
}

impl Default for GainHDRMetadata {
    fn default() -> Self {
        Self {
            content_max_brightness: 1000f32,
            display_max_brightness: 250f32,
        }
    }
}

impl GainHDRMetadata {
    pub fn new(content_max_brightness: f32, display_max_brightness: f32) -> Self {
        Self {
            content_max_brightness,
            display_max_brightness,
        }
    }
}

fn make_icc_transform(
    input_color_space: GamutColorSpace,
    output_color_space: GamutColorSpace,
) -> (ColorProfile, ColorProfile, Matrix3f) {
    let target_gamut = match output_color_space {
        GamutColorSpace::Srgb => ColorProfile::new_srgb(),
        GamutColorSpace::DisplayP3 => ColorProfile::new_display_p3(),
        GamutColorSpace::Bt2020 => ColorProfile::new_bt2020(),
    };
    let source_gamut = match input_color_space {
        GamutColorSpace::Srgb => ColorProfile::new_srgb(),
        GamutColorSpace::DisplayP3 => ColorProfile::new_display_p3(),
        GamutColorSpace::Bt2020 => ColorProfile::new_bt2020(),
    };
    let matrix = source_gamut
        .transform_matrix(&target_gamut)
        .unwrap_or(Matrix3f::IDENTITY);
    (source_gamut, target_gamut, matrix)
}

fn create_tone_mapper_u8<const CN: usize>(
    hdr_transfer_function: HdrTransferFunction,
    input_color_space: GamutColorSpace,
    display_transfer_function: TransferFunction,
    output_color_space: GamutColorSpace,
    method: ToneMappingMethod,
    gamut_clipping: GamutClipping,
) -> Box<dyn ToneMapper<u8, CN> + Send + Sync> {
    let linear_table = hdr_transfer_function.generate_linear_table_u8();
    let gamma_table = display_transfer_function.generate_gamma_table_u8();
    let conversion = if input_color_space != output_color_space {
        Some(make_icc_transform(input_color_space, output_color_space))
    } else {
        None
    };
    let im_stage: Option<Box<dyn InPlaceStage + Send + Sync>> = conversion.as_ref().map(|x| {
        let c: Box<dyn InPlaceStage + Send + Sync> = if gamut_clipping == GamutClipping::Clip {
            Box::new(MatrixGamutClipping::<CN> {
                gamut_color_conversion: x.2,
            })
        } else {
            Box::new(MatrixStage::<CN> {
                gamut_color_conversion: x.2,
            })
        };
        c
    });
    match method {
        ToneMappingMethod::Drago(params) => {
            let tone_map = DragoToneMapperImpl::<u8, 256, CN, 8192> {
                linear_map: linear_table,
                gamma_map: gamma_table,
                primaries: input_color_space.luma_primaries(),
                gamut_clipping,
                tone_map: DragoToneMapper::new(params),
            };
            Box::new(tone_map)
        }
        _ => {
            let tone_map = make_mapper::<CN>(input_color_space, method);

            Box::new(ToneMapperImpl::<u8, 256, CN, 8192> {
                linear_map: linear_table,
                gamma_map: gamma_table,
                im_stage,
                tone_map,
            })
        }
    }
}

fn create_tone_mapper_u16<const CN: usize>(
    bit_depth: usize,
    hdr_transfer_function: HdrTransferFunction,
    input_color_space: GamutColorSpace,
    display_transfer_function: TransferFunction,
    output_color_space: GamutColorSpace,
    method: ToneMappingMethod,
    gamut_clipping: GamutClipping,
) -> Box<dyn ToneMapper<u16, CN> + Send + Sync> {
    assert!((8..=16).contains(&bit_depth));
    let linear_table = hdr_transfer_function.generate_linear_table_u16(bit_depth);
    let gamma_table = display_transfer_function.generate_gamma_table_u16(bit_depth);
    let conversion = if input_color_space != output_color_space {
        Some(make_icc_transform(input_color_space, output_color_space))
    } else {
        None
    };
    let im_stage: Option<Box<dyn InPlaceStage + Send + Sync>> = conversion.as_ref().map(|x| {
        let c: Box<dyn InPlaceStage + Send + Sync> = if gamut_clipping == GamutClipping::Clip {
            Box::new(MatrixGamutClipping::<CN> {
                gamut_color_conversion: x.2,
            })
        } else {
            Box::new(MatrixStage::<CN> {
                gamut_color_conversion: x.2,
            })
        };
        c
    });
    match method {
        ToneMappingMethod::Drago(params) => {
            let tone_map = DragoToneMapperImpl::<u16, 65536, CN, 65536> {
                linear_map: linear_table,
                gamma_map: gamma_table,
                primaries: input_color_space.luma_primaries(),
                gamut_clipping,
                tone_map: DragoToneMapper::new(params),
            };
            Box::new(tone_map)
        }
        _ => {
            let tone_map = make_mapper::<CN>(input_color_space, method);
            Box::new(ToneMapperImpl::<u16, 65536, CN, 65536> {
                linear_map: linear_table,
                gamma_map: gamma_table,
                im_stage,
                tone_map,
            })
        }
    }
}

fn make_mapper<const CN: usize>(
    input_color_space: GamutColorSpace,
    method: ToneMappingMethod,
) -> Box<SyncToneMap> {
    let tone_map: Box<SyncToneMap> = match method {
        ToneMappingMethod::Rec2408(data) => Box::new(Rec2408ToneMapper::<CN>::new(
            data.content_max_brightness,
            data.display_max_brightness,
            203f32,
            input_color_space.luma_primaries(),
        )),
        ToneMappingMethod::Filmic => Box::new(FilmicToneMapper::<CN>::default()),
        ToneMappingMethod::Aces => Box::new(AcesToneMapper::<CN>::default()),
        ToneMappingMethod::ExtendedReinhard => Box::new(ExtendedReinhardToneMapper::<CN> {
            primaries: input_color_space.luma_primaries(),
        }),
        ToneMappingMethod::ReinhardJodie => Box::new(ReinhardJodieToneMapper::<CN> {
            primaries: input_color_space.luma_primaries(),
        }),
        ToneMappingMethod::Reinhard => Box::new(ReinhardToneMapper::<CN>::default()),
        ToneMappingMethod::Clamp => Box::new(ClampToneMapper::<CN>::default()),
        ToneMappingMethod::Alu => Box::new(AluToneMapper::<CN>::default()),
        ToneMappingMethod::Drago(_) => unreachable!(),
        ToneMappingMethod::FilmicSpline(params) => {
            let spline = create_spline(params);
            Box::new(SplineToneMapper::<CN> {
                spline,
                primaries: input_color_space.luma_primaries(),
            })
        }
    };
    tone_map
}

macro_rules! define8 {
    ($method: ident, $cn: expr, $name: expr) => {
        #[doc = concat!("Creates ", $name," tone mapper

# Arguments

* `content_hdr_metadata`: see [GainHDRMetadata]
* `hdr_transfer_function`: see [HdrTransferFunction]
* `input_color_space`: see [GamutColorSpace]
* `display_transfer_function`: see [TransferFunction]
* `output_color_space`: see [GamutColorSpace]
* `method`: see [ToneMappingMethod]

returns: Box<dyn ToneMapper<u8> + Send+Sync, Global>")]
        pub fn $method(
            hdr_transfer_function: HdrTransferFunction,
            input_color_space: GamutColorSpace,
            display_transfer_function: TransferFunction,
            output_color_space: GamutColorSpace,
            method: ToneMappingMethod,
            gamut_clipping: GamutClipping,
        ) -> Box<dyn ToneMapper<u8, $cn> + Send + Sync> {
            create_tone_mapper_u8::<$cn>(
                hdr_transfer_function,
                input_color_space,
                display_transfer_function,
                output_color_space,
                method,
                gamut_clipping,
            )
        }
    };
}

define8!(create_tone_mapper_rgb, 3, "RGB8");
define8!(create_tone_mapper_rgba, 4, "RGBA8");

macro_rules! define16 {
    ($method: ident, $cn: expr, $bp: expr, $name: expr) => {
        #[doc = concat!("Creates ", $name," tone mapper

# Arguments

* `content_hdr_metadata`: see [GainHDRMetadata]
* `hdr_transfer_function`: see [HdrTransferFunction]
* `input_color_space`: see [GamutColorSpace]
* `display_transfer_function`: see [TransferFunction]
* `output_color_space`: see [GamutColorSpace]
* `method`: see [ToneMappingMethod]

returns: Box<dyn ToneMapper<u8> + Send+Sync, Global>")]
        pub fn $method(
            hdr_transfer_function: HdrTransferFunction,
            input_color_space: GamutColorSpace,
            display_transfer_function: TransferFunction,
            output_color_space: GamutColorSpace,
            method: ToneMappingMethod,
            gamut_clipping: GamutClipping,
        ) -> Box<dyn ToneMapper<u16, $cn> + Send + Sync> {
            create_tone_mapper_u16::<$cn>(
                $bp,
                hdr_transfer_function,
                input_color_space,
                display_transfer_function,
                output_color_space,
                method,
                gamut_clipping,
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
