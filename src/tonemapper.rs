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
use crate::cms::{make_icc_transform, GamutColorSpace, Xyz};
use crate::err::ForgeError;
use crate::gamma::HdrTransferFunction;
use crate::mappers::{
    AcesToneMapper, AluToneMapper, ClampToneMapper, ExtendedReinhardToneMapper, FilmicToneMapper,
    Rec2408ToneMapper, ReinhardJodieToneMapper, ReinhardToneMapper, ToneMap,
};
use crate::mlaf::mlaf;
use crate::{ToneMappingMethod, TransferFunction};
use num_traits::AsPrimitive;

type SyncToneMap = dyn ToneMap + Send + Sync;

pub(crate) struct ToneMapperImpl<T: Copy, const N: usize, const CN: usize> {
    pub(crate) linear_map: Box<[f32; N]>,
    pub(crate) gamma_map: Box<[T; 65636]>,
    pub(crate) gamut_color_conversion: Option<[Xyz; 3]>,
    tone_map: Box<SyncToneMap>,
}

pub trait ToneMapper<T> {
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

impl<T: Copy + AsPrimitive<usize>, const N: usize, const CN: usize> ToneMapper<T>
    for ToneMapperImpl<T, N, CN>
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

        if let Some(c) = self.gamut_color_conversion {
            for chunk in linearized_content.chunks_exact_mut(CN) {
                let r = mlaf(mlaf(chunk[0] * c[0].x, chunk[1], c[0].y), chunk[2], c[0].z);
                let g = mlaf(mlaf(chunk[0] * c[1].x, chunk[1], c[1].y), chunk[2], c[1].z);
                let b = mlaf(mlaf(chunk[0] * c[2].x, chunk[1], c[2].y), chunk[2], c[2].z);
                chunk[0] = r;
                chunk[1] = g;
                chunk[2] = b;
            }
        }

        for (dst, src) in dst
            .chunks_exact_mut(CN)
            .zip(linearized_content.chunks_exact(CN))
        {
            let r = mlaf(0.5f32, src[0], 65535f32) as u16;
            let g = mlaf(0.5f32, src[1], 65535f32) as u16;
            let b = mlaf(0.5f32, src[2], 65535f32) as u16;
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
}

#[derive(Debug, Clone, Copy)]
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

pub type SyncToneMapper8Bit = dyn ToneMapper<u8> + Send + Sync;
pub type SyncToneMapper16Bit = dyn ToneMapper<u16> + Send + Sync;

fn create_tone_mapper_u8<const CN: usize>(
    content_hdr_metadata: GainHDRMetadata,
    hdr_transfer_function: HdrTransferFunction,
    input_color_space: GamutColorSpace,
    display_transfer_function: TransferFunction,
    output_color_space: GamutColorSpace,
    method: ToneMappingMethod,
) -> Box<SyncToneMapper8Bit> {
    let linear_table = hdr_transfer_function.generate_linear_table_u8();
    let gamma_table = display_transfer_function.generate_gamma_table_u8();
    let conversion = if input_color_space != output_color_space {
        Some(make_icc_transform(input_color_space, output_color_space))
    } else {
        None
    };
    let tone_map = make_mapper::<CN>(content_hdr_metadata, input_color_space, method);
    Box::new(ToneMapperImpl::<u8, 256, CN> {
        linear_map: linear_table,
        gamma_map: gamma_table,
        gamut_color_conversion: conversion,
        tone_map,
    })
}

fn create_tone_mapper_u16<const CN: usize>(
    bit_depth: usize,
    content_hdr_metadata: GainHDRMetadata,
    hdr_transfer_function: HdrTransferFunction,
    input_color_space: GamutColorSpace,
    display_transfer_function: TransferFunction,
    output_color_space: GamutColorSpace,
    method: ToneMappingMethod,
) -> Box<SyncToneMapper16Bit> {
    assert!((8..=16).contains(&bit_depth));
    let linear_table = hdr_transfer_function.generate_linear_table_u16(bit_depth);
    let gamma_table = display_transfer_function.generate_gamma_table_u16(bit_depth);
    let conversion = if input_color_space != output_color_space {
        Some(make_icc_transform(input_color_space, output_color_space))
    } else {
        None
    };
    let tone_map = make_mapper::<CN>(content_hdr_metadata, input_color_space, method);
    Box::new(ToneMapperImpl::<u16, 65536, CN> {
        linear_map: linear_table,
        gamma_map: gamma_table,
        gamut_color_conversion: conversion,
        tone_map,
    })
}

fn make_mapper<const CN: usize>(
    content_hdr_metadata: GainHDRMetadata,
    input_color_space: GamutColorSpace,
    method: ToneMappingMethod,
) -> Box<SyncToneMap> {
    let tone_map: Box<SyncToneMap> = match method {
        ToneMappingMethod::Rec2408 => Box::new(Rec2408ToneMapper::<CN>::new(
            content_hdr_metadata.content_max_brightness,
            content_hdr_metadata.display_max_brightness,
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
            content_hdr_metadata: GainHDRMetadata,
            hdr_transfer_function: HdrTransferFunction,
            input_color_space: GamutColorSpace,
            display_transfer_function: TransferFunction,
            output_color_space: GamutColorSpace,
            method: ToneMappingMethod,
        ) -> Box<SyncToneMapper8Bit> {
            create_tone_mapper_u8::<$cn>(
                content_hdr_metadata,
                hdr_transfer_function,
                input_color_space,
                display_transfer_function,
                output_color_space,
                method,
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
            content_hdr_metadata: GainHDRMetadata,
            hdr_transfer_function: HdrTransferFunction,
            input_color_space: GamutColorSpace,
            display_transfer_function: TransferFunction,
            output_color_space: GamutColorSpace,
            method: ToneMappingMethod,
        ) -> Box<SyncToneMapper16Bit> {
            create_tone_mapper_u16::<$cn>(
                $bp,
                content_hdr_metadata,
                hdr_transfer_function,
                input_color_space,
                display_transfer_function,
                output_color_space,
                method,
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
