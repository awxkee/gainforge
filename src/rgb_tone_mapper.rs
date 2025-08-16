/*
 * // Copyright (c) Radzivon Bartoshyk 8/2025. All rights reserved.
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
use crate::{m_clamp, ForgeError, RgbToneMapperParameters, ToneMapper};
use moxcms::{filmlike_clip, CmsError, InPlaceStage, Matrix3f, Rgb};
use num_traits::AsPrimitive;
use std::fmt::Debug;

pub(crate) struct ToneMapperImpl<T: Copy, const N: usize, const CN: usize, const GAMMA_SIZE: usize>
{
    pub(crate) linear_map_r: Box<[f32; N]>,
    pub(crate) linear_map_g: Box<[f32; N]>,
    pub(crate) linear_map_b: Box<[f32; N]>,
    pub(crate) gamma_map_r: Box<[T; 65536]>,
    pub(crate) gamma_map_g: Box<[T; 65536]>,
    pub(crate) gamma_map_b: Box<[T; 65536]>,
    pub(crate) im_stage: Option<Box<dyn InPlaceStage + Sync + Send>>,
    pub(crate) tone_map: Box<crate::tonemapper::SyncToneMap>,
    pub(crate) params: RgbToneMapperParameters,
}

pub(crate) struct MatrixStage<const CN: usize> {
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

pub(crate) struct MatrixGamutClipping<const CN: usize> {
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
                rgb = filmlike_clip(rgb);
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
                chunk[0] = rgb.r.max(0.);
                chunk[1] = rgb.g.max(0.);
                chunk[2] = rgb.b.max(0.);
            }
        }

        let scale_value = (GAMMA_SIZE - 1) as f32;

        for (dst, src) in dst
            .chunks_exact_mut(CN)
            .zip(linearized_content.chunks_exact(CN))
        {
            let r = mlaf(0.5, src[0], scale_value).min(u16::MAX as f32) as u16;
            let g = mlaf(0.5, src[1], scale_value).min(u16::MAX as f32) as u16;
            let b = mlaf(0.5, src[2], scale_value).min(u16::MAX as f32) as u16;
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
