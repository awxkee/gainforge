/*
 * // Copyright (c) Radzivon Bartoshyk 4/2026. All rights reserved.
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
use crate::util::{MangledCoercion, split_by_twos, split_by_twos_mut};
use crate::{ForgeError, ToneMapper};
use moxcms::Matrix3f;
use num_traits::AsPrimitive;
use std::arch::aarch64::*;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub(crate) struct DisplayReinhardParamsNeon {
    w_a: f32,
    w_b: f32,
    primaries: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExtendedReinhardNeon {
    primaries: [f32; 4],
}

impl ExtendedReinhardNeon {
    pub(crate) fn new(primaries: [f32; 3]) -> Self {
        Self {
            primaries: [primaries[0], primaries[1], primaries[2], 0.0],
        }
    }
}

impl DisplayReinhardParamsNeon {
    pub(crate) fn new(
        content_max_brightness: f32,
        display_max_brightness: f32,
        white_point: f32,
        primaries: [f32; 3],
    ) -> Self {
        let ld = content_max_brightness / white_point;
        let w_a = (display_max_brightness / white_point) / (ld * ld);
        let w_b = 1.0f32 / (display_max_brightness / white_point);
        Self {
            w_a,
            w_b,
            primaries: [primaries[0], primaries[1], primaries[2], 0.0],
        }
    }
}

#[repr(align(16), C)]
#[allow(unused)]
pub(crate) struct NeonAlignedU16(pub(crate) [u16; 8]);

pub(crate) trait HotScaleMapperNeon {
    fn map(&self, color: [float32x4_t; 3]) -> float32x4_t;
}

impl HotScaleMapperNeon for DisplayReinhardParamsNeon {
    #[inline(always)]
    fn map(&self, color: [float32x4_t; 3]) -> float32x4_t {
        unsafe {
            let mut luma = vmulq_n_f32(color[0], self.primaries[0]);
            luma = vfmaq_n_f32(luma, color[1], self.primaries[1]);
            luma = vfmaq_n_f32(luma, color[2], self.primaries[2]);
            vdivq_f32(
                vfmaq_n_f32(vdupq_n_f32(1.), luma, self.w_a),
                vfmaq_n_f32(vdupq_n_f32(1.), luma, self.w_b),
            )
        }
    }
}

impl HotScaleMapperNeon for ExtendedReinhardNeon {
    #[inline(always)]
    fn map(&self, color: [float32x4_t; 3]) -> float32x4_t {
        unsafe {
            let mut luma = vmulq_n_f32(color[0], self.primaries[0]);
            luma = vfmaq_n_f32(luma, color[1], self.primaries[1]);
            luma = vfmaq_n_f32(luma, color[2], self.primaries[2]);
            vdivq_f32(luma, vaddq_f32(vdupq_n_f32(1.), luma))
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn vzip1q_f32_f64(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vreinterpretq_f32_f64(vzip1q_f64(
        vreinterpretq_f64_f32(a),
        vreinterpretq_f64_f32(b),
    ))
}

pub(crate) struct HotLumaMapperNeon<
    T: Clone + Copy + Default + 'static,
    H: HotScaleMapperNeon,
    const N: usize,
    const CN: usize,
> {
    pub(crate) linear: Box<[f32; N]>,
    pub(crate) gamma: Box<[T; 65536]>,
    pub(crate) bit_depth: usize,
    pub(crate) gamma_lut: usize,
    pub(crate) adaptation_matrix: Matrix3f,
    pub(crate) mapper: H,
    pub(crate) tone_map: Arc<crate::tonemapper::SyncToneMap>,
    pub(crate) exposure: f32,
}

pub(crate) type HotTunedReinhardNeon<T, const N: usize, const CN: usize> =
    HotLumaMapperNeon<T, DisplayReinhardParamsNeon, N, CN>;
pub(crate) type HotExtendedReinhardNeon<T, const N: usize, const CN: usize> =
    HotLumaMapperNeon<T, ExtendedReinhardNeon, N, CN>;

impl<
    T: Clone + Copy + Default + Debug + 'static + MangledCoercion,
    H: HotScaleMapperNeon,
    const N: usize,
    const CN: usize,
> HotLumaMapperNeon<T, H, N, CN>
where
    u32: AsPrimitive<T>,
{
    #[target_feature(enable = "neon")]
    fn tonemap_lane_impl(&self, src: &[T], dst: &mut [T]) -> Result<(), ForgeError> {
        let mut temporary0 = NeonAlignedU16([0; 8]);
        let mut temporary1 = NeonAlignedU16([0; 8]);
        let mut temporary2 = NeonAlignedU16([0; 8]);
        let mut temporary3 = NeonAlignedU16([0; 8]);

        if src.len() / CN != dst.len() / CN {
            return Err(ForgeError::LaneSizeMismatch);
        }
        if !src.len().is_multiple_of(CN) {
            return Err(ForgeError::LaneMultipleOfChannels);
        }
        if !dst.len().is_multiple_of(CN) {
            return Err(ForgeError::LaneMultipleOfChannels);
        }

        let t = self.adaptation_matrix.transpose();
        let scale = (self.gamma_lut - 1) as f32;
        let max_colors: T = ((1 << self.bit_depth) - 1).as_();

        let r_lin = &self.linear;

        let (src_chunks, src_remainder) = split_by_twos(src, CN);
        let (dst_chunks, dst_remainder) = split_by_twos_mut(dst, CN);

        unsafe {
            let m0 = vld1q_f32([t.v[0][0], t.v[0][1], t.v[0][2], 0.].as_ptr());
            let m1 = vld1q_f32([t.v[1][0], t.v[1][1], t.v[1][2], 0.].as_ptr());
            let m2 = vld1q_f32([t.v[2][0], t.v[2][1], t.v[2][2], 0.].as_ptr());

            let v_scale = vdupq_n_f32(scale);

            let rnd = vdupq_n_f32(0.5);

            if !src_chunks.is_empty() {
                let (src0, src1) = src_chunks.split_at(src_chunks.len() / 2);
                let (dst0, dst1) = dst_chunks.split_at_mut(dst_chunks.len() / 2);
                let src_iter0 = src0.chunks_exact(CN * 2);
                let src_iter1 = src1.chunks_exact(CN * 2);

                let (mut r0, mut g0, mut b0, mut a0);
                let (mut r1, mut g1, mut b1, mut a1);
                let (mut r2, mut g2, mut b2, mut a2);
                let (mut r3, mut g3, mut b3, mut a3);

                for (((src0, src1), dst0), dst1) in src_iter0
                    .zip(src_iter1)
                    .zip(dst0.chunks_exact_mut(CN * 2))
                    .zip(dst1.chunks_exact_mut(CN * 2))
                {
                    let r0p = &r_lin[src0[0]._as_usize()];
                    let g0p = &r_lin[src0[1]._as_usize()];
                    let b0p = &r_lin[src0[2]._as_usize()];

                    let r1p = &r_lin[src0[CN]._as_usize()];
                    let g1p = &r_lin[src0[1 + CN]._as_usize()];
                    let b1p = &r_lin[src0[2 + CN]._as_usize()];

                    let r2p = &r_lin[src1[0]._as_usize()];
                    let g2p = &r_lin[src1[1]._as_usize()];
                    let b2p = &r_lin[src1[2]._as_usize()];

                    let r3p = &r_lin[src1[CN]._as_usize()];
                    let g3p = &r_lin[src1[1 + CN]._as_usize()];
                    let b3p = &r_lin[src1[2 + CN]._as_usize()];

                    r0 = vld1q_dup_f32(r0p);
                    g0 = vld1q_dup_f32(g0p);
                    b0 = vld1q_dup_f32(b0p);

                    r1 = vld1q_dup_f32(r1p);
                    g1 = vld1q_dup_f32(g1p);
                    b1 = vld1q_dup_f32(b1p);

                    r2 = vld1q_dup_f32(r2p);
                    g2 = vld1q_dup_f32(g2p);
                    b2 = vld1q_dup_f32(b2p);

                    r3 = vld1q_dup_f32(r3p);
                    g3 = vld1q_dup_f32(g3p);
                    b3 = vld1q_dup_f32(b3p);

                    let z_r = vzip1q_f32_f64(vzip1q_f32(r0, r1), vzip1q_f32(r2, r3));
                    let z_g = vzip1q_f32_f64(vzip1q_f32(g0, g1), vzip1q_f32(g2, g3));
                    let z_b = vzip1q_f32_f64(vzip1q_f32(b0, b1), vzip1q_f32(b2, b3));

                    let scale = vmulq_n_f32(self.mapper.map([z_r, z_g, z_b]), self.exposure);

                    r0 = vmulq_laneq_f32::<0>(r0, scale);
                    g0 = vmulq_laneq_f32::<0>(g0, scale);
                    b0 = vmulq_laneq_f32::<0>(b0, scale);

                    r1 = vmulq_laneq_f32::<1>(r1, scale);
                    g1 = vmulq_laneq_f32::<1>(g1, scale);
                    b1 = vmulq_laneq_f32::<1>(b1, scale);

                    r2 = vmulq_laneq_f32::<2>(r2, scale);
                    g2 = vmulq_laneq_f32::<2>(g2, scale);
                    b2 = vmulq_laneq_f32::<2>(b2, scale);

                    r3 = vmulq_laneq_f32::<3>(r3, scale);
                    g3 = vmulq_laneq_f32::<3>(g3, scale);
                    b3 = vmulq_laneq_f32::<3>(b3, scale);

                    a0 = if CN == 4 { src0[3] } else { max_colors };

                    a1 = if CN == 4 { src0[3 + CN] } else { max_colors };

                    a2 = if CN == 4 { src1[3] } else { max_colors };

                    a3 = if CN == 4 { src1[3 + CN] } else { max_colors };

                    let v0_0 = vmulq_f32(r0, m0);
                    let v0_1 = vmulq_f32(r1, m0);
                    let v0_2 = vmulq_f32(r2, m0);
                    let v0_3 = vmulq_f32(r3, m0);

                    let v1_0 = vfmaq_f32(v0_0, g0, m1);
                    let v1_1 = vfmaq_f32(v0_1, g1, m1);
                    let v1_2 = vfmaq_f32(v0_2, g2, m1);
                    let v1_3 = vfmaq_f32(v0_3, g3, m1);

                    let mut vr0 = vfmaq_f32(v1_0, b0, m2);
                    let mut vr1 = vfmaq_f32(v1_1, b1, m2);
                    let mut vr2 = vfmaq_f32(v1_2, b2, m2);
                    let mut vr3 = vfmaq_f32(v1_3, b3, m2);

                    vr0 = vfmaq_f32(rnd, vr0, v_scale);
                    vr1 = vfmaq_f32(rnd, vr1, v_scale);
                    vr2 = vfmaq_f32(rnd, vr2, v_scale);
                    vr3 = vfmaq_f32(rnd, vr3, v_scale);

                    vr0 = vminq_f32(vr0, v_scale);
                    vr1 = vminq_f32(vr1, v_scale);
                    vr2 = vminq_f32(vr2, v_scale);
                    vr3 = vminq_f32(vr3, v_scale);

                    let zx0 = vcvtq_u32_f32(vr0);
                    let zx1 = vcvtq_u32_f32(vr1);
                    let zx2 = vcvtq_u32_f32(vr2);
                    let zx3 = vcvtq_u32_f32(vr3);

                    vst1q_u32(temporary0.0.as_mut_ptr() as *mut _, zx0);
                    vst1q_u32(temporary1.0.as_mut_ptr() as *mut _, zx1);
                    vst1q_u32(temporary2.0.as_mut_ptr() as *mut _, zx2);
                    vst1q_u32(temporary3.0.as_mut_ptr() as *mut _, zx3);

                    dst0[0] = self.gamma[temporary0.0[0] as usize];
                    dst0[1] = self.gamma[temporary0.0[2] as usize];
                    dst0[2] = self.gamma[temporary0.0[4] as usize];
                    if CN == 4 {
                        dst0[3] = a0;
                    }

                    dst0[CN] = self.gamma[temporary1.0[0] as usize];
                    dst0[1 + CN] = self.gamma[temporary1.0[2] as usize];
                    dst0[2 + CN] = self.gamma[temporary1.0[4] as usize];
                    if CN == 4 {
                        dst0[3 + CN] = a1;
                    }

                    dst1[0] = self.gamma[temporary2.0[0] as usize];
                    dst1[1] = self.gamma[temporary2.0[2] as usize];
                    dst1[2] = self.gamma[temporary2.0[4] as usize];
                    if CN == 4 {
                        dst1[3] = a2;
                    }

                    dst1[CN] = self.gamma[temporary3.0[0] as usize];
                    dst1[1 + CN] = self.gamma[temporary3.0[2] as usize];
                    dst1[2 + CN] = self.gamma[temporary3.0[4] as usize];
                    if CN == 4 {
                        dst1[3 + CN] = a3;
                    }
                }
            }

            for (src, dst) in src_remainder
                .as_chunks::<CN>()
                .0
                .iter()
                .zip(dst_remainder.as_chunks_mut::<CN>().0.iter_mut())
            {
                let rp = &r_lin[src[0]._as_usize()];
                let gp = &r_lin[src[1]._as_usize()];
                let bp = &r_lin[src[2]._as_usize()];
                let mut r = vld1q_dup_f32(rp);
                let mut g = vld1q_dup_f32(gp);
                let mut b = vld1q_dup_f32(bp);
                let a = if CN == 4 { src[3] } else { max_colors };

                let scale = vmulq_n_f32(self.mapper.map([r, g, b]), self.exposure);
                r = vmulq_f32(r, scale);
                g = vmulq_f32(g, scale);
                b = vmulq_f32(b, scale);

                let v0 = vmulq_f32(r, m0);
                let v1 = vfmaq_f32(v0, g, m1);
                let mut v = vfmaq_f32(v1, b, m2);

                v = vfmaq_f32(rnd, v, v_scale);
                v = vminq_f32(v, v_scale);

                let zx = vcvtq_u32_f32(v);
                vst1q_u32(temporary0.0.as_mut_ptr() as *mut _, zx);

                dst[0] = self.gamma[temporary0.0[0] as usize];
                dst[1] = self.gamma[temporary0.0[2] as usize];
                dst[2] = self.gamma[temporary0.0[4] as usize];
                if CN == 4 {
                    dst[3] = a;
                }
            }
        }

        Ok(())
    }
}

impl<
    T: Clone + Copy + Default + Debug + 'static + MangledCoercion,
    H: HotScaleMapperNeon,
    const N: usize,
    const CN: usize,
> ToneMapper<T> for HotLumaMapperNeon<T, H, N, CN>
where
    u32: AsPrimitive<T>,
{
    fn tonemap_lane(&self, src: &[T], dst: &mut [T]) -> Result<(), ForgeError> {
        unsafe { self.tonemap_lane_impl(src, dst) }
    }

    fn tonemap_linearized_lane(&self, in_place: &mut [f32]) -> Result<(), ForgeError> {
        assert!(CN == 3 || CN == 4);
        if !in_place.len().is_multiple_of(CN) {
            return Err(ForgeError::LaneMultipleOfChannels);
        }
        self.tone_map.process_lane(in_place);
        Ok(())
    }
}
