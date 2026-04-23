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
use crate::mappers::c_uncharted2_tonemap_partial;
use crate::neon::scale_rgb::{NeonAlignedU16, vzip1q_f32_f64};
use crate::util::{MangledCoercion, split_by_twos, split_by_twos_mut};
use crate::{ForgeError, ToneMapper};
use moxcms::Matrix3f;
use num_traits::AsPrimitive;
use std::arch::aarch64::*;
use std::fmt::Debug;
use std::sync::Arc;

pub(crate) trait HotEachScaleMapperNeon {
    fn map(&self, color: [float32x4_t; 3]) -> [float32x4_t; 3];
}

#[derive(Default)]
pub(crate) struct ReinhardNeon {}

impl HotEachScaleMapperNeon for ReinhardNeon {
    #[inline(always)]
    fn map(&self, color: [float32x4_t; 3]) -> [float32x4_t; 3] {
        unsafe {
            let one = vdupq_n_f32(1.);

            let z = vzip1q_f32_f64(
                vzip1q_f32(color[0], color[1]),
                vzip1q_f32(color[2], vdupq_n_f32(0.)),
            );

            let out = vdivq_f32(z, vaddq_f32(one, z));

            [
                vdupq_laneq_f32::<0>(out),
                vdupq_laneq_f32::<1>(out),
                vdupq_laneq_f32::<2>(out),
            ]
        }
    }
}

pub(crate) struct ReinhardJodieNeon {
    pub(crate) primaries: [f32; 4],
}

impl ReinhardJodieNeon {
    pub(crate) fn new(primaries: [f32; 3]) -> Self {
        Self {
            primaries: [primaries[0], primaries[1], primaries[2], 0.0],
        }
    }
}

impl HotEachScaleMapperNeon for ReinhardJodieNeon {
    #[inline(always)]
    fn map(&self, color: [float32x4_t; 3]) -> [float32x4_t; 3] {
        unsafe {
            let one = vdupq_n_f32(1.);

            let mut luma = vmulq_n_f32(color[0], self.primaries[0]);
            luma = vfmaq_n_f32(luma, color[1], self.primaries[1]);
            luma = vfmaq_n_f32(luma, color[2], self.primaries[2]);

            let z = vzip1q_f32_f64(
                vzip1q_f32(color[0], color[1]),
                vzip1q_f32(color[2], vdupq_n_f32(0.)),
            );

            let tv = vdivq_f32(z, vaddq_f32(one, z));
            let luma_scale = vdivq_f32(one, vaddq_f32(one, luma));

            let c_scaled = vmulq_f32(z, luma_scale);

            let out = vfmaq_f32(c_scaled, tv, vsubq_f32(tv, c_scaled));

            [
                vdupq_laneq_f32::<0>(out),
                vdupq_laneq_f32::<1>(out),
                vdupq_laneq_f32::<2>(out),
            ]
        }
    }
}

#[derive(Default)]
pub(crate) struct HableNeon;

impl HableNeon {
    const A: f32 = 0.15f32;
    const B: f32 = 0.50f32;
    const C: f32 = 0.10f32;
    const D: f32 = 0.20f32;
    const E: f32 = 0.02f32;
    const F: f32 = 0.30f32;
    const W: f32 = 11.2f32;
    const EXPOSURE_BIAS: f32 = 2.0f32;
    const EF: f32 = Self::E / Self::F;
    const W_S: f32 = 1.0f32 / c_uncharted2_tonemap_partial(Self::W);
}

impl HableNeon {
    #[inline(always)]
    fn uncharted2_partial(x: float32x4_t) -> float32x4_t {
        unsafe {
            let r0 = vfmaq_n_f32(vdupq_n_f32(Self::C * Self::B), x, Self::A);
            let r1 = vfmaq_n_f32(vdupq_n_f32(Self::B), x, Self::A);
            let r2 = vfmaq_f32(vdupq_n_f32(Self::D * Self::E), x, r0);
            let r3 = vfmaq_f32(vdupq_n_f32(Self::D * Self::F), x, r1);
            let r = vrecpeq_f32(r3);
            let r = vmulq_f32(vrecpsq_f32(r3, r), r);
            vsubq_f32(vmulq_f32(r2, r), vdupq_n_f32(Self::EF))
        }
    }
}

impl HotEachScaleMapperNeon for HableNeon {
    #[inline(always)]
    fn map(&self, color: [float32x4_t; 3]) -> [float32x4_t; 3] {
        unsafe {
            let z = vzip1q_f32_f64(
                vzip1q_f32(color[0], color[1]),
                vzip1q_f32(color[2], vdupq_n_f32(0.)),
            );

            let z = vmulq_n_f32(z, Self::EXPOSURE_BIAS);

            let out = vmulq_n_f32(Self::uncharted2_partial(z), Self::W_S);
            [
                vdupq_laneq_f32::<0>(out),
                vdupq_laneq_f32::<1>(out),
                vdupq_laneq_f32::<2>(out),
            ]
        }
    }
}

pub(crate) struct AcesNeon {
    input0: float32x4_t,
    input1: float32x4_t,
    input2: float32x4_t,
    output0: float32x4_t,
    output1: float32x4_t,
    output2: float32x4_t,
}

impl Default for AcesNeon {
    fn default() -> Self {
        Self {
            input0: unsafe { vld1q_f32([0.59719f32, 0.07600f32, 0.02840f32, 0.].as_ptr()) },
            input1: unsafe { vld1q_f32([0.35458f32, 0.90834f32, 0.13383f32, 0.].as_ptr()) },
            input2: unsafe { vld1q_f32([0.04823f32, 0.01566f32, 0.83777f32, 0.].as_ptr()) },
            output0: unsafe { vld1q_f32([1.60475f32, -0.10208f32, -0.00327f32, 0.].as_ptr()) },
            output1: unsafe { vld1q_f32([-0.53108f32, 1.10813f32, -0.07276f32, 0.].as_ptr()) },
            output2: unsafe { vld1q_f32([-0.07367f32, -0.00605f32, 1.07602f32, 0.].as_ptr()) },
        }
    }
}

impl AcesNeon {
    #[inline(always)]
    fn mul_input(&self, r: float32x4_t, g: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe {
            vfmaq_f32(
                vfmaq_f32(vmulq_f32(r, self.input0), g, self.input1),
                b,
                self.input2,
            )
        }
    }

    #[inline(always)]
    fn mul_output(&self, r: float32x4_t, g: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe {
            vfmaq_f32(
                vfmaq_f32(vmulq_f32(r, self.output0), g, self.output1),
                b,
                self.output2,
            )
        }
    }

    #[inline(always)]
    fn aces_curve(v: float32x4_t) -> float32x4_t {
        unsafe {
            let ca = vfmaq_f32(
                vdupq_n_f32(-0.000090537f32),
                v,
                vaddq_f32(v, vdupq_n_f32(0.0245786f32)),
            );
            let cb = vfmaq_f32(
                vdupq_n_f32(0.238081f32),
                v,
                vfmaq_n_f32(vdupq_n_f32(0.4329510f32), v, 0.983729f32),
            );
            vdivq_f32(ca, cb)
        }
    }
}

impl HotEachScaleMapperNeon for AcesNeon {
    #[inline(always)]
    fn map(&self, color: [float32x4_t; 3]) -> [float32x4_t; 3] {
        unsafe {
            let z = self.mul_input(color[0], color[1], color[2]);

            let c = Self::aces_curve(z);

            let out = self.mul_output(
                vdupq_laneq_f32::<0>(c),
                vdupq_laneq_f32::<1>(c),
                vdupq_laneq_f32::<2>(c),
            );
            [
                vdupq_laneq_f32::<0>(out),
                vdupq_laneq_f32::<1>(out),
                vdupq_laneq_f32::<2>(out),
            ]
        }
    }
}

pub(crate) type HotReinhardJodieNeon<T, const N: usize, const CN: usize> =
    HotEachMapperNeon<T, ReinhardJodieNeon, N, CN>;
pub(crate) type HotFilmicNeon<T, const N: usize, const CN: usize> =
    HotEachMapperNeon<T, HableNeon, N, CN>;
pub(crate) type HotAcesNeon<T, const N: usize, const CN: usize> =
    HotEachMapperNeon<T, AcesNeon, N, CN>;
pub(crate) type HotReinhardNeon<T, const N: usize, const CN: usize> =
    HotEachMapperNeon<T, ReinhardNeon, N, CN>;

pub(crate) struct HotEachMapperNeon<
    T: Clone + Copy + Default + 'static,
    H: HotEachScaleMapperNeon,
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

impl<
    T: Clone + Copy + Default + Debug + 'static + MangledCoercion,
    H: HotEachScaleMapperNeon,
    const N: usize,
    const CN: usize,
> HotEachMapperNeon<T, H, N, CN>
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

                    [r0, g0, b0] = self.mapper.map([
                        vmulq_n_f32(r0, self.exposure),
                        vmulq_n_f32(g0, self.exposure),
                        vmulq_n_f32(b0, self.exposure),
                    ]);

                    [r1, g1, b1] = self.mapper.map([
                        vmulq_n_f32(r1, self.exposure),
                        vmulq_n_f32(g1, self.exposure),
                        vmulq_n_f32(b1, self.exposure),
                    ]);

                    [r2, g2, b2] = self.mapper.map([
                        vmulq_n_f32(r2, self.exposure),
                        vmulq_n_f32(g2, self.exposure),
                        vmulq_n_f32(b2, self.exposure),
                    ]);

                    [r3, g3, b3] = self.mapper.map([
                        vmulq_n_f32(r3, self.exposure),
                        vmulq_n_f32(g3, self.exposure),
                        vmulq_n_f32(b3, self.exposure),
                    ]);

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

                [r, g, b] = self.mapper.map([
                    vmulq_n_f32(r, self.exposure),
                    vmulq_n_f32(g, self.exposure),
                    vmulq_n_f32(b, self.exposure),
                ]);

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
    H: HotEachScaleMapperNeon,
    const N: usize,
    const CN: usize,
> ToneMapper<T> for HotEachMapperNeon<T, H, N, CN>
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
