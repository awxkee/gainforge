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
use crate::util::{MangledCoercion, split_by_twos, split_by_twos_mut};
use crate::{ForgeError, ToneMapper};
use moxcms::Matrix3f;
use num_traits::AsPrimitive;
use std::arch::x86_64::*;
use std::fmt::Debug;
use std::sync::Arc;

pub(crate) const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) fn _mm256_zip4_ps(a: __m256, b: __m256, c: __m256, d: __m256) -> __m256 {
    let ab_lo = _mm256_unpacklo_ps(a, b);
    let cd_lo = _mm256_unpacklo_ps(c, d);

    let t0 = _mm256_shuffle_ps::<{ shuffle(1, 0, 1, 0) }>(ab_lo, cd_lo);
    let t1 = _mm256_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(ab_lo, cd_lo);

    _mm256_permute2f128_ps::<0x20>(t0, t1)
}

pub(crate) type HotReinhardJodieAvx<T, const N: usize, const CN: usize> =
    HotEachMapperAvx<T, ReinhardJodieAvx, N, CN>;
pub(crate) type HotReinhardAvx<T, const N: usize, const CN: usize> =
    HotEachMapperAvx<T, ReinhardAvx, N, CN>;
pub(crate) type HotHableAvx<T, const N: usize, const CN: usize> =
    HotEachMapperAvx<T, HableAvx, N, CN>;
pub(crate) type HotAcesAvx<T, const N: usize, const CN: usize> =
    HotEachMapperAvx<T, AcesAvx, N, CN>;

#[derive(Default)]
pub(crate) struct ReinhardAvx {}

impl HotEachScaleMapperAvx for ReinhardAvx {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn map(&self, color: [__m256; 3]) -> [__m256; 3] {
        let one = _mm256_set1_ps(1.);

        let z = _mm256_zip4_ps(color[0], color[1], color[2], _mm256_setzero_ps());

        let out = _mm256_div_ps(z, _mm256_add_ps(one, z));

        [
            _mm256_permute_ps::<{ shuffle(0, 0, 0, 0) }>(out),
            _mm256_permute_ps::<{ shuffle(1, 1, 1, 1) }>(out),
            _mm256_permute_ps::<{ shuffle(2, 2, 2, 2) }>(out),
        ]
    }
}

pub(crate) struct ReinhardJodieAvx {
    pub(crate) primaries: [__m256; 4],
}

impl ReinhardJodieAvx {
    pub(crate) fn new(primaries: [f32; 3]) -> Self {
        unsafe {
            Self {
                primaries: [
                    _mm256_set1_ps(primaries[0]),
                    _mm256_set1_ps(primaries[1]),
                    _mm256_set1_ps(primaries[2]),
                    _mm256_setzero_ps(),
                ],
            }
        }
    }
}

impl HotEachScaleMapperAvx for ReinhardJodieAvx {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn map(&self, color: [__m256; 3]) -> [__m256; 3] {
        let one = _mm256_set1_ps(1.);

        let mut luma = _mm256_mul_ps(color[0], self.primaries[0]);
        luma = _mm256_fmadd_ps(color[1], self.primaries[1], luma);
        luma = _mm256_fmadd_ps(color[2], self.primaries[2], luma);

        let z = _mm256_zip4_ps(color[0], color[1], color[2], _mm256_setzero_ps());

        let tv = _mm256_div_ps(z, _mm256_add_ps(one, z));
        let luma_scale = _mm256_div_ps(one, _mm256_add_ps(one, luma));

        let c_scaled = _mm256_mul_ps(z, luma_scale);

        let out = _mm256_fmadd_ps(_mm256_sub_ps(tv, c_scaled), c_scaled, tv);

        [
            _mm256_permute_ps::<{ shuffle(0, 0, 0, 0) }>(out),
            _mm256_permute_ps::<{ shuffle(1, 1, 1, 1) }>(out),
            _mm256_permute_ps::<{ shuffle(2, 2, 2, 2) }>(out),
        ]
    }
}

#[derive(Default)]
pub(crate) struct HableAvx;

impl HableAvx {
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

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
fn rcp_refined(x: __m256) -> __m256 {
    let r = _mm256_rcp_ps(x);
    // NR step: r = r * (2 - x*r)
    let two = _mm256_set1_ps(2.0);
    _mm256_mul_ps(r, _mm256_fnmadd_ps(x, r, two))
}

impl HableAvx {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn uncharted2_partial(x: __m256) -> __m256 {
        let r0 = _mm256_fmadd_ps(
            x,
            _mm256_set1_ps(Self::A),
            _mm256_set1_ps(Self::C * Self::B),
        );
        let r1 = _mm256_fmadd_ps(x, _mm256_set1_ps(Self::A), _mm256_set1_ps(Self::B));
        let r2 = _mm256_fmadd_ps(x, r0, _mm256_set1_ps(Self::D * Self::E));
        let r3 = _mm256_fmadd_ps(x, r1, _mm256_set1_ps(Self::D * Self::F));
        _mm256_fmsub_ps(r2, rcp_refined(r3), _mm256_set1_ps(Self::EF))
    }
}

impl HotEachScaleMapperAvx for HableAvx {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn map(&self, color: [__m256; 3]) -> [__m256; 3] {
        let z = _mm256_zip4_ps(color[0], color[1], color[2], _mm256_setzero_ps());

        let z = _mm256_mul_ps(z, _mm256_set1_ps(Self::EXPOSURE_BIAS));

        let out = _mm256_mul_ps(Self::uncharted2_partial(z), _mm256_set1_ps(Self::W_S));
        [
            _mm256_permute_ps::<{ shuffle(0, 0, 0, 0) }>(out),
            _mm256_permute_ps::<{ shuffle(1, 1, 1, 1) }>(out),
            _mm256_permute_ps::<{ shuffle(2, 2, 2, 2) }>(out),
        ]
    }
}

pub(crate) struct AcesAvx {
    input0: __m256,
    input1: __m256,
    input2: __m256,
    output0: __m256,
    output1: __m256,
    output2: __m256,
}

impl Default for AcesAvx {
    fn default() -> Self {
        Self {
            input0: unsafe {
                _mm256_setr_ps(
                    0.59719f32, 0.07600f32, 0.02840f32, 0., 0.59719f32, 0.07600f32, 0.02840f32, 0.,
                )
            },
            input1: unsafe {
                _mm256_setr_ps(
                    0.35458f32, 0.90834f32, 0.13383f32, 0., 0.35458f32, 0.90834f32, 0.13383f32, 0.,
                )
            },
            input2: unsafe {
                _mm256_setr_ps(
                    0.04823f32, 0.01566f32, 0.83777f32, 0., 0.04823f32, 0.01566f32, 0.83777f32, 0.,
                )
            },
            output0: unsafe {
                _mm256_setr_ps(
                    1.60475f32,
                    -0.10208f32,
                    -0.00327f32,
                    0.,
                    1.60475f32,
                    -0.10208f32,
                    -0.00327f32,
                    0.,
                )
            },
            output1: unsafe {
                _mm256_setr_ps(
                    -0.53108f32,
                    1.10813f32,
                    -0.07276f32,
                    0.,
                    -0.53108f32,
                    1.10813f32,
                    -0.07276f32,
                    0.,
                )
            },
            output2: unsafe {
                _mm256_setr_ps(
                    -0.07367f32,
                    -0.00605f32,
                    1.07602f32,
                    0.,
                    -0.07367f32,
                    -0.00605f32,
                    1.07602f32,
                    0.,
                )
            },
        }
    }
}

impl AcesAvx {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn mul_input(&self, r: __m256, g: __m256, b: __m256) -> __m256 {
        _mm256_fmadd_ps(
            b,
            self.input2,
            _mm256_fmadd_ps(g, self.input1, _mm256_mul_ps(r, self.input0)),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn mul_output(&self, r: __m256, g: __m256, b: __m256) -> __m256 {
        _mm256_fmadd_ps(
            b,
            self.output2,
            _mm256_fmadd_ps(g, self.output1, _mm256_mul_ps(r, self.output0)),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn aces_curve(v: __m256) -> __m256 {
        let ca = _mm256_fmadd_ps(
            v,
            _mm256_add_ps(v, _mm256_set1_ps(0.0245786f32)),
            _mm256_set1_ps(-0.000090537f32),
        );
        let cb = _mm256_fmadd_ps(
            v,
            _mm256_fmadd_ps(v, _mm256_set1_ps(0.983729f32), _mm256_set1_ps(0.4329510f32)),
            _mm256_set1_ps(0.238081f32),
        );
        _mm256_div_ps(ca, cb)
    }
}

impl HotEachScaleMapperAvx for AcesAvx {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn map(&self, color: [__m256; 3]) -> [__m256; 3] {
        let z = self.mul_input(color[0], color[1], color[2]);

        let c = Self::aces_curve(z);

        let out = self.mul_output(
            _mm256_permute_ps::<{ shuffle(0, 0, 0, 0) }>(c),
            _mm256_permute_ps::<{ shuffle(1, 1, 1, 1) }>(c),
            _mm256_permute_ps::<{ shuffle(2, 2, 2, 2) }>(c),
        );
        [
            _mm256_permute_ps::<{ shuffle(0, 0, 0, 0) }>(out),
            _mm256_permute_ps::<{ shuffle(1, 1, 1, 1) }>(out),
            _mm256_permute_ps::<{ shuffle(2, 2, 2, 2) }>(out),
        ]
    }
}

#[repr(align(16), C)]
#[allow(unused)]
pub(crate) struct AvxAlignedU16(pub(crate) [u16; 16]);

pub(crate) trait HotEachScaleMapperAvx {
    unsafe fn map(&self, color: [__m256; 3]) -> [__m256; 3];
}

pub(crate) struct HotEachMapperAvx<
    T: Clone + Copy + Default + 'static,
    H: HotEachScaleMapperAvx,
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
    H: HotEachScaleMapperAvx,
    const N: usize,
    const CN: usize,
> HotEachMapperAvx<T, H, N, CN>
where
    u32: AsPrimitive<T>,
{
    #[target_feature(enable = "avx2", enable = "fma")]
    fn tonemap_lane_impl(&self, src: &[T], dst: &mut [T]) -> Result<(), ForgeError> {
        let mut temporary0 = AvxAlignedU16([0; 16]);
        let mut temporary1 = AvxAlignedU16([0; 16]);

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
            let m0 = _mm256_loadu_ps(
                [
                    t.v[0][0], t.v[0][1], t.v[0][2], 0., t.v[0][0], t.v[0][1], t.v[0][2], 0.,
                ]
                .as_ptr(),
            );
            let m1 = _mm256_loadu_ps(
                [
                    t.v[1][0], t.v[1][1], t.v[1][2], 0., t.v[1][0], t.v[1][1], t.v[1][2], 0.,
                ]
                .as_ptr(),
            );
            let m2 = _mm256_loadu_ps(
                [
                    t.v[2][0], t.v[2][1], t.v[2][2], 0., t.v[2][0], t.v[2][1], t.v[2][2], 0.,
                ]
                .as_ptr(),
            );

            let v_scale = _mm256_set1_ps(scale);

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

                    r0 = _mm_broadcast_ss(r0p);
                    g0 = _mm_broadcast_ss(g0p);
                    b0 = _mm_broadcast_ss(b0p);

                    r1 = _mm_broadcast_ss(r1p);
                    g1 = _mm_broadcast_ss(g1p);
                    b1 = _mm_broadcast_ss(b1p);

                    r2 = _mm_broadcast_ss(r2p);
                    g2 = _mm_broadcast_ss(g2p);
                    b2 = _mm_broadcast_ss(b2p);

                    r3 = _mm_broadcast_ss(r3p);
                    g3 = _mm_broadcast_ss(g3p);
                    b3 = _mm_broadcast_ss(b3p);

                    let mut zr0 = _mm256_setr_m128(r0, r1);
                    let mut zg0 = _mm256_setr_m128(g0, g1);
                    let mut zb0 = _mm256_setr_m128(b0, b1);

                    let mut zr1 = _mm256_setr_m128(r2, r3);
                    let mut zg1 = _mm256_setr_m128(g2, g3);
                    let mut zb1 = _mm256_setr_m128(b2, b3);

                    [zr0, zg0, zb0] = self.mapper.map([
                        _mm256_mul_ps(zr0, _mm256_set1_ps(self.exposure)),
                        _mm256_mul_ps(zg0, _mm256_set1_ps(self.exposure)),
                        _mm256_mul_ps(zb0, _mm256_set1_ps(self.exposure)),
                    ]);

                    [zr1, zg1, zb1] = self.mapper.map([
                        _mm256_mul_ps(zr1, _mm256_set1_ps(self.exposure)),
                        _mm256_mul_ps(zg1, _mm256_set1_ps(self.exposure)),
                        _mm256_mul_ps(zb1, _mm256_set1_ps(self.exposure)),
                    ]);

                    a0 = if CN == 4 { src0[3] } else { max_colors };

                    a1 = if CN == 4 { src0[3 + CN] } else { max_colors };

                    a2 = if CN == 4 { src1[3] } else { max_colors };

                    a3 = if CN == 4 { src1[3 + CN] } else { max_colors };

                    let v0_0 = _mm256_mul_ps(zr0, m0);
                    let v0_1 = _mm256_mul_ps(zr1, m0);

                    let v1_0 = _mm256_fmadd_ps(zg0, m1, v0_0);
                    let v1_1 = _mm256_fmadd_ps(zg1, m1, v0_1);

                    let mut vr0 = _mm256_fmadd_ps(zb0, m2, v1_0);
                    let mut vr1 = _mm256_fmadd_ps(zb1, m2, v1_1);

                    vr0 = _mm256_mul_ps(vr0, v_scale);
                    vr1 = _mm256_mul_ps(vr1, v_scale);

                    vr0 = _mm256_min_ps(vr0, v_scale);
                    vr1 = _mm256_min_ps(vr1, v_scale);

                    vr0 = _mm256_max_ps(vr0, _mm256_setzero_ps());
                    vr1 = _mm256_max_ps(vr1, _mm256_setzero_ps());

                    let zx0 = _mm256_cvtps_epi32(vr0);
                    let zx1 = _mm256_cvtps_epi32(vr1);

                    _mm256_store_si256(temporary0.0.as_mut_ptr() as *mut _, zx0);
                    _mm256_store_si256(temporary1.0.as_mut_ptr() as *mut _, zx1);

                    dst0[0] = self.gamma[temporary0.0[0] as usize];
                    dst0[1] = self.gamma[temporary0.0[2] as usize];
                    dst0[2] = self.gamma[temporary0.0[4] as usize];
                    if CN == 4 {
                        dst0[3] = a0;
                    }

                    dst0[CN] = self.gamma[temporary0.0[8] as usize];
                    dst0[1 + CN] = self.gamma[temporary0.0[10] as usize];
                    dst0[2 + CN] = self.gamma[temporary0.0[12] as usize];
                    if CN == 4 {
                        dst0[3 + CN] = a1;
                    }

                    dst1[0] = self.gamma[temporary1.0[0] as usize];
                    dst1[1] = self.gamma[temporary1.0[2] as usize];
                    dst1[2] = self.gamma[temporary1.0[4] as usize];
                    if CN == 4 {
                        dst1[3] = a2;
                    }

                    dst1[CN] = self.gamma[temporary1.0[8] as usize];
                    dst1[1 + CN] = self.gamma[temporary1.0[10] as usize];
                    dst1[2 + CN] = self.gamma[temporary1.0[12] as usize];
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

                let mut r = _mm256_broadcast_ss(rp);
                let mut g = _mm256_broadcast_ss(gp);
                let mut b = _mm256_broadcast_ss(bp);

                let a = if CN == 4 { src[3] } else { max_colors };

                [r, g, b] = self.mapper.map([
                    _mm256_mul_ps(r, _mm256_set1_ps(self.exposure)),
                    _mm256_mul_ps(g, _mm256_set1_ps(self.exposure)),
                    _mm256_mul_ps(b, _mm256_set1_ps(self.exposure)),
                ]);

                let v0 = _mm256_mul_ps(r, m0);
                let v1 = _mm256_fmadd_ps(g, m1, v0);
                let mut v = _mm256_fmadd_ps(b, m2, v1);

                v = _mm256_mul_ps(v, v_scale);
                v = _mm256_min_ps(v, v_scale);

                v = _mm256_max_ps(v, _mm256_setzero_ps());

                let zx = _mm256_cvtps_epi32(v);
                _mm256_store_si256(temporary0.0.as_mut_ptr().cast(), zx);

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
    H: HotEachScaleMapperAvx,
    const N: usize,
    const CN: usize,
> ToneMapper<T> for HotEachMapperAvx<T, H, N, CN>
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
