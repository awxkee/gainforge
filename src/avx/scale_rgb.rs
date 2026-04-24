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
use crate::avx::each_rgb::{AvxAlignedU16, shuffle};
use crate::util::{MangledCoercion, split_by_twos, split_by_twos_mut};
use crate::{ForgeError, ToneMapper};
use moxcms::Matrix3f;
use num_traits::AsPrimitive;
use std::arch::x86_64::*;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub(crate) struct DisplayReinhardParamsAvx {
    w_a: __m128,
    w_b: __m128,
    primaries: [__m128; 3],
}

impl DisplayReinhardParamsAvx {
    pub(crate) fn new(
        content_max_brightness: f32,
        display_max_brightness: f32,
        white_point: f32,
        primaries: [f32; 3],
    ) -> Self {
        unsafe {
            Self::new_impl(
                content_max_brightness,
                display_max_brightness,
                white_point,
                primaries,
            )
        }
    }

    #[target_feature(enable = "avx2")]
    fn new_impl(
        content_max_brightness: f32,
        display_max_brightness: f32,
        white_point: f32,
        primaries: [f32; 3],
    ) -> Self {
        let ld = content_max_brightness / white_point;
        let w_a = (display_max_brightness / white_point) / (ld * ld);
        let w_b = 1.0f32 / (display_max_brightness / white_point);
        Self {
            w_a: _mm_set1_ps(w_a),
            w_b: _mm_set1_ps(w_b),
            primaries: [
                _mm_set1_ps(primaries[0]),
                _mm_set1_ps(primaries[1]),
                _mm_set1_ps(primaries[2]),
            ],
        }
    }
}

impl HotScaleMapperAvx for DisplayReinhardParamsAvx {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn map(&self, color: [__m128; 3]) -> __m128 {
        let mut luma = _mm_mul_ps(color[0], self.primaries[0]);
        luma = _mm_fmadd_ps(color[1], self.primaries[1], luma);
        luma = _mm_fmadd_ps(color[2], self.primaries[2], luma);
        let ones = _mm_set1_ps(1.);
        _mm_div_ps(
            _mm_fmadd_ps(luma, self.w_a, ones),
            _mm_fmadd_ps(luma, self.w_b, ones),
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExtendedReinhardAvx {
    primaries: [__m128; 3],
    recip_max_l_sqr: __m128,
}

impl ExtendedReinhardAvx {
    pub(crate) fn new(primaries: [f32; 3], max_l: f32) -> Self {
        unsafe { Self::new_impl(primaries, max_l) }
    }

    #[target_feature(enable = "avx2")]
    fn new_impl(primaries: [f32; 3], max_l: f32) -> Self {
        Self {
            primaries: [
                _mm_set1_ps(primaries[0]),
                _mm_set1_ps(primaries[1]),
                _mm_set1_ps(primaries[2]),
            ],
            recip_max_l_sqr: _mm_set1_ps(1. / (max_l * max_l)),
        }
    }
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
fn _m128_rcp_refined(x: __m128) -> __m128 {
    let r = _mm_rcp_ps(x);
    // NR step: r = r * (2 - x*r)
    let two = _mm_set1_ps(2.0);
    _mm_mul_ps(r, _mm_fnmadd_ps(x, r, two))
}

impl HotScaleMapperAvx for ExtendedReinhardAvx {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn map(&self, color: [__m128; 3]) -> __m128 {
        let mut luma = _mm_mul_ps(color[0], self.primaries[0]);
        luma = _mm_fmadd_ps(color[1], self.primaries[1], luma);
        luma = _mm_fmadd_ps(color[2], self.primaries[2], luma);

        let mut recip_luma = _m128_rcp_refined(luma);
        let mask = _mm_cmpeq_ps(luma, _mm_setzero_ps());
        recip_luma = _mm_andnot_ps(mask, recip_luma);

        let numerator = _mm_mul_ps(
            luma,
            _mm_add_ps(_mm_set1_ps(1.0), _mm_mul_ps(luma, self.recip_max_l_sqr)),
        );
        _mm_mul_ps(
            _mm_div_ps(numerator, _mm_add_ps(_mm_set1_ps(1.0), luma)),
            recip_luma,
        )
    }
}

pub(crate) trait HotScaleMapperAvx {
    unsafe fn map(&self, color: [__m128; 3]) -> __m128;
}

#[inline]
#[target_feature(enable = "avx2")]
fn _m128_pack4_ps(a: __m128, b: __m128, c: __m128, d: __m128) -> __m128 {
    let q0 = _mm_unpacklo_ps(a, b);
    let q1 = _mm_unpacklo_ps(c, d);
    _mm_shuffle_ps::<{ shuffle(1, 0, 1, 0) }>(q0, q1)
}

pub(crate) type HotTunedReinhardAvx<T, const N: usize, const CN: usize> =
    HotLumaMapperAvx<T, DisplayReinhardParamsAvx, N, CN>;
pub(crate) type HotExtendedReinhardAvx<T, const N: usize, const CN: usize> =
    HotLumaMapperAvx<T, ExtendedReinhardAvx, N, CN>;

pub(crate) struct HotLumaMapperAvx<
    T: Clone + Copy + Default + 'static,
    H: HotScaleMapperAvx,
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
    H: HotScaleMapperAvx,
    const N: usize,
    const CN: usize,
> HotLumaMapperAvx<T, H, N, CN>
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
            let m0 = _mm256_setr_ps(
                t.v[0][0], t.v[0][1], t.v[0][2], 0., t.v[0][0], t.v[0][1], t.v[0][2], 0.,
            );
            let m1 = _mm256_setr_ps(
                t.v[1][0], t.v[1][1], t.v[1][2], 0., t.v[1][0], t.v[1][1], t.v[1][2], 0.,
            );
            let m2 = _mm256_setr_ps(
                t.v[2][0], t.v[2][1], t.v[2][2], 0., t.v[2][0], t.v[2][1], t.v[2][2], 0.,
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

                    let z_r = _m128_pack4_ps(r0, r1, r2, r3);
                    let z_g = _m128_pack4_ps(g0, g1, g2, g3);
                    let z_b = _m128_pack4_ps(b0, b1, b2, b3);

                    let scale =
                        _mm_mul_ps(self.mapper.map([z_r, z_g, z_b]), _mm_set1_ps(self.exposure));

                    let qs0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(scale, scale);
                    r0 = _mm_mul_ps(r0, qs0);
                    g0 = _mm_mul_ps(g0, qs0);
                    b0 = _mm_mul_ps(b0, qs0);

                    let qs1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(scale, scale);
                    r1 = _mm_mul_ps(r1, qs1);
                    g1 = _mm_mul_ps(g1, qs1);
                    b1 = _mm_mul_ps(b1, qs1);

                    let qs2 = _mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(scale, scale);
                    r2 = _mm_mul_ps(r2, qs2);
                    g2 = _mm_mul_ps(g2, qs2);
                    b2 = _mm_mul_ps(b2, qs2);

                    let qs2 = _mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(scale, scale);
                    r3 = _mm_mul_ps(r3, qs2);
                    g3 = _mm_mul_ps(g3, qs2);
                    b3 = _mm_mul_ps(b3, qs2);

                    a0 = if CN == 4 { src0[3] } else { max_colors };

                    a1 = if CN == 4 { src0[3 + CN] } else { max_colors };

                    a2 = if CN == 4 { src1[3] } else { max_colors };

                    a3 = if CN == 4 { src1[3 + CN] } else { max_colors };

                    let ar0 = _mm256_setr_m128(r0, r1);
                    let ar1 = _mm256_setr_m128(r2, r3);

                    let v0_0 = _mm256_mul_ps(ar0, m0);
                    let v0_1 = _mm256_mul_ps(ar1, m0);

                    let ag0 = _mm256_setr_m128(g0, g1);
                    let ag1 = _mm256_setr_m128(g2, g3);

                    let v1_0 = _mm256_fmadd_ps(ag0, m1, v0_0);
                    let v1_1 = _mm256_fmadd_ps(ag1, m1, v0_1);

                    let ab0 = _mm256_setr_m128(b0, b1);
                    let ab1 = _mm256_setr_m128(b2, b3);

                    let mut vr0 = _mm256_fmadd_ps(ab0, m2, v1_0);
                    let mut vr1 = _mm256_fmadd_ps(ab1, m2, v1_1);

                    vr0 = _mm256_mul_ps(vr0, v_scale);
                    vr1 = _mm256_mul_ps(vr1, v_scale);

                    vr0 = _mm256_min_ps(vr0, v_scale);
                    vr1 = _mm256_min_ps(vr1, v_scale);

                    let zx0 = _mm256_cvttps_epi32(vr0);
                    let zx1 = _mm256_cvttps_epi32(vr1);

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
                let mut r = _mm_broadcast_ss(rp);
                let mut g = _mm_broadcast_ss(gp);
                let mut b = _mm_broadcast_ss(bp);
                let a = if CN == 4 { src[3] } else { max_colors };

                let scale = _mm_mul_ps(self.mapper.map([r, g, b]), _mm_set1_ps(self.exposure));
                r = _mm_mul_ps(r, scale);
                g = _mm_mul_ps(g, scale);
                b = _mm_mul_ps(b, scale);

                let v0 = _mm_mul_ps(r, _mm256_castps256_ps128(m0));
                let v1 = _mm_fmadd_ps(g, _mm256_castps256_ps128(m1), v0);
                let mut v = _mm_fmadd_ps(b, _mm256_castps256_ps128(m2), v1);

                v = _mm_mul_ps(v, _mm256_castps256_ps128(v_scale));
                v = _mm_min_ps(v, _mm256_castps256_ps128(v_scale));

                let zx = _mm_cvtps_epi32(v);
                _mm_store_si128(temporary0.0.as_mut_ptr() as *mut _, zx);

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
    H: HotScaleMapperAvx,
    const N: usize,
    const CN: usize,
> ToneMapper<T> for HotLumaMapperAvx<T, H, N, CN>
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
