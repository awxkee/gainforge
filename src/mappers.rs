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
use crate::mlaf::mlaf;
use crate::spline::FilmicSplineParameters;
use crate::GainHdrMetadata;
use std::ops::{Add, Div, Mul, Sub};

/// Defines tone mapping method
///
/// All tone mappers are local unless other is stated.
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub enum ToneMappingMethod {
    Rec2408(GainHdrMetadata),
    Filmic,
    Aces,
    Reinhard,
    ExtendedReinhard,
    ReinhardJodie,
    Clamp,
    Alu,
    FilmicSpline(FilmicSplineParameters),
}

pub(crate) trait ToneMap {
    fn process_lane(&self, in_place: &mut [f32]);
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Rec2408ToneMapper<const CN: usize> {
    w_a: f32,
    w_b: f32,
    primaries: [f32; 3],
}

impl<const CN: usize> Rec2408ToneMapper<CN> {
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
            primaries,
        }
    }
}

impl<const CN: usize> Rec2408ToneMapper<CN> {
    #[inline(always)]
    fn tonemap(&self, luma: f32) -> f32 {
        (1f32 + self.w_a * luma) / (1f32 + self.w_b * luma)
    }
}

impl<const CN: usize> ToneMap for Rec2408ToneMapper<CN> {
    fn process_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            let luma = chunk[0] * self.primaries[0]
                + chunk[1] * self.primaries[1]
                + chunk[2] * self.primaries[2];
            if luma == 0. {
                chunk[0] = 0.;
                chunk[1] = 0.;
                chunk[2] = 0.;
                continue;
            }
            let scale = self.tonemap(luma);
            chunk[0] = (chunk[0] * scale).min(1f32);
            chunk[1] = (chunk[1] * scale).min(1f32);
            chunk[2] = (chunk[2] * scale).min(1f32);
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FilmicToneMapper<const CN: usize> {}

#[inline(always)]
const fn uncharted2_tonemap_partial(x: f32) -> f32 {
    const A: f32 = 0.15f32;
    const B: f32 = 0.50f32;
    const C: f32 = 0.10f32;
    const D: f32 = 0.20f32;
    const E: f32 = 0.02f32;
    const F: f32 = 0.30f32;
    ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
}

impl<const CN: usize> FilmicToneMapper<CN> {
    #[inline(always)]
    fn uncharted2_filmic(&self, v: f32) -> f32 {
        let exposure_bias = 2.0f32;
        let curr = uncharted2_tonemap_partial(v * exposure_bias);

        const W: f32 = 11.2f32;
        const W_S: f32 = 1.0f32 / uncharted2_tonemap_partial(W);
        curr * W_S
    }
}

impl<const CN: usize> ToneMap for FilmicToneMapper<CN> {
    fn process_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            chunk[0] = self.uncharted2_filmic(chunk[0]).min(1f32);
            chunk[1] = self.uncharted2_filmic(chunk[1]).min(1f32);
            chunk[2] = self.uncharted2_filmic(chunk[2]).min(1f32);
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AcesToneMapper<const CN: usize> {}

#[derive(Copy, Clone)]
pub(crate) struct Rgb {
    pub(crate) r: f32,
    pub(crate) g: f32,
    pub(crate) b: f32,
}

impl Mul<Rgb> for Rgb {
    type Output = Self;

    fn mul(self, rhs: Rgb) -> Self::Output {
        Self {
            r: self.r * rhs.r,
            g: self.g * rhs.g,
            b: self.b * rhs.b,
        }
    }
}

impl Add<f32> for Rgb {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        Self {
            r: self.r + rhs,
            g: self.g + rhs,
            b: self.b + rhs,
        }
    }
}

impl Sub<f32> for Rgb {
    type Output = Self;
    fn sub(self, rhs: f32) -> Self::Output {
        Self {
            r: self.r - rhs,
            g: self.g - rhs,
            b: self.b - rhs,
        }
    }
}

impl Mul<f32> for Rgb {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            r: self.r * rhs,
            g: self.g * rhs,
            b: self.b * rhs,
        }
    }
}

impl Div<f32> for Rgb {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        Self {
            r: self.r / rhs,
            g: self.g / rhs,
            b: self.b / rhs,
        }
    }
}

impl Div<Rgb> for Rgb {
    type Output = Self;
    fn div(self, rhs: Rgb) -> Self::Output {
        Self {
            r: self.r / rhs.r,
            g: self.g / rhs.g,
            b: self.b / rhs.b,
        }
    }
}

impl<const CN: usize> AcesToneMapper<CN> {
    #[inline(always)]
    fn mul_input(&self, color: Rgb) -> Rgb {
        let a = mlaf(
            mlaf(0.35458f32 * color.g, 0.04823f32, color.b),
            0.59719f32,
            color.r,
        );
        let b = mlaf(
            mlaf(0.07600f32 * color.r, 0.90834f32, color.g),
            0.01566f32,
            color.b,
        );
        let c = mlaf(
            mlaf(0.02840f32 * color.r, 0.13383f32, color.g),
            0.83777f32,
            color.b,
        );
        Rgb { r: a, g: b, b: c }
    }

    #[inline(always)]
    fn mul_output(&self, color: Rgb) -> Rgb {
        let a = mlaf(
            mlaf(1.60475f32 * color.r, -0.53108f32, color.g),
            -0.07367f32,
            color.b,
        );
        let b = mlaf(
            mlaf(-0.10208f32 * color.r, 1.10813f32, color.g),
            -0.00605f32,
            color.b,
        );
        let c = mlaf(
            mlaf(-0.00327f32 * color.r, -0.07276f32, color.g),
            1.07602f32,
            color.b,
        );
        Rgb { r: a, g: b, b: c }
    }
}

impl<const CN: usize> ToneMap for AcesToneMapper<CN> {
    fn process_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            let color_in = self.mul_input(Rgb {
                r: chunk[0],
                g: chunk[1],
                b: chunk[2],
            });
            let ca = color_in * (color_in + 0.0245786f32) - 0.000090537f32;
            let cb = color_in * (color_in * 0.983729f32 + 0.4329510f32) + 0.238081f32;
            let c_out = self.mul_output(ca / cb);
            chunk[0] = c_out.r.min(1f32);
            chunk[1] = c_out.g.min(1f32);
            chunk[2] = c_out.b.min(1f32);
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ReinhardToneMapper<const CN: usize> {}

impl<const CN: usize> ToneMap for ReinhardToneMapper<CN> {
    fn process_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            chunk[0] = (chunk[0] / (1f32 + chunk[0])).min(1f32);
            chunk[1] = (chunk[1] / (1f32 + chunk[1])).min(1f32);
            chunk[2] = (chunk[2] / (1f32 + chunk[2])).min(1f32);
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ExtendedReinhardToneMapper<const CN: usize> {
    pub(crate) primaries: [f32; 3],
}

impl<const CN: usize> ToneMap for ExtendedReinhardToneMapper<CN> {
    fn process_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            let luma = chunk[0] * self.primaries[0]
                + chunk[1] * self.primaries[1]
                + chunk[2] * self.primaries[2];
            if luma == 0. {
                chunk[0] = 0.;
                chunk[1] = 0.;
                chunk[2] = 0.;
                continue;
            }
            let new_luma = luma / (1f32 + luma);
            chunk[0] = (chunk[0] * new_luma).min(1f32);
            chunk[1] = (chunk[1] * new_luma).min(1f32);
            chunk[2] = (chunk[2] * new_luma).min(1f32);
        }
    }
}

#[inline(always)]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    mlaf(a, t, b - a)
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ReinhardJodieToneMapper<const CN: usize> {
    pub(crate) primaries: [f32; 3],
}

impl<const CN: usize> ToneMap for ReinhardJodieToneMapper<CN> {
    fn process_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            let luma = chunk[0] * self.primaries[0]
                + chunk[1] * self.primaries[1]
                + chunk[2] * self.primaries[2];
            if luma == 0. {
                chunk[0] = 0.;
                chunk[1] = 0.;
                chunk[2] = 0.;
                continue;
            }
            let tv_r = chunk[0] / (1.0f32 + chunk[0]);
            let tv_g = chunk[1] / (1.0f32 + chunk[1]);
            let tv_b = chunk[2] / (1.0f32 + chunk[2]);

            chunk[0] = lerp(chunk[0] / (1f32 + luma), tv_r, tv_r).min(1f32);
            chunk[1] = lerp(chunk[1] / (1f32 + luma), tv_g, tv_g).min(1f32);
            chunk[2] = lerp(chunk[1] / (1f32 + luma), tv_b, tv_b).min(1f32);
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ClampToneMapper<const CN: usize> {}

impl<const CN: usize> ToneMap for ClampToneMapper<CN> {
    fn process_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            chunk[0] = chunk[0].min(1f32).max(0f32);
            chunk[1] = chunk[1].min(1f32).max(0f32);
            chunk[2] = chunk[2].min(1f32).max(0f32);
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AluToneMapper<const CN: usize> {}

impl<const CN: usize> AluToneMapper<CN> {
    #[inline(always)]
    fn alu(&self, v: f32) -> f32 {
        let c = (v - 0.004).max(0f32);
        (c * (c * 6.2 + 0.5)) / (c * (c * 6.2 + 1.7) + 0.06)
    }
}

impl<const CN: usize> ToneMap for AluToneMapper<CN> {
    fn process_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            chunk[0] = self.alu(chunk[0]).min(1f32);
            chunk[1] = self.alu(chunk[1]).min(1f32);
            chunk[2] = self.alu(chunk[2]).min(1f32);
        }
    }
}
