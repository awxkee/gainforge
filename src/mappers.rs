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
use moxcms::{FusedLog2, FusedPow, Matrix3f, Rgb, Vector3f};
use std::ops::Mul;

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq, Default)]
pub enum AgxLook {
    Agx,
    #[default]
    Punchy,
    Golden,
    Custom(AgxCustomLook),
}

/// Defines a tone mapping method.
///
/// All tone mappers are local unless other is stated.
///
/// See [this blog post](https://64.github.io/tonemapping/) for more details on
/// many of the supported tone mapping methods.
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub enum ToneMappingMethod {
    /// ITU-R broadcasting TV [recommendation 2408](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2408-4-2021-PDF-E.pdf)
    Rec2408(GainHdrMetadata),
    /// The ['Uncharted 2' filmic](https://www.gdcvault.com/play/1012351/Uncharted-2-HDR)
    /// tone mapping method.
    Filmic,
    /// The [Academy Color Encoding System](https://github.com/ampas/aces-core)
    /// filmic tone mapping method.
    Aces,
    /// Erik Reinhard's tone mapper from the paper "Photographic tone
    /// reproduction for digital images".
    Reinhard,
    /// Same as `Reinhard` but scales the output to the full dynamic
    /// range of the image.
    ExtendedReinhard,
    /// A variation of `Reinhard` that uses mixes color-based- with
    /// luminance-based tone mapping.
    ReinhardJodie,
    /// Simply clamp the output to the available dynamic range.
    Clamp,
    /// This is a parameterized curve based on the Blender Filmic tone mapping
    /// method similar to the module found in Ansel/Darktable.
    FilmicSpline(FilmicSplineParameters),
    /// Blender AGX tone mapper.
    /// It's not really supposed to be used on other color model than RGB.
    Agx(AgxLook),
}

pub(crate) trait ToneMap {
    fn process_lane(&self, in_place: &mut [f32]);
    /// This method always expect first item to be luma.
    fn process_luma_lane(&self, in_place: &mut [f32]);
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
        mlaf(1f32, self.w_a, luma) / mlaf(1f32, self.w_b, luma)
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

    fn process_luma_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            let luma = chunk[0];
            if luma == 0. {
                chunk[0] = 0.;
                continue;
            }
            let scale = self.tonemap(luma);
            chunk[0] = (chunk[0] * scale).min(1f32);
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

    fn process_luma_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            chunk[0] = self.uncharted2_filmic(chunk[0]).min(1f32);
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AcesToneMapper<const CN: usize> {}

impl<const CN: usize> AcesToneMapper<CN> {
    #[inline(always)]
    fn mul_input(&self, color: Rgb<f32>) -> Rgb<f32> {
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
    fn mul_output(&self, color: Rgb<f32>) -> Rgb<f32> {
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

    fn process_luma_lane(&self, in_place: &mut [f32]) {
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

    fn process_luma_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            chunk[0] = (chunk[0] / (1f32 + chunk[0])).min(1f32);
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

    fn process_luma_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            let luma = chunk[0];
            if luma == 0. {
                chunk[0] = 0.;
                continue;
            }
            let new_luma = luma / (1f32 + luma);
            chunk[0] = (chunk[0] * new_luma).min(1f32);
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

            let luma_scale = 1. / (1f32 + luma);

            chunk[0] = lerp(chunk[0] * luma_scale, tv_r, tv_r).min(1f32);
            chunk[1] = lerp(chunk[1] * luma_scale, tv_g, tv_g).min(1f32);
            chunk[2] = lerp(chunk[1] * luma_scale, tv_b, tv_b).min(1f32);
        }
    }

    fn process_luma_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            let luma = chunk[0];
            if luma == 0. {
                chunk[0] = 0.;
                continue;
            }
            let tv_r = chunk[0] / (1.0f32 + chunk[0]);

            chunk[0] = lerp(chunk[0] / (1f32 + luma), tv_r, tv_r).min(1f32);
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

    fn process_luma_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            chunk[0] = chunk[0].min(1f32).max(0f32);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AgxToneMapper<const CN: usize> {
    pub(crate) primaries: [f32; 3],
    pub(crate) agx_custom_look: AgxCustomLook,
}

const AGX_INSET: Matrix3f = Matrix3f {
    v: [
        [0.856627153315983, 0.137318972929847, 0.11189821299995],
        [0.0951212405381588, 0.761241990602591, 0.0767994186031903],
        [0.0482516061458583, 0.101439036467562, 0.811302368396859],
    ],
};

const AGX_OUTSET_INV: Matrix3f = Matrix3f {
    v: [
        [0.899796955911611, 0.11142098895748, 0.11142098895748],
        [0.0871996192028351, 0.875575586156966, 0.0871996192028349],
        [0.013003424885555, 0.0130034248855548, 0.801379391839686],
    ],
};

#[inline]
fn agx_default_contrast(x: f32) -> f32 {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;

    let w0 = mlaf(0.002857, -0.1718, x);
    let w1 = mlaf(4.361, -28.72, x);
    let w2 = mlaf(92.06, -126.7, x);
    let w3 = mlaf(78.01, -17.86, x);

    let z0 = mlaf(w0, x2, w1);
    let z1 = mlaf(x4 * w2, x6, w3);

    z1 + z0
}

const AGX_OUTSET: Matrix3f = AGX_OUTSET_INV.inverse();

const AGX_MIN_EV: f32 = -12.47393; // log2(pow(2, LOG2_MIN) * MIDDLE_GRAY)
const AGX_MAX_EV: f32 = 4.026069; // log2(pow(2, LOG2_MAX) * MIDDLE_GRAY)

#[derive(Copy, Clone, Default)]
pub(crate) struct AgxPunchy {}

#[derive(Copy, Clone, Default)]
pub(crate) struct AgxGolden {}

#[derive(Copy, Clone, Default)]
pub(crate) struct AgxDefault {}

#[derive(Copy, Clone, Default, Debug, PartialOrd, PartialEq)]
pub struct AgxCustomLook {
    pub slope: Rgb<f32>,
    pub power: Rgb<f32>,
    pub saturation: Rgb<f32>,
    pub offset: Rgb<f32>,
}

impl AgxPunchy {
    pub(crate) fn custom_look() -> AgxCustomLook {
        AgxCustomLook {
            slope: Rgb::new(1.0, 1.0, 1.0),
            power: Rgb::new(1.0, 1.0, 1.0),
            saturation: Rgb::new(1.4, 1.4, 1.4),
            offset: Rgb::default(),
        }
    }
}

impl AgxGolden {
    pub(crate) fn custom_look() -> AgxCustomLook {
        AgxCustomLook {
            slope: Rgb::new(1.0, 0.9, 0.5),
            power: Rgb::new(0.8, 0.8, 0.8),
            saturation: Rgb::new(1.2, 1.2, 1.2),
            offset: Rgb::default(),
        }
    }
}

impl AgxDefault {
    pub(crate) fn custom_look() -> AgxCustomLook {
        AgxCustomLook {
            slope: Rgb::new(1.0, 1.0, 1.0),
            power: Rgb::new(1.0, 1.0, 1.0),
            saturation: Rgb::new(1.0, 1.0, 1.0),
            offset: Rgb::default(),
        }
    }
}

impl<const CN: usize> AgxToneMapper<CN> {
    #[inline]
    fn look(&self, rgb: Rgb<f32>) -> Rgb<f32> {
        let slope = self.agx_custom_look.slope;
        let power = self.agx_custom_look.power;
        let sat = self.agx_custom_look.saturation;
        let offset = self.agx_custom_look.offset;

        let dot = offset.mla(rgb, slope).max(0.);

        let z = dot.f_pow(power);
        let luma = mlaf(
            mlaf(self.primaries[0] * z.r, self.primaries[1], z.g),
            self.primaries[2],
            z.b,
        );
        sat.mul(z - luma) + luma
    }

    #[inline]
    fn apply(&self, v: Rgb<f32>) -> Rgb<f32> {
        let z = v.abs();
        let vec = Vector3f { v: [z.r, z.g, z.b] };
        let z0 = AGX_INSET.f_mul_vector(vec);
        let mut z1 = Rgb {
            r: z0.v[0],
            g: z0.v[1],
            b: z0.v[2],
        };
        z1 = z1.max(1e-10);
        let z2 = z1.f_log2().max(AGX_MIN_EV).min(AGX_MAX_EV);
        const RECIP_EV: f32 = 1.0 / (AGX_MAX_EV - AGX_MIN_EV);
        let z3 = (z2 - AGX_MIN_EV) * RECIP_EV;
        let z_contrast = Rgb {
            r: agx_default_contrast(z3.r),
            g: agx_default_contrast(z3.g),
            b: agx_default_contrast(z3.b),
        };
        let z4 = self.look(z_contrast);
        let vec1 = Vector3f {
            v: [z4.r, z4.g, z4.b],
        };
        let z5 = AGX_OUTSET.f_mul_vector(vec1);
        Rgb {
            r: z5.v[0],
            g: z5.v[1],
            b: z5.v[2],
        }
    }
}

impl<const CN: usize> ToneMap for AgxToneMapper<CN> {
    fn process_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            let rgb = Rgb {
                r: chunk[0],
                g: chunk[1],
                b: chunk[2],
            };
            let new_rgb = self.apply(rgb);
            chunk[0] = new_rgb.r.min(1.).max(0.);
            chunk[1] = new_rgb.g.min(1.).max(0.);
            chunk[2] = new_rgb.b.min(1.).max(0.);
        }
    }

    fn process_luma_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            let rgb = Rgb {
                r: chunk[0],
                g: chunk[0],
                b: chunk[0],
            };
            let new_rgb = self.apply(rgb);
            chunk[0] = new_rgb.r.min(1.).max(0.);
        }
    }
}
