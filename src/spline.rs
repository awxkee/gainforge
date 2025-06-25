/*
 * // Copyright (c) Radzivon Bartoshyk 3/2025. All rights reserved.
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
use crate::m_clamp;
use crate::mappers::ToneMap;
use moxcms::Rgb;
use pxfm::f_powf;

#[derive(Copy, Clone, Debug, PartialOrd, PartialEq)]
pub struct FilmicSplineParameters {
    pub output_power: f32,
    pub latitude: f32,           // $MIN: 0.01 $MAX: 99 $DEFAULT: 33.0
    pub white_point_source: f32, // $MIN: 0.1 $MAX: 16 $DEFAULT: 4.0 $DESCRIPTION: "white relative exposure"
    pub black_point_source: f32, // $MIN: -16 $MAX: -0.1 $DEFAULT: -8.0 $DESCRIPTION: "black relative exposure"
    pub contrast: f32,           // $MIN: 0 $MAX: 5 $DEFAULT: 1.18
    pub black_point_target: f32, // $MIN: 0.000 $MAX: 20.000 $DEFAULT: 0.01517634 $DESCRIPTION: "target black luminance"
    pub grey_point_target: f32, // $MIN: 1 $MAX: 50 $DEFAULT: 18.45 $DESCRIPTION: "target middle gray"
    pub white_point_target: f32, // $MIN: 0 $MAX: 1600 $DEFAULT: 100 $DESCRIPTION: "target white luminance"
    pub balance: f32, // $MIN: -50 $MAX: 50 $DEFAULT: 0.0 $DESCRIPTION: "shadows \342\206\224 highlights balance"
    pub saturation: f32, // $MIN: -200 $MAX: 200 $DEFAULT: 0 $DESCRIPTION: "extreme luminance saturation"
}

impl Default for FilmicSplineParameters {
    fn default() -> Self {
        Self {
            output_power: 1.0,
            latitude: 33f32,
            white_point_source: 3f32,
            black_point_source: -8f32,
            contrast: 1.18,
            black_point_target: 0.01517634f32,
            grey_point_target: 18.45f32,
            white_point_target: 100.0,
            balance: 0.0,
            saturation: 0f32,
        }
    }
}

#[derive(Copy, Clone, Default)]
pub(crate) struct FilmicSpline {
    pub(crate) x: [f32; 5],
    pub(crate) y: [f32; 5],
    pub(crate) m1: [f32; 3],
    pub(crate) m2: [f32; 3],
    pub(crate) m3: [f32; 3],
    pub(crate) m4: [f32; 3],
    pub(crate) m5: [f32; 3],
    pub(crate) latitude_max: f32,
    pub(crate) latitude_min: f32,
    pub(crate) grey_source: f32,
    pub(crate) black_source: f32,
    pub(crate) dynamic_range: f32,
    pub(crate) sigma_toe: f32,
    pub(crate) sigma_shoulder: f32,
    pub(crate) saturation: f32,
}

impl FilmicSpline {
    pub(crate) fn apply(&self, x: f32) -> f32 {
        if x < self.latitude_min {
            let xi = self.latitude_min - x;
            let rat = xi * (xi * self.m2[0] + 1f32);
            self.m4[0] - self.m1[0] * rat / (rat + self.m3[0])
        } else if x > self.latitude_max {
            let xi = x - self.latitude_max;
            let rat = xi * (xi * self.m2[1] + 1f32);
            self.m4[1] + self.m1[1] * rat / (rat + self.m3[1])
        } else {
            self.m1[2] + x * self.m2[2]
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct SplineToneMapper<const CN: usize> {
    pub(crate) spline: FilmicSpline,
    pub(crate) primaries: [f32; 3],
}

impl<const CN: usize> SplineToneMapper<CN> {
    #[inline(always)]
    fn shaper(&self, x: f32) -> f32 {
        (((x.max(1.52587890625e-05f32) / self.spline.grey_source).log2()
            - self.spline.black_source)
            / self.spline.dynamic_range)
            .min(1.0f32)
            .max(0f32)
    }

    #[inline(always)]
    fn desaturate(&self, x: f32, sigma_toe: f32, sigma_shoulder: f32, saturation: f32) -> f32 {
        let radius_toe = x;
        let radius_shoulder = 1.0f32 - x;
        let sat2 = 0.5f32 / f32::sqrt(saturation);
        let key_toe = f32::exp(-radius_toe * radius_toe / sigma_toe * sat2);
        let key_shoulder = f32::exp(-radius_shoulder * radius_shoulder / sigma_shoulder * sat2);

        saturation - (key_toe + key_shoulder) * (saturation)
    }
}

impl<const CN: usize> ToneMap for SplineToneMapper<CN> {
    fn process_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            let rgb = Rgb::new(chunk[0], chunk[1], chunk[2]);
            let mut norm =
                (rgb.r * self.primaries[0] + rgb.g * self.primaries[1] + rgb.b * self.primaries[2])
                    .max(1.52587890625e-05f32);
            let mut ratios = rgb / norm;
            let min_ratio = ratios.r.min(ratios.b).min(ratios.g);
            if min_ratio < 0f32 {
                ratios = ratios - min_ratio;
            }

            norm = self.shaper(norm);

            let desat = self.desaturate(
                norm,
                self.spline.sigma_toe,
                self.spline.sigma_shoulder,
                self.spline.saturation,
            );

            let mapped = self.spline.apply(norm).min(1f32).max(0f32);

            ratios.r = ratios.r + (1.0f32 - ratios.r) * (1.0f32 - desat);
            ratios.g = ratios.g + (1.0f32 - ratios.g) * (1.0f32 - desat);
            ratios.b = ratios.b + (1.0f32 - ratios.b) * (1.0f32 - desat);

            ratios *= mapped;

            chunk[0] = ratios.r.min(1f32).max(0f32);
            chunk[1] = ratios.g.min(1f32).max(0f32);
            chunk[2] = ratios.b.min(1f32).max(0f32);
        }
    }

    fn process_luma_lane(&self, in_place: &mut [f32]) {
        for chunk in in_place.chunks_exact_mut(CN) {
            let rgb = Rgb::new(chunk[0], chunk[1], chunk[2]);
            let mut norm = rgb.r.max(1.52587890625e-05f32);
            let mut ratios = rgb / norm;
            let min_ratio = ratios.r;
            if min_ratio < 0f32 {
                ratios = ratios - min_ratio;
            }

            norm = self.shaper(norm);

            let desat = self.desaturate(
                norm,
                self.spline.sigma_toe,
                self.spline.sigma_shoulder,
                self.spline.saturation,
            );

            let mapped = self.spline.apply(norm).min(1f32).max(0f32);

            ratios.r = ratios.r + (1.0f32 - ratios.r) * (1.0f32 - desat);

            ratios *= mapped;

            chunk[0] = ratios.r.min(1f32).max(0f32);
        }
    }
}

#[inline]
fn sqf(x: f32) -> f32 {
    x * x
}

pub(crate) fn create_spline(p: FilmicSplineParameters) -> FilmicSpline {
    let grey_display = f_powf(0.1845f32, 1.0f32 / (p.output_power));
    let hardness = p.output_power;
    // latitude in %
    let latitude = m_clamp(p.latitude, 0.0f32, 100.0f32) / 100.0f32;

    let white_source = p.white_point_source;
    let black_source = p.black_point_source;
    let dynamic_range = white_source - black_source;

    // luminance after log encoding
    let black_log = 0.0f32; // assumes user set log as in the autotuner
    let grey_log = p.black_point_source.abs() / dynamic_range;
    let white_log = 1.0f32; // assumes user set log as in the autotuner

    let black_display = f_powf(
        m_clamp(p.black_point_target, 0.0f32, p.grey_point_target) / 100.0f32,
        1.0f32 / (p.output_power),
    ); // in %;
    let white_display = f_powf(
        f32::max(p.white_point_target, p.grey_point_target) / 100.0f32,
        1.0f32 / (p.output_power),
    );
    let balance = m_clamp(p.balance, -50.0f32, 50.0f32) / 100.0f32; // in %
    let slope = p.contrast * dynamic_range / 8.0f32;
    let mut min_contrast = 1.0f32; // otherwise, white_display and black_display cannot be reached
                                   // make sure there is enough contrast to be able to construct the top right part of the curve
                                   // make sure there is enough contrast to be able to construct the bottom left part of the curve

    min_contrast = f32::max(
        min_contrast,
        (white_display - grey_display) / (white_log - grey_log),
    );
    if min_contrast.is_nan() || min_contrast.is_infinite() {
        min_contrast = 0.0f32;
    }
    const SAFETY_MARGIN: f32 = 0.01f32;
    min_contrast += SAFETY_MARGIN;
    // we want a slope that depends only on contrast at gray point.
    // let's consider f(x) = (contrast*x+linear_intercept)^hardness
    // f'(x) = contrast * hardness * (contrast*x+linear_intercept)^(hardness-1)
    // linear_intercept = grey_display - (contrast * grey_log);
    // f'(grey_log) = contrast * hardness * (contrast * grey_log + grey_display - (contrast * grey_log))^(hardness-1)
    //              = contrast * hardness * grey_display^(hardness-1)
    // f'(grey_log) = target_contrast <=> contrast = target_contrast / (hardness * grey_display^(hardness-1))
    let mut contrast = slope / (hardness * f_powf(grey_display, hardness - 1.0f32));
    let clamped_contrast = m_clamp(contrast, min_contrast, 100.0f32);
    contrast = clamped_contrast;

    // interception
    let linear_intercept = grey_display - (contrast * grey_log);

    // consider the line of equation y = contrast * x + linear_intercept
    // we want to keep y in [black_display, white_display] (with some safety margin)
    // thus, we compute x values such as y=black_display and y=white_display
    // latitude will influence position of toe and shoulder in the [xmin, xmax] segment
    let xmin = (black_display + SAFETY_MARGIN * (white_display - black_display) - linear_intercept)
        / contrast;
    let xmax = (white_display - SAFETY_MARGIN * (white_display - black_display) - linear_intercept)
        / contrast;

    // nodes for mapping from log encoding to desired target luminance
    // X coordinates
    let mut toe_log = (1.0f32 - latitude) * grey_log + latitude * xmin;
    let mut shoulder_log = (1.0f32 - latitude) * grey_log + latitude * xmax;

    // Apply the highlights/shadows balance as a shift along the contrast slope
    // negative values drag to the left and compress the shadows, on the UI negative is the inverse
    let balance_correction = if balance > 0.0f32 {
        2.0f32 * balance * (shoulder_log - grey_log)
    } else {
        2.0f32 * balance * (grey_log - toe_log)
    };
    toe_log -= balance_correction;
    shoulder_log -= balance_correction;
    toe_log = f32::max(toe_log, xmin);
    shoulder_log = f32::min(shoulder_log, xmax);

    // y coordinates
    let toe_display = toe_log * contrast + linear_intercept;
    let shoulder_display = shoulder_log * contrast + linear_intercept;

    let mut spline = FilmicSpline::default();

    spline.x[0] = black_log;
    spline.x[1] = toe_log;
    spline.x[2] = grey_log;
    spline.x[3] = shoulder_log;
    spline.x[4] = white_log;

    spline.y[0] = black_display;
    spline.y[1] = toe_display;
    spline.y[2] = grey_display;
    spline.y[3] = shoulder_display;
    spline.y[4] = white_display;

    spline.latitude_min = spline.x[1];
    spline.latitude_max = spline.x[3];

    spline.black_source = black_source;
    spline.grey_source = 0.1845f32;
    spline.dynamic_range = dynamic_range;

    spline.saturation = 2.0f32 * p.saturation / 100.0f32 + 1.0f32;

    spline.sigma_toe = f_powf(spline.latitude_min / 3.0f32, 2.0f32);
    spline.sigma_shoulder = f_powf((1.0f32 - spline.latitude_max) / 3.0f32, 2.0f32);

    // let tl = spline.x[1];
    // let tl2 = tl * tl;
    // let Tl3 = tl2 * tl;
    // let Tl4 = Tl3 * Tl;

    // let sl = spline.x[3];
    // let Sl2 = sl * sl;
    // let Sl3 = Sl2 * Sl;
    // let Sl4 = Sl3 * Sl;

    // if type polynomial :
    // y = M5 * x⁴ + M4 * x³ + M3 * x² + M2 * x¹ + M1 * x⁰
    // else if type rational :
    // y = M1 * (M2 * (x - x_0)² + (x - x_0)) / (M2 * (x - x_0)² + (x - x_0) + M3)
    // We then compute M1 to M5 coeffs using the imposed conditions over the curve.
    // M1 to M5 are 3x1 vectors, where each element belongs to a part of the curve.

    // solve the linear central part - affine function
    spline.m2[2] = contrast; // * x¹ (slope)
    spline.m1[2] = spline.y[1] - spline.m2[2] * spline.x[1]; // * x⁰ (offset)
    spline.m3[2] = 0f32; // * x²
    spline.m4[2] = 0f32; // * x³
    spline.m5[2] = 0f32; // * x⁴

    let p1: [f32; 2] = [black_log, black_display];
    let p0: [f32; 2] = [toe_log, toe_display];
    let x = p0[0] - p1[0];
    let y = p0[1] - p1[1];
    let g = contrast;
    let jx = sqf(x * g / y + 1f32).max(4f32);
    let b = g / (2f32 * y) + ((jx - 4f32).sqrt() - 1f32) / (2f32 * x);
    let c = y / g * (b * sqf(x) + x) / (b * sqf(x) + x - (y / g));
    let a = c * g;
    spline.m1[0] = a;
    spline.m2[0] = b;
    spline.m3[0] = c;
    spline.m4[0] = toe_display;

    let p1_0: [f32; 2] = [white_log, white_display];
    let p0_0: [f32; 2] = [shoulder_log, shoulder_display];
    let x = p1_0[0] - p0_0[0];
    let y = p1_0[1] - p0_0[1];
    let g = contrast;
    let b = g / (2f32 * y) + ((sqf(x * g / y + 1f32) - 4f32).sqrt() - 1f32) / (2f32 * x);
    let c = y / g * (b * sqf(x) + x) / (b * sqf(x) + x - (y / g));
    let a = c * g;
    spline.m1[1] = a;
    spline.m2[1] = b;
    spline.m3[1] = c;
    spline.m4[1] = shoulder_display;
    spline
}
