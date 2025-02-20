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
use std::ops::Mul;

const SRGB_LUMA_PRIMARIES: [f32; 3] = [0.212639f32, 0.715169f32, 0.072192f32];
const DISPLAY_P3_LUMA_PRIMARIES: [f32; 3] = [0.2289746f32, 0.6917385f32, 0.0792869f32];
const BT2020_LUMA_PRIMARIES: [f32; 3] = [0.2627f32, 0.677998f32, 0.059302f32];

#[derive(Clone, Debug, Copy)]
pub(crate) struct Xyz {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) z: f32,
}

impl Xyz {
    pub(crate) fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

impl Mul<f32> for Xyz {
    type Output = Xyz;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<Xyz> for Xyz {
    type Output = Xyz;

    #[inline]
    fn mul(self, rhs: Xyz) -> Self::Output {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

#[derive(Clone, Debug, Copy)]
pub struct Xy {
    x: f32,
    y: f32,
}

const DISPLAY_P3_PRIMARIES: [Xy; 3] = [
    Xy {
        x: 0.68f32,
        y: 0.32f32,
    },
    Xy {
        x: 0.265f32,
        y: 0.69f32,
    },
    Xy {
        x: 0.15f32,
        y: 0.06f32,
    },
];

const SRGB_PRIMARIES: [Xy; 3] = [
    Xy {
        x: 0.640f32,
        y: 0.330f32,
    },
    Xy {
        x: 0.300f32,
        y: 0.600f32,
    },
    Xy {
        x: 0.150f32,
        y: 0.060f32,
    },
];

const BT2020_PRIMARIES: [Xy; 3] = [
    Xy {
        x: 0.7080f32,
        y: 0.2920f32,
    },
    Xy {
        x: 0.170032,
        y: 0.7970f32,
    },
    Xy {
        x: 0.1310f32,
        y: 0.0460f32,
    },
];

pub(crate) const ILLUMINANT_D65: Xy = Xy {
    x: 0.3127,
    y: 0.3290,
};

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum GamutColorSpace {
    Srgb,
    DisplayP3,
    Bt2020,
}

impl GamutColorSpace {
    pub fn luma_primaries(&self) -> [f32; 3] {
        match self {
            GamutColorSpace::Srgb => SRGB_LUMA_PRIMARIES,
            GamutColorSpace::DisplayP3 => DISPLAY_P3_LUMA_PRIMARIES,
            GamutColorSpace::Bt2020 => BT2020_LUMA_PRIMARIES,
        }
    }

    pub(crate) fn primaries_xy(&self) -> [Xy; 3] {
        match self {
            GamutColorSpace::Srgb => SRGB_PRIMARIES,
            GamutColorSpace::DisplayP3 => DISPLAY_P3_PRIMARIES,
            GamutColorSpace::Bt2020 => BT2020_PRIMARIES,
        }
    }
}

#[inline]
fn xy_to_xyz(xy: Xy) -> Xyz {
    Xyz {
        x: xy.x / xy.y,
        y: 1f32,
        z: (1f32 - xy.x - xy.y) / xy.y,
    }
}

#[inline]
fn get_primaries_xyz(primaries_xy: [Xy; 3]) -> [Xyz; 3] {
    [
        xy_to_xyz(primaries_xy[0]),
        xy_to_xyz(primaries_xy[1]),
        xy_to_xyz(primaries_xy[2]),
    ]
}

#[inline]
fn primaries_determinant(v: [Xyz; 3]) -> f32 {
    let a0 = v[0].x * v[1].y * v[2].z;
    let a1 = v[0].y * v[1].z * v[2].x;
    let a2 = v[0].z * v[1].x * v[2].y;

    let s0 = v[0].z * v[1].y * v[2].x;
    let s1 = v[0].y * v[1].x * v[2].z;
    let s2 = v[0].x * v[1].z * v[2].y;

    let j = a0 + a1 + a2 - s0 - s1 - s2;
    assert_ne!(j, 0f32);
    j
}

#[inline]
pub fn get_white_point_xyz(xy: Xy) -> Xyz {
    xy_to_xyz(xy)
}

#[inline]
pub(crate) fn inverse(v: [Xyz; 3]) -> [Xyz; 3] {
    let det = 1. / primaries_determinant(v);
    let a = v[0].x;
    let b = v[0].y;
    let c = v[0].z;
    let d = v[1].x;
    let e = v[1].y;
    let f = v[1].z;
    let g = v[2].x;
    let h = v[2].y;
    let i = v[2].z;

    [
        Xyz::new(
            (e * i - f * h) * det,
            (c * h - b * i) * det,
            (b * f - c * e) * det,
        ),
        Xyz::new(
            (f * g - d * i) * det,
            (a * i - c * g) * det,
            (c * d - a * f) * det,
        ),
        Xyz::new(
            (d * h - e * g) * det,
            (b * g - a * h) * det,
            (a * e - b * d) * det,
        ),
    ]
}

#[inline]
pub(crate) fn mat_mul_vector_column(m: [Xyz; 3], v: Xyz) -> Xyz {
    let x = m[0].x * v.x + m[1].x * v.y + m[2].x * v.z;
    let y = m[0].y * v.x + m[1].y * v.y + m[2].y * v.z;
    let z = m[0].z * v.x + m[1].z * v.y + m[2].z * v.z;
    Xyz { x, y, z }
}

#[inline]
pub(crate) fn mat_mul(m0: [Xyz; 3], m1: [Xyz; 3]) -> [Xyz; 3] {
    [
        mat_mul_vector_column(m0, m1[0]),
        mat_mul_vector_column(m0, m1[1]),
        mat_mul_vector_column(m0, m1[2]),
    ]
}

#[inline]
pub(crate) fn to_column_wise(v: [Xyz; 3]) -> [Xyz; 3] {
    [
        Xyz {
            x: v[0].x,
            y: v[1].x,
            z: v[2].x,
        },
        Xyz {
            x: v[0].y,
            y: v[1].y,
            z: v[2].y,
        },
        Xyz {
            x: v[0].z,
            y: v[1].z,
            z: v[2].z,
        },
    ]
}

pub(crate) fn gamut_to_xyz(primaries_xy: [Xy; 3], white_point: Xy) -> [Xyz; 3] {
    let mut xyz_matrix = get_primaries_xyz(primaries_xy);
    let wp = get_white_point_xyz(white_point);
    let inverted_xyz_matrix = inverse(xyz_matrix);
    let s = mat_mul_vector_column(inverted_xyz_matrix, wp);
    xyz_matrix[0] = xyz_matrix[0] * s.x;
    xyz_matrix[1] = xyz_matrix[1] * s.y;
    xyz_matrix[2] = xyz_matrix[2] * s.z;
    xyz_matrix
}
