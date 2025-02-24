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
use crate::cms::matrix::{Matrix3f, Vector3f};
use crate::cms::profile::Chromacity;

const SRGB_LUMA_PRIMARIES: [f32; 3] = [0.212639f32, 0.715169f32, 0.072192f32];
const DISPLAY_P3_LUMA_PRIMARIES: [f32; 3] = [0.2289746f32, 0.6917385f32, 0.0792869f32];
const BT2020_LUMA_PRIMARIES: [f32; 3] = [0.2627f32, 0.677998f32, 0.059302f32];

const DISPLAY_P3_PRIMARIES: [Chromacity; 3] = [
    Chromacity {
        x: 0.68f32,
        y: 0.32f32,
    },
    Chromacity {
        x: 0.265f32,
        y: 0.69f32,
    },
    Chromacity {
        x: 0.15f32,
        y: 0.06f32,
    },
];

const SRGB_PRIMARIES: [Chromacity; 3] = [
    Chromacity {
        x: 0.640f32,
        y: 0.330f32,
    },
    Chromacity {
        x: 0.300f32,
        y: 0.600f32,
    },
    Chromacity {
        x: 0.150f32,
        y: 0.060f32,
    },
];

const BT2020_PRIMARIES: [Chromacity; 3] = [
    Chromacity {
        x: 0.7080f32,
        y: 0.2920f32,
    },
    Chromacity {
        x: 0.170032,
        y: 0.7970f32,
    },
    Chromacity {
        x: 0.1310f32,
        y: 0.0460f32,
    },
];

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

    pub(crate) fn primaries_xy(&self) -> [Chromacity; 3] {
        match self {
            GamutColorSpace::Srgb => SRGB_PRIMARIES,
            GamutColorSpace::DisplayP3 => DISPLAY_P3_PRIMARIES,
            GamutColorSpace::Bt2020 => BT2020_PRIMARIES,
        }
    }
}

#[inline]
fn xy_to_xyz(xy: Chromacity) -> Vector3f {
    Vector3f {
        v: [xy.x / xy.y, 1f32, (1f32 - xy.x - xy.y) / xy.y],
    }
}

#[inline]
fn get_primaries_xyz(primaries_xy: [Chromacity; 3]) -> Matrix3f {
    let r = xy_to_xyz(primaries_xy[0]);
    let g = xy_to_xyz(primaries_xy[1]);
    let b = xy_to_xyz(primaries_xy[2]);
    Matrix3f {
        v: [
            [r.v[0], g.v[0], b.v[0]],
            [r.v[1], g.v[1], b.v[1]],
            [r.v[2], g.v[2], b.v[2]],
        ],
    }
}

#[inline]
pub fn get_white_point_xyz(xy: Chromacity) -> Vector3f {
    xy_to_xyz(xy)
}

pub(crate) fn gamut_to_xyz(
    primaries_xy: [Chromacity; 3],
    white_point: Chromacity,
) -> Option<Matrix3f> {
    let xyz_matrix = get_primaries_xyz(primaries_xy);
    let wp = get_white_point_xyz(white_point);
    let inverted_xyz = xyz_matrix.inverse()?;
    let s = inverted_xyz.mul_vector(wp);
    let mut v = xyz_matrix.mul_row_vector::<0>(s);
    v = v.mul_row_vector::<1>(s);
    v = v.mul_row_vector::<2>(s);
    Some(v)
}
