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
use crate::cms::Matrix3f;
use crate::iso_gain_map::{GainLUT, GainMap};
use crate::mappers::Rgb;
use crate::mlaf::mlaf;
use crate::{ColorProfile, GamutColorSpace, TransferFunction};

#[allow(clippy::too_many_arguments)]
pub fn apply_gain_map_rgb(
    image: &[u8],
    stride: usize,
    image_icc_profile: &Option<ColorProfile>,
    gain_map_image: &[u8],
    gain_map_image_stride: usize,
    gain_map_icc_profile: &Option<ColorProfile>,
    destination_gamut: GamutColorSpace,
    width: usize,
    height: usize,
    gain_map: GainMap,
    weight: f32,
) -> Option<Vec<u8>> {
    apply_gain_map::<3, 3>(
        image,
        stride,
        image_icc_profile,
        gain_map_image,
        gain_map_image_stride,
        gain_map_icc_profile,
        destination_gamut,
        width,
        height,
        gain_map,
        weight,
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_gain_map<const N: usize, const GAIN_N: usize>(
    image: &[u8],
    stride: usize,
    image_icc_profile: &Option<ColorProfile>,
    gain_map_image: &[u8],
    gain_map_image_stride: usize,
    gain_map_icc_profile: &Option<ColorProfile>,
    destination_gamut: GamutColorSpace,
    width: usize,
    height: usize,
    gain_map: GainMap,
    weight: f32,
) -> Option<Vec<u8>> {
    let gain_image_linearize_map_r: Box<[f32; 256]>;
    let gain_image_linearize_map_g: Box<[f32; 256]>;
    let gain_image_linearize_map_b: Box<[f32; 256]>;

    let target_gamut = match destination_gamut {
        GamutColorSpace::Srgb => ColorProfile::new_srgb(),
        GamutColorSpace::DisplayP3 => ColorProfile::new_display_p3(),
        GamutColorSpace::Bt2020 => ColorProfile::new_bt2020(),
    };

    let transform = if let Some(icc) = image_icc_profile {
        icc.rgb_to_xyz_other_xyz_rgb(&target_gamut)
    } else {
        None
    };

    let lut = GainLUT::<256>::new(gain_map, weight);

    let image_linearize_map_r: Box<[f32; 256]>;
    let image_linearize_map_g: Box<[f32; 256]>;
    let image_linearize_map_b: Box<[f32; 256]>;

    let output_gamma_map_r: Box<[u8; 65536]> = match target_gamut.red_trc.clone() {
        None => return None,
        Some(trc) => trc.build_gamma_table::<u8, 65536, 8192, 8>().unwrap(),
    };
    let output_gamma_map_g: Box<[u8; 65536]> = match target_gamut.green_trc.clone() {
        None => return None,
        Some(trc) => trc.build_gamma_table::<u8, 65536, 8192, 8>().unwrap(),
    };
    let output_gamma_map_b: Box<[u8; 65536]> = match target_gamut.blue_trc.clone() {
        None => return None,
        Some(trc) => trc.build_gamma_table::<u8, 65536, 8192, 8>().unwrap(),
    };

    match image_icc_profile {
        None => {
            let srgb = TransferFunction::Srgb.generate_linear_table_u8();
            image_linearize_map_r = srgb.clone();
            image_linearize_map_g = srgb.clone();
            image_linearize_map_b = srgb;
        }
        Some(icc) => {
            image_linearize_map_r = icc.build_r_linearize_table::<256>()?;
            image_linearize_map_g = icc.build_g_linearize_table::<256>()?;
            image_linearize_map_b = icc.build_b_linearize_table::<256>()?;
        }
    }

    match gain_map_icc_profile {
        None => {
            let srgb = TransferFunction::Rec709.generate_linear_table_u8();
            gain_image_linearize_map_r = srgb.clone();
            gain_image_linearize_map_g = srgb.clone();
            gain_image_linearize_map_b = srgb;
        }
        Some(icc) => {
            gain_image_linearize_map_r = icc.build_r_linearize_table::<256>()?;
            gain_image_linearize_map_g = icc.build_g_linearize_table::<256>()?;
            gain_image_linearize_map_b = icc.build_b_linearize_table::<256>()?;
        }
    }

    let mut linearized_image_content = vec![0f32; width * N];
    let mut linearized_gain_content = vec![0f32; width * GAIN_N];
    let mut working_lane = vec![0f32; width * N];

    let mut dst_image = vec![0u8; width * height * N];

    for ((gain_lane, image_lane), dst_lane) in gain_map_image
        .chunks_exact(gain_map_image_stride)
        .zip(image.chunks_exact(stride))
        .zip(dst_image.chunks_exact_mut(width * N))
    {
        for (src, dst) in gain_lane[..width * GAIN_N]
            .chunks_exact(3)
            .zip(linearized_gain_content.chunks_exact_mut(GAIN_N))
        {
            dst[0] = gain_image_linearize_map_r[src[0] as usize];
            dst[1] = gain_image_linearize_map_g[src[1] as usize];
            dst[2] = gain_image_linearize_map_b[src[2] as usize];
        }

        for (src, dst) in image_lane[..width * N]
            .chunks_exact(N)
            .zip(linearized_image_content.chunks_exact_mut(N))
        {
            dst[0] = image_linearize_map_r[src[0] as usize];
            dst[1] = image_linearize_map_g[src[1] as usize];
            dst[2] = image_linearize_map_b[src[2] as usize];
        }

        for ((gain, src), dst) in linearized_gain_content
            .chunks_exact(GAIN_N)
            .zip(linearized_image_content.chunks_exact(N))
            .zip(working_lane.chunks_exact_mut(N))
        {
            let applied_gain = lut.apply_gain(
                Rgb {
                    r: src[0],
                    g: src[1],
                    b: src[2],
                },
                Rgb {
                    r: gain[0],
                    g: gain[1],
                    b: gain[2],
                },
            );
            dst[0] = applied_gain.r;
            dst[1] = applied_gain.g;
            dst[2] = applied_gain.b;
            if N == 4 {
                dst[3] = f32::from_bits(255u32);
            }
        }

        if let Some(transform) = transform {
            let is_identity_transform = transform.test_equality(Matrix3f::IDENTITY);
            if !is_identity_transform {
                for chunk in working_lane.chunks_exact_mut(N) {
                    chunk[0] = mlaf(
                        mlaf(chunk[0] * transform.v[0][0], chunk[1], transform.v[0][1]),
                        chunk[2],
                        transform.v[0][2],
                    )
                    .min(1f32)
                    .max(0f32);

                    chunk[1] = mlaf(
                        mlaf(chunk[0] * transform.v[1][0], chunk[1], transform.v[1][1]),
                        chunk[2],
                        transform.v[1][2],
                    )
                    .min(1f32)
                    .max(0f32);

                    chunk[2] = mlaf(
                        mlaf(chunk[0] * transform.v[2][0], chunk[1], transform.v[2][1]),
                        chunk[2],
                        transform.v[2][2],
                    )
                    .min(1f32)
                    .max(0f32);
                }
            }
        }

        for (dst, src) in dst_lane
            .chunks_exact_mut(N)
            .zip(working_lane.chunks_exact(N))
        {
            let r = mlaf(0.5f32, src[0], 8191f32) as u16;
            let g = mlaf(0.5f32, src[1], 8191f32) as u16;
            let b = mlaf(0.5f32, src[2], 8191f32) as u16;
            dst[0] = output_gamma_map_r[r as usize];
            dst[1] = output_gamma_map_g[g as usize];
            dst[2] = output_gamma_map_b[b as usize];
            if N == 4 {
                dst[3] = src[3].to_bits() as u8;
            }
        }
    }

    Some(dst_image)
}
