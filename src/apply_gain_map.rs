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
use crate::iso_gain_map::{GainLUT, GainMap};
use crate::mappers::Rgb;
use crate::mlaf::mlaf;
use crate::{ForgeError, GainImage, GainImageMut};
use moxcms::{ColorProfile, Matrix3f};
use num_traits::AsPrimitive;
use std::fmt::{Debug, Display};

/// Applies gain map on 8 bit RGB image
///
/// # Arguments
///
/// * `image`: Source image
/// * `image_icc_profile`: Source image ICC profile
/// * `dst_image`: Destination image
/// * `destination_profile`: Destination image ICC profile
/// * `gain_map_image`: Gain map
/// * `gain_map_icc_profile`: Gain map ICC profile
/// * `gain_map`: Gain map metadata
/// * `weight`: gain map weight
///
/// returns: Result<(), ForgeError>
///
#[allow(clippy::too_many_arguments)]
pub fn apply_gain_map_rgb(
    image: &GainImage<u8, 3>,
    image_icc_profile: &Option<ColorProfile>,
    dst_image: &mut GainImageMut<u8, 3>,
    destination_profile: &ColorProfile,
    gain_map_image: &GainImage<u8, 3>,
    gain_map_icc_profile: &Option<ColorProfile>,
    gain_map: GainMap,
    weight: f32,
) -> Result<(), ForgeError> {
    apply_gain_map::<u8, 3, 3, 256, 8192, 8>(
        image,
        dst_image,
        image_icc_profile,
        gain_map_image,
        gain_map_icc_profile,
        destination_profile,
        gain_map,
        weight,
    )
}

/// Applies gain map on 8 bit RGBA image
///
/// # Arguments
///
/// * `image`: Source image
/// * `image_icc_profile`: Source image ICC profile
/// * `dst_image`: Destination image
/// * `destination_profile`: Destination image ICC profile
/// * `gain_map_image`: Gain map
/// * `gain_map_icc_profile`: Gain map ICC profile
/// * `gain_map`: Gain map metadata
/// * `weight`: gain map weight
///
/// returns: Result<(), ForgeError>
///
#[allow(clippy::too_many_arguments)]
pub fn apply_gain_map_rgba(
    image: &GainImage<u8, 4>,
    image_icc_profile: &Option<ColorProfile>,
    dst_image: &mut GainImageMut<u8, 4>,
    destination_profile: &ColorProfile,
    gain_map_image: &GainImage<u8, 3>,
    gain_map_icc_profile: &Option<ColorProfile>,
    gain_map: GainMap,
    weight: f32,
) -> Result<(), ForgeError> {
    apply_gain_map::<u8, 4, 3, 256, 8192, 8>(
        image,
        dst_image,
        image_icc_profile,
        gain_map_image,
        gain_map_icc_profile,
        destination_profile,
        gain_map,
        weight,
    )
}

/// Applies gain map on 10 bit RGB image
///
/// # Arguments
///
/// * `image`: Source image
/// * `image_icc_profile`: Source image ICC profile
/// * `dst_image`: Destination image
/// * `destination_profile`: Destination image ICC profile
/// * `gain_map_image`: Gain map
/// * `gain_map_icc_profile`: Gain map ICC profile
/// * `gain_map`: Gain map metadata
/// * `weight`: gain map weight
///
/// returns: Result<(), ForgeError>
///
#[allow(clippy::too_many_arguments)]
pub fn apply_gain_map_rgb10(
    image: &GainImage<u16, 3>,
    image_icc_profile: &Option<ColorProfile>,
    dst_image: &mut GainImageMut<u16, 3>,
    destination_profile: &ColorProfile,
    gain_map_image: &GainImage<u16, 3>,
    gain_map_icc_profile: &Option<ColorProfile>,
    gain_map: GainMap,
    weight: f32,
) -> Result<(), ForgeError> {
    apply_gain_map::<u16, 3, 3, 1024, 8192, 10>(
        image,
        dst_image,
        image_icc_profile,
        gain_map_image,
        gain_map_icc_profile,
        destination_profile,
        gain_map,
        weight,
    )
}

/// Applies gain map on 10 bit RGBA image
///
/// # Arguments
///
/// * `image`: Source image
/// * `image_icc_profile`: Source image ICC profile
/// * `dst_image`: Destination image
/// * `destination_profile`: Destination image ICC profile
/// * `gain_map_image`: Gain map
/// * `gain_map_icc_profile`: Gain map ICC profile
/// * `gain_map`: Gain map metadata
/// * `weight`: gain map weight
///
/// returns: Result<(), ForgeError>
///
#[allow(clippy::too_many_arguments)]
pub fn apply_gain_map_rgba10(
    image: &GainImage<u16, 4>,
    image_icc_profile: &Option<ColorProfile>,
    dst_image: &mut GainImageMut<u16, 4>,
    destination_profile: &ColorProfile,
    gain_map_image: &GainImage<u16, 3>,
    gain_map_icc_profile: &Option<ColorProfile>,
    gain_map: GainMap,
    weight: f32,
) -> Result<(), ForgeError> {
    apply_gain_map::<u16, 4, 3, 1024, 8192, 10>(
        image,
        dst_image,
        image_icc_profile,
        gain_map_image,
        gain_map_icc_profile,
        destination_profile,
        gain_map,
        weight,
    )
}

/// Applies gain map on 12 bit RGB image
///
/// # Arguments
///
/// * `image`: Source image
/// * `image_icc_profile`: Source image ICC profile
/// * `dst_image`: Destination image
/// * `destination_profile`: Destination image ICC profile
/// * `gain_map_image`: Gain map
/// * `gain_map_icc_profile`: Gain map ICC profile
/// * `gain_map`: Gain map metadata
/// * `weight`: gain map weight
///
/// returns: Result<(), ForgeError>
///
#[allow(clippy::too_many_arguments)]
pub fn apply_gain_map_rgb12(
    image: &GainImage<u16, 3>,
    image_icc_profile: &Option<ColorProfile>,
    dst_image: &mut GainImageMut<u16, 3>,
    destination_profile: &ColorProfile,
    gain_map_image: &GainImage<u16, 3>,
    gain_map_icc_profile: &Option<ColorProfile>,
    gain_map: GainMap,
    weight: f32,
) -> Result<(), ForgeError> {
    apply_gain_map::<u16, 3, 3, 4096, 16384, 12>(
        image,
        dst_image,
        image_icc_profile,
        gain_map_image,
        gain_map_icc_profile,
        destination_profile,
        gain_map,
        weight,
    )
}

/// Applies gain map on 12 bit RGBA image
///
/// # Arguments
///
/// * `image`: Source image
/// * `image_icc_profile`: Source image ICC profile
/// * `dst_image`: Destination image
/// * `destination_profile`: Destination image ICC profile
/// * `gain_map_image`: Gain map
/// * `gain_map_icc_profile`: Gain map ICC profile
/// * `gain_map`: Gain map metadata
/// * `weight`: gain map weight
///
/// returns: Result<(), ForgeError>
///
#[allow(clippy::too_many_arguments)]
pub fn apply_gain_map_rgba12(
    image: &GainImage<u16, 4>,
    image_icc_profile: &Option<ColorProfile>,
    dst_image: &mut GainImageMut<u16, 4>,
    destination_profile: &ColorProfile,
    gain_map_image: &GainImage<u16, 3>,
    gain_map_icc_profile: &Option<ColorProfile>,
    gain_map: GainMap,
    weight: f32,
) -> Result<(), ForgeError> {
    apply_gain_map::<u16, 4, 3, 4096, 16384, 12>(
        image,
        dst_image,
        image_icc_profile,
        gain_map_image,
        gain_map_icc_profile,
        destination_profile,
        gain_map,
        weight,
    )
}

/// Applies gain map on 16 bit RGB image
///
/// # Arguments
///
/// * `image`: Source image
/// * `image_icc_profile`: Source image ICC profile
/// * `dst_image`: Destination image
/// * `destination_profile`: Destination image ICC profile
/// * `gain_map_image`: Gain map
/// * `gain_map_icc_profile`: Gain map ICC profile
/// * `gain_map`: Gain map metadata
/// * `weight`: gain map weight
///
/// returns: Result<(), ForgeError>
///
#[allow(clippy::too_many_arguments)]
pub fn apply_gain_map_rgb16(
    image: &GainImage<u16, 3>,
    image_icc_profile: &Option<ColorProfile>,
    dst_image: &mut GainImageMut<u16, 3>,
    destination_profile: &ColorProfile,
    gain_map_image: &GainImage<u16, 3>,
    gain_map_icc_profile: &Option<ColorProfile>,
    gain_map: GainMap,
    weight: f32,
) -> Result<(), ForgeError> {
    apply_gain_map::<u16, 3, 3, 65536, 65536, 16>(
        image,
        dst_image,
        image_icc_profile,
        gain_map_image,
        gain_map_icc_profile,
        destination_profile,
        gain_map,
        weight,
    )
}

/// Applies gain map on 16 bit RGBA image
///
/// # Arguments
///
/// * `image`: Source image
/// * `image_icc_profile`: Source image ICC profile
/// * `dst_image`: Destination image
/// * `destination_profile`: Destination image ICC profile
/// * `gain_map_image`: Gain map
/// * `gain_map_icc_profile`: Gain map ICC profile
/// * `gain_map`: Gain map metadata
/// * `weight`: gain map weight
///
/// returns: Result<(), ForgeError>
///
#[allow(clippy::too_many_arguments)]
pub fn apply_gain_map_rgba16(
    image: &GainImage<u16, 4>,
    image_icc_profile: &Option<ColorProfile>,
    dst_image: &mut GainImageMut<u16, 4>,
    destination_profile: &ColorProfile,
    gain_map_image: &GainImage<u16, 3>,
    gain_map_icc_profile: &Option<ColorProfile>,
    gain_map: GainMap,
    weight: f32,
) -> Result<(), ForgeError> {
    apply_gain_map::<u16, 4, 3, 65536, 65536, 16>(
        image,
        dst_image,
        image_icc_profile,
        gain_map_image,
        gain_map_icc_profile,
        destination_profile,
        gain_map,
        weight,
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_gain_map<
    T: Copy + 'static + Default + Debug + AsPrimitive<usize> + Display,
    const N: usize,
    const GAIN_N: usize,
    const LIN_DEPTH: usize,
    const GAMMA_DEPTH: usize,
    const BIT_DEPTH: usize,
>(
    image: &GainImage<T, N>,
    dst_image: &mut GainImageMut<T, N>,
    image_icc_profile: &Option<ColorProfile>,
    gain_map_image: &GainImage<T, GAIN_N>,
    gain_map_icc_profile: &Option<ColorProfile>,
    destination_gamut: &ColorProfile,
    gain_map: GainMap,
    weight: f32,
) -> Result<(), ForgeError>
where
    f32: AsPrimitive<T>,
    u32: AsPrimitive<T>,
{
    image.check_layout()?;
    dst_image.check_layout()?;
    gain_map_image.check_layout()?;
    image.size_matches_arb::<GAIN_N>(gain_map_image)?;
    image.size_matches_mut(dst_image)?;
    assert!(GAMMA_DEPTH == 8192 || GAMMA_DEPTH == 16384 || GAMMA_DEPTH == 65536);
    assert!(BIT_DEPTH == 8 || BIT_DEPTH == 10 || BIT_DEPTH == 12 || BIT_DEPTH == 16);

    let transform = if let Some(icc) = image_icc_profile {
        icc.transform_matrix(destination_gamut)
    } else {
        None
    };

    let lut = GainLUT::<LIN_DEPTH>::new(gain_map, weight);

    let output_gamma_map_r: Box<[T; 65536]> = destination_gamut
        .red_trc
        .clone()
        .ok_or(ForgeError::InvalidIcc)
        .and_then(|x| {
            destination_gamut
                .build_gamma_table::<T, 65536, GAMMA_DEPTH, BIT_DEPTH>(&Some(x))
                .map_err(|_| ForgeError::InvalidIcc)
        })?;
    let output_gamma_map_g: Box<[T; 65536]> = destination_gamut
        .green_trc
        .clone()
        .ok_or(ForgeError::InvalidIcc)
        .and_then(|x| {
            destination_gamut
                .build_gamma_table::<T, 65536, GAMMA_DEPTH, BIT_DEPTH>(&Some(x))
                .map_err(|_| ForgeError::InvalidIcc)
        })?;
    let output_gamma_map_b: Box<[T; 65536]> = destination_gamut
        .blue_trc
        .clone()
        .ok_or(ForgeError::InvalidIcc)
        .and_then(|x| {
            destination_gamut
                .build_gamma_table::<T, 65536, GAMMA_DEPTH, BIT_DEPTH>(&Some(x))
                .map_err(|_| ForgeError::InvalidIcc)
        })?;

    let temporary_srgb = ColorProfile::new_srgb();

    let img_profile = image_icc_profile
        .as_ref()
        .or(Some(&temporary_srgb))
        .ok_or(ForgeError::InvalidIcc)?;

    let image_linearize_map_r = img_profile
        .build_r_linearize_table::<LIN_DEPTH>()
        .map_err(|_| ForgeError::InvalidIcc)?;
    let image_linearize_map_g = img_profile
        .build_g_linearize_table::<LIN_DEPTH>()
        .map_err(|_| ForgeError::InvalidIcc)?;
    let image_linearize_map_b = img_profile
        .build_b_linearize_table::<LIN_DEPTH>()
        .map_err(|_| ForgeError::InvalidIcc)?;

    let gain_map_icc_profile = (if gain_map.use_base_cg {
        Some(image_icc_profile.as_ref().unwrap_or(&temporary_srgb))
    } else {
        gain_map_icc_profile.as_ref()
    })
    .ok_or(ForgeError::InvalidGainMapConfiguration)?;

    let gain_image_linearize_map_r = gain_map_icc_profile
        .build_r_linearize_table::<LIN_DEPTH>()
        .map_err(|_| ForgeError::InvalidIcc)?;
    let gain_image_linearize_map_g = gain_map_icc_profile
        .build_g_linearize_table::<LIN_DEPTH>()
        .map_err(|_| ForgeError::InvalidIcc)?;
    let gain_image_linearize_map_b = gain_map_icc_profile
        .build_b_linearize_table::<LIN_DEPTH>()
        .map_err(|_| ForgeError::InvalidIcc)?;

    let mut linearized_image_content = vec![0f32; image.width * N];
    let mut linearized_gain_content = vec![0f32; image.width * GAIN_N];
    let mut working_lane = vec![0f32; image.width * N];

    let src_stride = image.row_stride();
    let dst_stride = dst_image.row_stride();
    let dst_image = dst_image.data.borrow_mut();
    let width = image.width;

    for ((gain_lane, image_lane), dst_lane) in gain_map_image
        .data
        .as_ref()
        .chunks_exact(gain_map_image.row_stride())
        .zip(image.data.as_ref().chunks_exact(src_stride))
        .zip(dst_image.chunks_exact_mut(dst_stride))
    {
        for (src, dst) in gain_lane[..width * GAIN_N]
            .chunks_exact(3)
            .zip(linearized_gain_content.chunks_exact_mut(GAIN_N))
        {
            dst[0] = gain_image_linearize_map_r[src[0].as_()];
            dst[1] = gain_image_linearize_map_g[src[1].as_()];
            dst[2] = gain_image_linearize_map_b[src[2].as_()];
        }

        for (src, dst) in image_lane[..width * N]
            .chunks_exact(N)
            .zip(linearized_image_content.chunks_exact_mut(N))
        {
            dst[0] = image_linearize_map_r[src[0].as_()];
            dst[1] = image_linearize_map_g[src[1].as_()];
            dst[2] = image_linearize_map_b[src[2].as_()];
            if N == 4 {
                dst[3] = f32::from_bits(src[3].as_() as u32);
            }
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
                dst[3] = src[3];
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

        let gamma_scale = (GAMMA_DEPTH - 1) as f32;

        for (dst, src) in dst_lane
            .chunks_exact_mut(N)
            .zip(working_lane.chunks_exact(N))
        {
            let r = mlaf(0.5f32, src[0], gamma_scale) as u16;
            let g = mlaf(0.5f32, src[1], gamma_scale) as u16;
            let b = mlaf(0.5f32, src[2], gamma_scale) as u16;
            dst[0] = output_gamma_map_r[r as usize];
            dst[1] = output_gamma_map_g[g as usize];
            dst[2] = output_gamma_map_b[b as usize];
            if N == 4 {
                dst[3] = src[3].to_bits().as_();
            }
        }
    }

    Ok(())
}
