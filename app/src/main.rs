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

use gainforge::{
    create_tone_mapper_rgb, GainHDRMetadata, GamutColorSpace, HdrTransferFunction,
    ToneMappingMethod, TransferFunction,
};

fn main() {
    let img = image::ImageReader::open("./assets/hdr.avif")
        .unwrap()
        .decode()
        .unwrap();
    let rgb = img.to_rgb8();

    let tone_mapper = create_tone_mapper_rgb(
        GainHDRMetadata::new(1000f32, 250f32),
        HdrTransferFunction::Pq,
        GamutColorSpace::Bt2020,
        TransferFunction::Srgb,
        GamutColorSpace::Srgb,
        ToneMappingMethod::Alu,
    );
    let dims = rgb.dimensions();
    let mut dst = vec![0u8; rgb.len()];
    for (src, dst) in rgb
        .chunks_exact(rgb.dimensions().0 as usize * 3)
        .zip(dst.chunks_exact_mut(rgb.dimensions().0 as usize * 3))
    {
        tone_mapper.tonemap_lane(src, dst).unwrap();
    }

    image::save_buffer(
        "processed_alu.jpg",
        &dst,
        dims.0,
        dims.1,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
}
