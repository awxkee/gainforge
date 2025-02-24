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
mod mlaf;
mod parse;

use crate::parse::find_iso_chunks;
use gainforge::{
    apply_gain_map_rgb, make_gainmap_weight, ColorProfile, GamutColorSpace, IsoGainMap,
};
use image::codecs::jpeg::JpegDecoder;
use image::ImageDecoder;
use jpeg_decoder::Decoder;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use turbojpeg::PixelFormat;

pub struct AssociatedImages {
    pub image: Vec<u8>,
    pub gain_map: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub icc_profile: Option<ColorProfile>,
    pub gain_map_icc_profile: Option<ColorProfile>,
    pub metadata: IsoGainMap,
}

fn extract_images(file_path: &str) -> AssociatedImages {
    let file = File::open(file_path).expect("Failed to open file");

    let mut decoder = Decoder::new(BufReader::new(file));

    // Decode first image (Primary)
    let primary_image = decoder.decode().expect("Failed to decode primary image");
    let primary_metadata = decoder.info().expect("No metadata found");

    let image_icc = if let Some(icc) = decoder.icc_profile() {
        match ColorProfile::new_from_slice(&icc) {
            Ok(a0) => Some(a0),
            Err(_) => None,
        }
    } else {
        None
    };

    let last_pos = decoder.reader.seek(SeekFrom::Current(0)).unwrap();
    let file = File::open(file_path).expect("Failed to open file");
    let mut reader2 = BufReader::new(file);
    reader2.seek(SeekFrom::Start(last_pos)).unwrap();
    let mut dst_vec = Vec::new();
    reader2.read_to_end(&mut dst_vec).unwrap();

    // Read the second image from JPEG file

    let mut decoder = JpegDecoder::new(Cursor::new(dst_vec.to_vec())).unwrap();
    let mut img1 = vec![0u8; decoder.total_bytes() as usize];
    decoder.read_image(&mut img1).unwrap();

    let gain_map_image_info =
        turbojpeg::decompress(&dst_vec, PixelFormat::RGB).expect("Failed to decompress image");

    let mut decoder2 = Decoder::new(Cursor::new(dst_vec.to_vec()));

    let gain_map_icc = if let Some(icc) = decoder2.icc_profile() {
        match ColorProfile::new_from_slice(&icc) {
            Ok(a0) => Some(a0),
            Err(_) => None,
        }
    } else {
        None
    };

    let mut gm_reader = BufReader::new(File::open(file_path).expect("Failed to open file"));
    let chunk = find_iso_chunks(&mut gm_reader).unwrap();

    AssociatedImages {
        image: primary_image,
        gain_map: gain_map_image_info.pixels,
        width: primary_metadata.width as usize,
        height: primary_metadata.height as usize,
        icc_profile: image_icc,
        gain_map_icc_profile: gain_map_icc,
        metadata: chunk,
    }
}

fn main() {
    let associated = extract_images("./assets/04.jpg");
    // decoder.read_info().unwrap();
    // let img = image::ImageReader::open("./assets/hdr.avif")
    //     .unwrap()
    //     .decode()
    //     .unwrap();
    // let rgb = img.to_rgb8();
    //
    // let tone_mapper = create_tone_mapper_rgb(
    //     GainHDRMetadata::new(1000f32, 250f32),
    //     HdrTransferFunction::Pq,
    //     GamutColorSpace::Bt2020,
    //     TransferFunction::Srgb,
    //     GamutColorSpace::Srgb,
    //     ToneMappingMethod::Alu,
    // );
    // let dims = rgb.dimensions();
    // let mut dst = vec![0u8; rgb.len()];
    // for (src, dst) in rgb
    //     .chunks_exact(rgb.dimensions().0 as usize * 3)
    //     .zip(dst.chunks_exact_mut(rgb.dimensions().0 as usize * 3))
    // {
    //     tone_mapper.tonemap_lane(src, dst).unwrap();
    // }

    let gainmap = associated.metadata.to_gain_map();

    let display_boost = 1.5f32;
    let gainmap_weight = make_gainmap_weight(gainmap, display_boost);
    println!("weight {}", gainmap_weight);

    let instant = std::time::Instant::now();

    let dst = apply_gain_map_rgb(
        &associated.image,
        associated.width * 3,
        &associated.icc_profile,
        &associated.gain_map,
        associated.width * 3,
        &associated.icc_profile,
        GamutColorSpace::Srgb,
        associated.width,
        associated.height,
        gainmap,
        gainmap_weight,
    )
    .unwrap();

    println!("Time {:?}", instant.elapsed());

    image::save_buffer(
        "processed_alu10.jpg",
        &dst,
        associated.width as u32,
        associated.height as u32,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
}
