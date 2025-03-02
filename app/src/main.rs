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

use gainforge::{
    apply_gain_map_rgb, apply_gain_map_rgb10, apply_gain_map_rgb12, apply_gain_map_rgb16,
    make_gainmap_weight, GainImage, GainImageMut, GamutColorSpace, IsoGainMap, MpfInfo,
};
use moxcms::ColorProfile;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom};
use zune_jpeg::JpegDecoder;

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

    let mut reader = BufReader::new(file);

    let mut decoder = JpegDecoder::new(&mut reader);
    decoder
        .decode_headers()
        .expect("Failed to decode JPEG headers");
    // Decode first image (Primary)
    let primary_image = decoder.decode().expect("Failed to decode primary image");
    let primary_metadata = decoder.info().expect("No metadata found");

    let parsed_mpf =
        MpfInfo::from_bytes(&decoder.info().unwrap().multi_picture_information.unwrap()).unwrap();
    println!("{:?}", parsed_mpf);
    println!("{:#?}", parsed_mpf.version.unwrap().test_version());

    if let Some(xmp_data) = decoder.xmp() {
        println!("Found xmp data");
        if let Ok(xmp_string) = String::from_utf8(xmp_data.to_vec()) {
            println!("Found xmp data: {}", xmp_string);
        }
    }

    let image_icc = if let Some(icc) = decoder.icc_profile() {
        match ColorProfile::new_from_slice(&icc) {
            Ok(a0) => Some(a0),
            Err(_) => None,
        }
    } else {
        None
    };

    let file = File::open(file_path).expect("Failed to open file");
    let mut reader2 = BufReader::new(file);
    let stream_pos = reader.stream_position().unwrap() ;
    reader2.seek(SeekFrom::Start(stream_pos)).unwrap();
    let mut dst_vec = Vec::new();
    reader2.read_to_end(&mut dst_vec).unwrap();

    // Read the second image from JPEG file

    let mut gm_reader = BufReader::new(File::open(file_path).expect("Failed to open file"));
    // let chunk = find_iso_chunks(&mut gm_reader).unwrap();

    let mut decoder = JpegDecoder::new(Cursor::new(dst_vec.to_vec()));
    decoder
        .decode_headers()
        .expect("Failed to decode JPEG headers");

    if let Some(xmp_data) = decoder.xmp() {
        println!("Found xmp data");
        if let Ok(xmp_string) = String::from_utf8(xmp_data.to_vec()) {
            println!("Found xmp data: {}", xmp_string);
        }
    }

    let gainmap_info = &decoder.info().unwrap().gain_map_info[0].data;
    let gain_map = IsoGainMap::from_bytes(&gainmap_info).unwrap();

    let gain_map_icc = if let Some(icc) = decoder.icc_profile() {
        match ColorProfile::new_from_slice(&icc) {
            Ok(a0) => Some(a0),
            Err(_) => None,
        }
    } else {
        None
    };

    if let Some(xmp_data) = decoder.xmp() {
        println!("Found xmp data");
        if let Ok(xmp_string) = String::from_utf8(xmp_data.to_vec()) {
            println!("Found xmp data: {}", xmp_string);
        }
    }

    let gain_map_image = decoder.decode().unwrap();

    AssociatedImages {
        image: primary_image,
        gain_map: gain_map_image,
        width: primary_metadata.width as usize,
        height: primary_metadata.height as usize,
        icc_profile: image_icc,
        gain_map_icc_profile: gain_map_icc,
        metadata: gain_map,
    }
}

fn main() {
    let associated = extract_images("./assets/uhdr_01.jpg");
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

    let display_boost = 1.7f32;
    let gainmap_weight = make_gainmap_weight(gainmap, display_boost);
    println!("weight {}", gainmap_weight);

    let instant = std::time::Instant::now();

    let source_image =
        GainImage::<u8, 3>::borrow(&associated.image, associated.width, associated.height);
    let gain_image =
        GainImage::<u8, 3>::borrow(&associated.gain_map, associated.width, associated.height);
    let mut dst_image = GainImageMut::<u16, 3>::alloc(associated.width, associated.height);

    let dest_profile = ColorProfile::new_srgb();

    apply_gain_map_rgb10(
        &source_image.expand_to_u16(10).to_immutable_ref(),
        &mut dst_image,
        &associated.icc_profile,
        &gain_image.expand_to_u16(10).to_immutable_ref(),
        &associated.gain_map_icc_profile,
        &dest_profile,
        gainmap,
        gainmap_weight,
    )
    .unwrap();

    println!("Time {:?}", instant.elapsed());

    image::save_buffer(
        "processed_alu10_d65.png",
        &dst_image
            .data
            .borrow()
            .iter()
            .map(|&x| (x << 6) | (x >> 4))
            .flat_map(|x| [x.to_ne_bytes()[0], x.to_ne_bytes()[1]])
            .collect::<Vec<u8>>(),
        associated.width as u32,
        associated.height as u32,
        image::ExtendedColorType::Rgb16,
    )
    .unwrap();
}
