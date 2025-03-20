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
    create_tone_mapper_rgb, create_tone_mapper_rgb16, BufferStore, CommonToneMapperParameters,
    FilmicSplineParameters, GainHdrMetadata, GainImage, GainImageMut, GamutClipping, IsoGainMap,
    JzazbzToneMapperParameters, MappingColorSpace, MpfInfo, RgbToneMapperParameters,
    ToneMappingMethod, TransferFunction, UhdrDirectoryContainer,
};
use moxcms::ColorProfile;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::time::Instant;
use zune_jpeg::JpegDecoder;

pub struct GainMapAssociationGroup {
    pub image: Vec<u8>,
    pub gain_map: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub icc_profile: Option<ColorProfile>,
    pub gain_map_icc_profile: Option<ColorProfile>,
    pub metadata: IsoGainMap,
}

fn extract_images(file_path: &str) -> GainMapAssociationGroup {
    let file = File::open(file_path).expect("Failed to open file");

    let mut reader = BufReader::new(file);

    let mut decoder = JpegDecoder::new(&mut reader);
    decoder
        .decode_headers()
        .expect("Failed to decode JPEG headers");
    // Decode first image (Primary)
    let primary_image = decoder.decode().expect("Failed to decode primary image");
    let primary_metadata = decoder.info().expect("No metadata found");

    // Multi picture format information, if you want to do something with it
    let parsed_mpf =
        MpfInfo::from_bytes(&decoder.info().unwrap().multi_picture_information.unwrap()).unwrap();

    let cv = Vec::new();
    let primary_xmp = decoder.xmp().unwrap_or(&cv);

    // UHDR directory info if needed
    let uhdr_directory = UhdrDirectoryContainer::from_xml(primary_xmp);

    let image_icc = decoder
        .icc_profile()
        .and_then(|icc| ColorProfile::new_from_slice(&icc).ok());

    // Read the second image from JPEG file

    // This might be done using MPF
    // or by last read stream position.

    let file = File::open(file_path).expect("Failed to open file");
    let mut reader2 = BufReader::new(file);
    // Zune have bug where some streams consumed in full, some or not, it might
    // be needed to adjust stream position using MPF or any other approach
    // At the moment some images works when +2 is added, some images are not.
    // Was fixed in the latest commits of zune-jpeg.
    let stream_pos = reader.stream_position().unwrap();
    reader2.seek(SeekFrom::Start(stream_pos)).unwrap();
    let mut dst_vec = Vec::new();
    reader2.read_to_end(&mut dst_vec).unwrap();

    let mut decoder = JpegDecoder::new(Cursor::new(dst_vec.to_vec()));

    decoder
        .decode_headers()
        .expect("Failed to decode JPEG headers");

    // Gain map might be stored either in XMP and APP2 iso chunk
    let xmp_data = decoder
        .xmp()
        .map(|x| x.to_vec())
        .or(Some(Vec::new()))
        .unwrap();

    // New zune-jpeg is required
    let gainmap_info = if decoder.info().unwrap().gain_map_info.len() > 0 {
        decoder.info().unwrap().gain_map_info[0].data.to_vec()
    } else {
        Vec::new()
    };
    let gain_map = IsoGainMap::from_metadata(&gainmap_info)
        .or_else(|_| IsoGainMap::from_xml_data(&xmp_data))
        .unwrap();

    let gain_map_icc = decoder
        .icc_profile()
        .and_then(|icc| ColorProfile::new_from_slice(&icc).ok());

    let mut gain_map_image = decoder.decode().unwrap();

    let gain_map_image_info = decoder.info().unwrap();

    // Gain map might have 3 components, or 1.
    // Might be in full size or 1/4.
    // this implementation always returns full image in 3 components.
    if gain_map_image_info.components == 1 {
        gain_map_image = gain_map_image.iter().flat_map(|&x| [x, x, x]).collect();
    }

    if gain_map_image_info.width != primary_metadata.width
        || gain_map_image_info.height != primary_metadata.height
    {
        let source_image = pic_scale::ImageStore::<u8, 3>::borrow(
            &gain_map_image,
            gain_map_image_info.width as usize,
            gain_map_image_info.height as usize,
        )
        .unwrap();
        let mut scaler = pic_scale::Scaler::new(pic_scale::ResamplingFunction::Lanczos3);
        scaler.set_workload_strategy(pic_scale::WorkloadStrategy::PreferQuality);
        let mut dst_image = pic_scale::ImageStoreMut::<u8, 3>::alloc(
            primary_metadata.width as usize,
            primary_metadata.height as usize,
        );
        use pic_scale::Scaling;
        scaler.resize_rgb(&source_image, &mut dst_image).unwrap();
        gain_map_image = dst_image.buffer.borrow().to_vec();
    }

    GainMapAssociationGroup {
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
    let img = image::ImageReader::open("./assets/hdr.avif")
        .unwrap()
        .decode()
        .unwrap();
    let rgb = img.to_rgb8();

    let tone_mapper = create_tone_mapper_rgb(
        &ColorProfile::new_bt2020_pq(),
        &ColorProfile::new_srgb(),
        // ToneMappingMethod::FilmicSpline(FilmicSplineParameters {
        //     saturation: 0f32,
        //     white_point_source: 8f32,
        //     black_point_source: -8f32,
        //     black_point_target: 0.01f32,
        //     ..Default::default()
        // }),
        // ToneMappingMethod::Reinhard,
        // ToneMappingMethod::Rec2408(GainHdrMetadata::new(2000f32, 250f32)),
        ToneMappingMethod::Filmic,
        MappingColorSpace::Jzazbz(JzazbzToneMapperParameters {
            exposure: 1.0f32,
            content_brightness: 2000f32,
        }),
        // MappingColorSpace::Jzazbz(JzazbzToneMapperParameters {
        //     content_brightness: 2000f32,
        //     exposure: 1f32,
        // }),
    )
    .unwrap();
    let dims = rgb.dimensions();
    let mut dst = vec![0u8; rgb.len()];
    let work_time = Instant::now();
    for (src, dst) in rgb
        .chunks_exact(rgb.dimensions().0 as usize * 3)
        .zip(dst.chunks_exact_mut(rgb.dimensions().0 as usize * 3))
    {
        tone_mapper.tonemap_lane(src, dst).unwrap();
    }
    println!("Exec time: {:?}", work_time.elapsed());
    // Load required associated images
    // let associated = extract_images("./assets/02.jpg");
    //
    // let gainmap = associated.metadata.to_gain_map();
    //
    // // Get maximum display boost from screen information
    // let display_boost = 1.3f32;
    // let gainmap_weight = make_gainmap_weight(gainmap, display_boost);
    //
    // let source_image =
    //     GainImage::<u8, 3>::borrow(&associated.image, associated.width, associated.height);
    // let gain_image =
    //     GainImage::<u8, 3>::borrow(&associated.gain_map, associated.width, associated.height);
    // let mut dst_image = GainImageMut::<u8, 3>::alloc(associated.width, associated.height);
    //
    // // Screen colorspace
    // let dest_profile = ColorProfile::new_srgb();
    //
    // // And finally apply gain map
    // apply_gain_map_rgb(
    //     &source_image,
    //     &associated.icc_profile,
    //     &mut dst_image,
    //     &dest_profile,
    //     &gain_image,
    //     &associated.gain_map_icc_profile,
    //     gainmap,
    //     gainmap_weight,
    // )
    // .unwrap();

    // let compressed = dst.iter().map(|&x| (x >> 8) as u8).collect::<Vec<_>>();

    image::save_buffer(
        "clamp_compress.jpg",
        &dst,
        img.width(),
        img.height(),
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
}
