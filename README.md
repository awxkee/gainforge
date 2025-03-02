# HDR tone mapping library in Rust

Library helps perform tone mapping from HDR to SDR

## Example

```rust
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
        ToneMappingMethod::Rec2408,
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
        "processed.jpg",
        &dst,
        dims.0,
        dims.1,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
```

# How to handle UHDR

Some patches on zune-image still in processing, resolving package
from zune-image might be required

```rust
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
    // Atm supported only from marker
    let parsed_mpf =
        MpfInfo::from_bytes(&decoder.info().unwrap().multi_picture_information.unwrap()).unwrap();

    let cv = Vec::new();
    let primary_xmp = decoder.xmp().unwrap_or(&cv);

    // UHDR directory info if needed
    let uhdr_directory = UhdrDirectoryContainer::from_xml(primary_xmp);

    let image_icc = decoder
        .icc_profile()
        .and_then(|icc| ColorProfile::new_from_slice(&icc).ok());

    let file = File::open(file_path).expect("Failed to open file");
    let mut reader2 = BufReader::new(file);
    // Zune have bug where some streams consumed in full, some or not, it might
    // be needed to adjust stream position using MPF or any other approach
    // At the moment some images works when +2 is added, some images are not
    let stream_pos = reader.stream_position().unwrap();
    reader2.seek(SeekFrom::Start(stream_pos)).unwrap();
    let mut dst_vec = Vec::new();
    reader2.read_to_end(&mut dst_vec).unwrap();

    // Read the second image from JPEG file

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

// Load required associated images
let associated = extract_images("./assets/uhdr_01.jpg");

let gainmap = associated.metadata.to_gain_map();

// Get maximum display boost from screen information
let display_boost = 1.3f32;
let gainmap_weight = make_gainmap_weight(gainmap, display_boost);

let source_image =
GainImage::<u8, 3>::borrow(&associated.image, associated.width, associated.height);
let gain_image =
GainImage::<u8, 3>::borrow(&associated.gain_map, associated.width, associated.height);
let mut dst_image = GainImageMut::<u8, 3>::alloc(associated.width, associated.height);

// Screen colorspace
let dest_profile = ColorProfile::new_srgb();

// And finally apply gain map
apply_gain_map_rgb(
    &source_image,
    &associated.icc_profile,
    &mut dst_image,
    &dest_profile,
    &gain_image,
    &associated.gain_map_icc_profile,
    gainmap,
    gainmap_weight,
)
.unwrap();
```

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
