# HDR tone mapping library in Rust

Library helps perform tonemapping from HDR to SDR

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

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
