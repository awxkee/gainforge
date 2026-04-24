#![no_main]

use arbitrary::Arbitrary;
use gainforge::{
    AgxLook, FilmicSplineParameters, GainHdrMetadata, GamutClipping, MappingColorSpace,
    RgbToneMapperParameters, ToneMappingMethod, create_tone_mapper_rgb,
};
use libfuzzer_sys::fuzz_target;
use moxcms::ColorProfile;

#[derive(Arbitrary, Debug)]
struct Target {
    w: u8,
    h: u8,
    v: u8,
    tone_mapping: ArbitraryToneMapping,
}

#[derive(Arbitrary, Debug)]
enum ArbitraryToneMapping {
    TunedReinhard(ArbitraryGainHdrMetadata),
    Itu2408(ArbitraryGainHdrMetadata),
    Filmic,
    Aces,
    Reinhard,
    ExtendedReinhard,
    ReinhardJodie,
    Clamp,
    FilmicSpline(ArbitraryFilmicSplineParameters),
    Agx,
}

#[derive(Arbitrary, Debug)]
struct ArbitraryGainHdrMetadata {
    content_max_brightness: f32,
    display_max_brightness: f32,
}

#[derive(Arbitrary, Debug)]
struct ArbitraryFilmicSplineParameters {
    output_power: f32,
    latitude: f32,           // $MIN: 0.01 $MAX: 99 $DEFAULT: 33.0
    white_point_source: f32, // $MIN: 0.1 $MAX: 16 $DEFAULT: 4.0 $DESCRIPTION: "white relative exposure"
    black_point_source: f32, // $MIN: -16 $MAX: -0.1 $DEFAULT: -8.0 $DESCRIPTION: "black relative exposure"
    contrast: f32,           // $MIN: 0 $MAX: 5 $DEFAULT: 1.18
    black_point_target: f32, // $MIN: 0.000 $MAX: 20.000 $DEFAULT: 0.01517634 $DESCRIPTION: "target black luminance"
    grey_point_target: f32,  // $MIN: 1 $MAX: 50 $DEFAULT: 18.45 $DESCRIPTION: "target middle gray"
    white_point_target: f32, // $MIN: 0 $MAX: 1600 $DEFAULT: 100 $DESCRIPTION: "target white luminance"
    balance: f32, // $MIN: -50 $MAX: 50 $DEFAULT: 0.0 $DESCRIPTION: "shadows \342\206\224 highlights balance"
    saturation: f32, // $MIN: -200 $MAX: 200 $DEFAULT: 0 $DESCRIPTION: "extreme luminance saturation"
}

impl From<ArbitraryToneMapping> for ToneMappingMethod {
    fn from(a: ArbitraryToneMapping) -> Self {
        match a {
            ArbitraryToneMapping::TunedReinhard(m) => {
                ToneMappingMethod::TunedReinhard(GainHdrMetadata {
                    display_max_brightness: m.display_max_brightness,
                    content_max_brightness: m.content_max_brightness,
                })
            }
            ArbitraryToneMapping::Itu2408(m) => ToneMappingMethod::Itu2408(GainHdrMetadata {
                display_max_brightness: m.display_max_brightness,
                content_max_brightness: m.content_max_brightness,
            }),
            ArbitraryToneMapping::Filmic => ToneMappingMethod::Filmic,
            ArbitraryToneMapping::Aces => ToneMappingMethod::Aces,
            ArbitraryToneMapping::Reinhard => ToneMappingMethod::Reinhard,
            ArbitraryToneMapping::ExtendedReinhard => {
                ToneMappingMethod::ExtendedReinhard { max_luma: 1. }
            }
            ArbitraryToneMapping::ReinhardJodie => ToneMappingMethod::ReinhardJodie,
            ArbitraryToneMapping::Clamp => ToneMappingMethod::Clamp,
            ArbitraryToneMapping::FilmicSpline(p) => {
                ToneMappingMethod::FilmicSpline(FilmicSplineParameters {
                    black_point_source: p.black_point_source,
                    balance: p.balance,
                    black_point_target: p.black_point_target,
                    contrast: p.contrast,
                    grey_point_target: p.grey_point_target,
                    white_point_source: p.white_point_source,
                    white_point_target: p.white_point_target,
                    output_power: p.output_power,
                    latitude: p.latitude,
                    saturation: p.saturation,
                })
            }
            ArbitraryToneMapping::Agx => ToneMappingMethod::Agx(AgxLook::Agx),
        }
    }
}

fuzz_target!(|data: Target| {
    let tone_mapping: ToneMappingMethod = data.tone_mapping.into();
    let tone_mapper = create_tone_mapper_rgb(
        &ColorProfile::new_bt2020_pq(),
        &ColorProfile::new_srgb(),
        tone_mapping,
        MappingColorSpace::Rgb(RgbToneMapperParameters {
            gamut_clipping: GamutClipping::NoClip,
            exposure: 1.0,
        }),
    )
    .unwrap();
    let src = vec![data.v; data.w as usize * data.h as usize * 3];
    let mut dst = vec![0u8; data.w as usize * data.h as usize * 3];
    tone_mapper.tonemap_lane(&src, &mut dst).unwrap();
});
