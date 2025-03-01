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
#![allow(clippy::manual_clamp, clippy::excessive_precision)]
mod apply_gain_map;
mod cms;
mod err;
mod gamma;
mod iso_gain_map;
mod mappers;
mod mlaf;
mod tonemapper;

pub use apply_gain_map::apply_gain_map_rgb;
pub use cms::GamutColorSpace;
pub use err::ForgeError;
pub use gamma::{HdrTransferFunction, TransferFunction};
pub use iso_gain_map::{
    make_gainmap_weight, IsoGainMap, MpfDataType, MpfEndianness, MpfEntry, MpfImageType, MpfInfo,
    MpfNumberOfImages, MpfTag,
};
pub use mappers::ToneMappingMethod;
pub use tonemapper::{
    create_tone_mapper_rgb, create_tone_mapper_rgb10, create_tone_mapper_rgb12,
    create_tone_mapper_rgb14, create_tone_mapper_rgb16, create_tone_mapper_rgba,
    create_tone_mapper_rgba10, create_tone_mapper_rgba12, create_tone_mapper_rgba14,
    create_tone_mapper_rgba16, GainHDRMetadata, SyncToneMapper16Bit, SyncToneMapper8Bit,
    ToneMapper,
};
