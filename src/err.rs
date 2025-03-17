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
use std::error::Error;
use std::fmt::Display;

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
/// Shows size mismatching
pub struct MismatchedSize {
    pub expected: usize,
    pub received: usize,
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum ForgeError {
    LaneSizeMismatch,
    LaneMultipleOfChannels,
    InvalidIcc,
    InvalidTrcCurve,
    InvalidCicp,
    CurveLutIsTooLarge,
    ParametricCurveZeroDivision,
    InvalidRenderingIntent,
    DivisionByZero,
    UnsupportedColorPrimaries(u8),
    UnsupportedTrc(u8),
    InvalidGainMapConfiguration,
    ImageSizeMismatch,
    ZeroBaseSize,
    MinimumSliceSizeMismatch(MismatchedSize),
    MinimumStrideSizeMismatch(MismatchedSize),
    UnknownError,
}

impl Display for ForgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForgeError::LaneSizeMismatch => write!(f, "Lanes length must match"),
            ForgeError::LaneMultipleOfChannels => {
                write!(f, "Lane length must not be multiple of channel count")
            }
            ForgeError::InvalidIcc => f.write_str("Invalid ICC profile"),
            ForgeError::InvalidCicp => f.write_str("Invalid CICP in ICC profile"),
            ForgeError::InvalidTrcCurve => f.write_str("Invalid TRC curve"),
            ForgeError::CurveLutIsTooLarge => f.write_str("Curve Lut is too large"),
            ForgeError::ParametricCurveZeroDivision => {
                f.write_str("Parametric Curve definition causes division by zero")
            }
            ForgeError::InvalidRenderingIntent => f.write_str("Invalid rendering intent"),
            ForgeError::DivisionByZero => f.write_str("Division by zero"),
            ForgeError::UnsupportedColorPrimaries(value) => {
                f.write_fmt(format_args!("Unsupported color primaries, {}", value))
            }
            ForgeError::UnsupportedTrc(value) => write!(f, "Unsupported TRC {}", value),
            ForgeError::InvalidGainMapConfiguration => {
                f.write_str("Invalid Gain map configuration")
            }
            ForgeError::ImageSizeMismatch => f.write_str("Image size does not match"),
            ForgeError::ZeroBaseSize => f.write_str("Image size must not be zero"),
            ForgeError::MinimumSliceSizeMismatch(size) => f.write_fmt(format_args!(
                "Minimum image slice size mismatch: expected={}, received={}",
                size.expected, size.received
            )),
            ForgeError::MinimumStrideSizeMismatch(size) => f.write_fmt(format_args!(
                "Minimum stride must have size at least {} but it is {}",
                size.expected, size.received
            )),
            ForgeError::UnknownError => f.write_str("Unknown error"),
        }
    }
}

impl Error for ForgeError {}
