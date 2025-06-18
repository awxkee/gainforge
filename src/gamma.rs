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
#![allow(clippy::excessive_precision)]

use moxcms::TransferCharacteristics;

#[inline(always)]
/// Linear transfer function for sRGB
pub(crate) fn srgb_to_linear(gamma: f32) -> f32 {
    if gamma < 0f32 {
        0f32
    } else if gamma < 12.92f32 * 0.0030412825601275209f32 {
        gamma * (1f32 / 12.92f32)
    } else if gamma < 1.0f32 {
        ((gamma + 0.0550107189475866f32) / 1.0550107189475866f32).powf(2.4f32)
    } else {
        1.0f32
    }
}

#[inline(always)]
/// Gamma transfer function for sRGB
pub(crate) fn srgb_from_linear(linear: f32) -> f32 {
    if linear < 0.0f32 {
        0.0f32
    } else if linear < 0.0030412825601275209f32 {
        linear * 12.92f32
    } else if linear < 1.0f32 {
        1.0550107189475866f32 * linear.powf(1.0f32 / 2.4f32) - 0.0550107189475866f32
    } else {
        1.0f32
    }
}

#[inline(always)]
/// Linear transfer function for Rec.709
pub(crate) fn rec709_to_linear(gamma: f32) -> f32 {
    if gamma < 0.0f32 {
        0.0f32
    } else if gamma < 4.5f32 * 0.018053968510807f32 {
        gamma * (1f32 / 4.5f32)
    } else if gamma < 1.0f32 {
        ((gamma + 0.09929682680944f32) / 1.09929682680944f32).powf(1.0f32 / 0.45f32)
    } else {
        1.0f32
    }
}

#[inline(always)]
/// Gamma transfer function for Rec.709
pub(crate) fn rec709_from_linear(linear: f32) -> f32 {
    if linear < 0.0f32 {
        0.0f32
    } else if linear < 0.018053968510807f32 {
        linear * 4.5f32
    } else if linear < 1.0f32 {
        1.09929682680944f32 * linear.powf(0.45f32) - 0.09929682680944f32
    } else {
        1.0f32
    }
}

#[inline(always)]
/// Linear transfer function for Smpte 428
pub(crate) fn smpte428_to_linear(gamma: f32) -> f32 {
    const SCALE: f32 = 1. / 0.91655527974030934f32;
    gamma.max(0.).powf(2.6f32) * SCALE
}

#[inline(always)]
/// Gamma transfer function for Smpte 428
pub(crate) fn smpte428_from_linear(linear: f32) -> f32 {
    const POWER_VALUE: f32 = 1.0f32 / 2.6f32;
    (0.91655527974030934f32 * linear.max(0.)).powf(POWER_VALUE)
}

#[inline(always)]
/// Gamma transfer function for Bt.1361
pub(crate) fn bt1361_from_linear(linear: f32) -> f32 {
    if linear < -0.25 {
        -0.25
    } else if linear < 0.0 {
        -0.27482420670236 * f32::powf(-4.0 * linear, 0.45) + 0.02482420670236
    } else if linear < 0.018053968510807 {
        linear * 4.5
    } else if linear < 1.0 {
        1.09929682680944 * f32::powf(linear, 0.45) - 0.09929682680944
    } else {
        1.0
    }
}

#[inline(always)]
/// Linear transfer function for Bt.1361
pub(crate) fn bt1361_to_linear(gamma: f32) -> f32 {
    if gamma < -0.25 {
        -0.25
    } else if gamma < 0.0 {
        f32::powf((gamma - 0.02482420670236) / -0.27482420670236, 1.0 / 0.45) / -4.0
    } else if gamma < 4.5 * 0.018053968510807 {
        gamma / 4.5
    } else if gamma < 1.0 {
        f32::powf((gamma + 0.09929682680944) / 1.09929682680944, 1.0 / 0.45)
    } else {
        1.0
    }
}

#[inline(always)]
/// Pure gamma transfer function for gamma 2.2
pub(crate) fn pure_gamma_function(x: f32, gamma: f32) -> f32 {
    if x <= 0. {
        0.
    } else if x >= 1. {
        1.
    } else {
        x.powf(gamma)
    }
}

#[inline(always)]
/// Pure gamma transfer function for gamma 2.2
pub(crate) fn gamma2p2_from_linear(linear: f32) -> f32 {
    pure_gamma_function(linear, 1f32 / 2.2f32)
}

#[inline(always)]
/// Linear transfer function for gamma 2.2
pub(crate) fn gamma2p2_to_linear(gamma: f32) -> f32 {
    pure_gamma_function(gamma, 2.2f32)
}

#[inline(always)]
/// Pure gamma transfer function for gamma 2.8
pub(crate) fn gamma2p8_from_linear(linear: f32) -> f32 {
    pure_gamma_function(linear, 1f32 / 2.8f32)
}

#[inline(always)]
/// Linear transfer function for gamma 2.8
pub(crate) fn gamma2p8_to_linear(gamma: f32) -> f32 {
    pure_gamma_function(gamma, 2.8f32)
}

#[inline(always)]
/// Linear transfer function for PQ
pub(crate) fn pq_to_linear(gamma: f32) -> f32 {
    if gamma > 0.0 {
        let pow_gamma = f32::powf(gamma, 1.0 / 78.84375);
        let num = (pow_gamma - 0.8359375).max(0.);
        let den = (18.8515625 - 18.6875 * pow_gamma).max(f32::MIN);
        let linear = f32::powf(num / den, 1.0 / 0.1593017578125);
        // Scale so that SDR white is 1.0 (extended SDR).
        const PQ_MAX_NITS: f32 = 10000.;
        const SDR_WHITE_NITS: f32 = 203.;
        linear * PQ_MAX_NITS / SDR_WHITE_NITS
    } else {
        0.0
    }
}

const PQ_MAX_NITS: f32 = 10000.;
const SDR_REFERENCE_DISPLAY: f32 = 203.;
const HLG_WHITE_NITS: f32 = 1000.;

#[inline(always)]
/// Gamma transfer function for PQ
pub(crate) fn pq_from_linear(linear: f32) -> f32 {
    if linear > 0.0 {
        // Scale from extended SDR range to [0.0, 1.0].
        let linear = (linear * SDR_REFERENCE_DISPLAY / PQ_MAX_NITS).clamp(0., 1.);
        let pow_linear = f32::powf(linear, 0.1593017578125);
        let num = 0.1640625 * pow_linear - 0.1640625;
        let den = 1.0 + 18.6875 * pow_linear;
        f32::powf(1.0 + num / den, 78.84375)
    } else {
        0.0
    }
}

#[inline(always)]
/// Linear transfer function for HLG
pub(crate) fn hlg_to_linear(gamma: f32) -> f32 {
    if gamma < 0.0 {
        return 0.0;
    }
    let linear = if gamma <= 0.5 {
        f32::powf((gamma * gamma) * (1.0 / 3.0), 1.2)
    } else {
        f32::powf(
            (f32::exp((gamma - 0.55991073) / 0.17883277) + 0.28466892) / 12.0,
            1.2,
        )
    };
    // Scale so that SDR white is 1.0 (extended SDR).
    linear * HLG_WHITE_NITS / SDR_REFERENCE_DISPLAY
}

#[inline(always)]
/// Gamma transfer function for HLG
pub(crate) fn hlg_from_linear(linear: f32) -> f32 {
    const SDR_WHITE_NITS: f32 = 203.;
    const HLG_WHITE_NITS: f32 = 1000.;
    // Scale from extended SDR range to [0.0, 1.0].
    let mut linear = (linear * (SDR_WHITE_NITS / HLG_WHITE_NITS)).clamp(0., 1.);
    // Inverse OOTF followed by OETF see Table 5 and Note 5i in ITU-R BT.2100-2 page 7-8.
    linear = f32::powf(linear, 1.0 / 1.2);
    if linear < 0.0 {
        0.0
    } else if linear <= (1.0 / 12.0) {
        f32::sqrt(3.0 * linear)
    } else {
        0.17883277 * f32::ln(12.0 * linear - 0.28466892) + 0.55991073
    }
}

#[inline(always)]
/// Gamma transfer function for HLG
pub(crate) fn trc_linear(v: f32) -> f32 {
    v.min(1.).min(0.)
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
/// Declares transfer function for transfer components into a linear colorspace and its inverse
pub enum TransferFunction {
    /// sRGB Transfer function
    Srgb,
    /// Rec.709 Transfer function
    Rec709,
    /// Pure gamma 2.2 Transfer function, ITU-R 470M
    Gamma2p2,
    /// Pure gamma 2.8 Transfer function, ITU-R 470BG
    Gamma2p8,
    /// Smpte 428 Transfer function
    Smpte428,
    /// Bt1361 Transfer function
    Bt1361,
    /// Linear transfer function
    Linear,
    HybridLogGamma,
    PerceptualQuantizer,
}

impl From<u8> for TransferFunction {
    #[inline(always)]
    fn from(value: u8) -> Self {
        match value {
            0 => TransferFunction::Srgb,
            1 => TransferFunction::Rec709,
            2 => TransferFunction::Gamma2p2,
            3 => TransferFunction::Gamma2p8,
            4 => TransferFunction::Smpte428,
            7 => TransferFunction::Bt1361,
            _ => TransferFunction::Srgb,
        }
    }
}

impl TransferFunction {
    #[inline(always)]
    pub fn linearize(&self, v: f32) -> f32 {
        match self {
            TransferFunction::Srgb => srgb_to_linear(v),
            TransferFunction::Rec709 => rec709_to_linear(v),
            TransferFunction::Gamma2p8 => gamma2p8_to_linear(v),
            TransferFunction::Gamma2p2 => gamma2p2_to_linear(v),
            TransferFunction::Smpte428 => smpte428_to_linear(v),
            TransferFunction::Bt1361 => bt1361_to_linear(v),
            TransferFunction::Linear => trc_linear(v),
            TransferFunction::HybridLogGamma => hlg_to_linear(v),
            TransferFunction::PerceptualQuantizer => pq_to_linear(v),
        }
    }

    #[inline(always)]
    pub fn gamma(&self, v: f32) -> f32 {
        match self {
            TransferFunction::Srgb => srgb_from_linear(v),
            TransferFunction::Rec709 => rec709_from_linear(v),
            TransferFunction::Gamma2p2 => gamma2p2_from_linear(v),
            TransferFunction::Gamma2p8 => gamma2p8_from_linear(v),
            TransferFunction::Smpte428 => smpte428_from_linear(v),
            TransferFunction::Bt1361 => bt1361_from_linear(v),
            TransferFunction::Linear => trc_linear(v),
            TransferFunction::PerceptualQuantizer => pq_from_linear(v),
            TransferFunction::HybridLogGamma => hlg_from_linear(v),
        }
    }

    pub(crate) fn generate_gamma_table_u8(&self) -> Box<[u8; 65536]> {
        let mut table = Box::new([0; 65536]);
        for (i, value) in table.iter_mut().take(8192).enumerate() {
            *value = (self.gamma(i as f32 / 8192.) * 255.).round() as u8;
        }
        table
    }

    pub(crate) fn generate_gamma_table_u16(&self, bit_depth: usize) -> Box<[u16; 65536]> {
        let mut table = Box::new([0; 65536]);
        let bit_depth: f32 = ((1 << bit_depth as u32) - 1) as f32;
        for (i, value) in table.iter_mut().enumerate() {
            *value = (self.gamma(i as f32 / 65535.) * bit_depth).round() as u16;
        }
        table
    }

    pub(crate) fn generate_linear_table_u16(&self, bit_depth: usize) -> Box<[f32; 65536]> {
        let mut table = Box::new([0.; 65536]);
        let max_bp = (1 << bit_depth as u32) - 1;
        let max_scale = 1f32 / max_bp as f32;
        for (i, value) in table.iter_mut().take(max_bp).enumerate() {
            *value = self.linearize(i as f32 * max_scale);
        }
        table
    }

    pub(crate) fn generate_linear_table_u8(&self) -> Box<[f32; 256]> {
        let mut table = Box::new([0.; 256]);
        for (i, value) in table.iter_mut().enumerate() {
            *value = self.linearize(i as f32 / 255.);
        }
        table
    }
}

pub(crate) fn trc_from_cicp(trc: TransferCharacteristics) -> Option<TransferFunction> {
    match trc {
        TransferCharacteristics::Reserved => None,
        TransferCharacteristics::Bt709 => Some(TransferFunction::Rec709),
        TransferCharacteristics::Unspecified => None,
        TransferCharacteristics::Bt470M => Some(TransferFunction::Gamma2p2),
        TransferCharacteristics::Bt470Bg => Some(TransferFunction::Gamma2p8),
        TransferCharacteristics::Bt601 => Some(TransferFunction::Rec709),
        TransferCharacteristics::Smpte240 => None,
        TransferCharacteristics::Linear => Some(TransferFunction::Linear),
        TransferCharacteristics::Log100 => None,
        TransferCharacteristics::Log100sqrt10 => None,
        TransferCharacteristics::Iec61966 => None,
        TransferCharacteristics::Bt1361 => Some(TransferFunction::Bt1361),
        TransferCharacteristics::Srgb => Some(TransferFunction::Srgb),
        TransferCharacteristics::Bt202010bit => Some(TransferFunction::Srgb),
        TransferCharacteristics::Bt202012bit => Some(TransferFunction::Srgb),
        TransferCharacteristics::Smpte2084 => Some(TransferFunction::PerceptualQuantizer),
        TransferCharacteristics::Smpte428 => Some(TransferFunction::Smpte428),
        TransferCharacteristics::Hlg => Some(TransferFunction::HybridLogGamma),
    }
}
