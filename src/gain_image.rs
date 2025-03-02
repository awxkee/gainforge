/*
 * // Copyright (c) Radzivon Bartoshyk 3/2025. All rights reserved.
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
use crate::err::MismatchedSize;
use crate::ForgeError;
use std::fmt::Debug;

#[derive(Debug)]
pub enum BufferStore<'a, T: Copy + Debug> {
    Borrowed(&'a mut [T]),
    Owned(Vec<T>),
}

impl<T: Copy + Debug> BufferStore<'_, T> {
    #[allow(clippy::should_implement_trait)]
    pub fn borrow(&self) -> &[T] {
        match self {
            Self::Borrowed(p_ref) => p_ref,
            Self::Owned(vec) => vec,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn borrow_mut(&mut self) -> &mut [T] {
        match self {
            Self::Borrowed(p_ref) => p_ref,
            Self::Owned(vec) => vec,
        }
    }
}

/// Immutable image store
pub struct GainImage<'a, T: Clone + Copy + Default + Debug, const N: usize> {
    pub data: std::borrow::Cow<'a, [T]>,
    pub width: usize,
    pub height: usize,
    /// Image stride, items per row, might be 0
    pub stride: usize,
}

/// Mutable image store
pub struct GainImageMut<'a, T: Clone + Copy + Default + Debug, const N: usize> {
    pub data: BufferStore<'a, T>,
    pub width: usize,
    pub height: usize,
    /// Image stride, items per row, might be 0
    pub stride: usize,
}

impl<'a, T: Clone + Copy + Default + Debug, const N: usize> GainImage<'a, T, N> {
    /// Allocates default image layout for given N channels count
    pub fn alloc(width: usize, height: usize) -> Self {
        Self {
            data: std::borrow::Cow::Owned(vec![T::default(); width * height * N]),
            width,
            height,
            stride: width * N,
        }
    }

    /// Borrows existing data
    /// Stride will be default `width * N`
    pub fn borrow(arr: &'a [T], width: usize, height: usize) -> Self {
        Self {
            data: std::borrow::Cow::Borrowed(arr),
            width,
            height,
            stride: width * N,
        }
    }

    /// Checks if it is matches the size of the other image
    #[inline]
    pub fn size_matches(&self, other: &GainImage<'_, T, N>) -> Result<(), ForgeError> {
        if self.width == other.width && self.height == other.height {
            return Ok(());
        }
        Err(ForgeError::ImageSizeMismatch)
    }

    /// Checks if it is matches the size of the other image
    #[inline]
    pub fn size_matches_arb<const G: usize>(&self, other: &GainImage<'_, T, G>) -> Result<(), ForgeError> {
        if self.width == other.width && self.height == other.height {
            return Ok(());
        }
        Err(ForgeError::ImageSizeMismatch)
    }

    /// Checks if it is matches the size of the other image
    #[inline]
    pub fn size_matches_mut(&self, other: &GainImageMut<'_, T, N>) -> Result<(), ForgeError> {
        if self.width == other.width && self.height == other.height {
            return Ok(());
        }
        Err(ForgeError::ImageSizeMismatch)
    }

    /// Checks if layout matches necessary requirements by using external channels count
    #[inline]
    pub fn check_layout_channels(&self, cn: usize) -> Result<(), ForgeError> {
        if self.width == 0 || self.height == 0 {
            return Err(ForgeError::ZeroBaseSize);
        }
        let data_len = self.data.as_ref().len();
        if data_len < self.stride * (self.height - 1) + self.width * cn {
            return Err(ForgeError::MinimumSliceSizeMismatch(MismatchedSize {
                expected: self.stride * self.height,
                received: data_len,
            }));
        }
        if (self.stride) < (self.width * cn) {
            return Err(ForgeError::MinimumStrideSizeMismatch(MismatchedSize {
                expected: self.width * cn,
                received: self.stride,
            }));
        }
        Ok(())
    }

    /// Returns row stride
    #[inline]
    pub fn row_stride(&self) -> usize {
        if self.stride == 0 {
            self.width * N
        } else {
            self.stride
        }
    }

    #[inline]
    pub fn check_layout(&self) -> Result<(), ForgeError> {
        if self.width == 0 || self.height == 0 {
            return Err(ForgeError::ZeroBaseSize);
        }
        let cn = N;
        if self.data.len() < self.stride * (self.height - 1) + self.width * cn {
            return Err(ForgeError::MinimumSliceSizeMismatch(MismatchedSize {
                expected: self.stride * self.height,
                received: self.data.len(),
            }));
        }
        if (self.stride) < (self.width * cn) {
            return Err(ForgeError::MinimumStrideSizeMismatch(MismatchedSize {
                expected: self.width * cn,
                received: self.stride,
            }));
        }
        Ok(())
    }
}

impl<'a, T: Clone + Copy + Default + Debug, const N: usize> GainImageMut<'a, T, N> {
    /// Allocates default image layout for given [FastBlurChannels]
    pub fn alloc(width: usize, height: usize) -> Self {
        Self {
            data: BufferStore::Owned(vec![T::default(); width * height * N]),
            width,
            height,
            stride: width * N,
        }
    }

    /// Mutable borrows existing data
    /// Stride will be default `width * channels.channels()`
    pub fn borrow(arr: &'a mut [T], width: usize, height: usize) -> Self {
        Self {
            data: BufferStore::Borrowed(arr),
            width,
            height,
            stride: width * N,
        }
    }

    /// Returns row stride
    #[inline]
    pub fn row_stride(&self) -> usize {
        if self.stride == 0 {
            self.width * N
        } else {
            self.stride
        }
    }

    /// Checks if layout matches necessary requirements
    #[inline]
    pub fn check_layout(&self) -> Result<(), ForgeError> {
        if self.width == 0 || self.height == 0 {
            return Err(ForgeError::ZeroBaseSize);
        }
        let data_len = self.data.borrow().len();
        if data_len < self.stride * (self.height - 1) + self.width * N {
            return Err(ForgeError::MinimumSliceSizeMismatch(MismatchedSize {
                expected: self.stride * self.height,
                received: data_len,
            }));
        }
        if (self.stride) < (self.width * N) {
            return Err(ForgeError::MinimumStrideSizeMismatch(MismatchedSize {
                expected: self.width * N,
                received: self.stride,
            }));
        }
        Ok(())
    }

    /// Checks if it is matches the size of the other image
    #[inline]
    pub fn size_matches(&self, other: &GainImage<'_, T, N>) -> Result<(), ForgeError> {
        if self.width == other.width && self.height == other.height {
            return Ok(());
        }
        Err(ForgeError::ImageSizeMismatch)
    }

    /// Checks if it is matches the size of the other image
    #[inline]
    pub fn size_matches_mut(&self, other: &GainImageMut<'_, T, N>) -> Result<(), ForgeError> {
        if self.width == other.width && self.height == other.height {
            return Ok(());
        }
        Err(ForgeError::ImageSizeMismatch)
    }

    pub fn to_immutable_ref(&self) -> GainImage<'_, T, N> {
        GainImage {
            data: std::borrow::Cow::Borrowed(self.data.borrow()),
            stride: self.row_stride(),
            width: self.width,
            height: self.height,
        }
    }
}

impl<const N: usize> GainImage<'_, u8, N> {
    pub fn expand_to_u16(&self, target_bit_depth: usize) -> GainImageMut<'_, u16, N> {
        assert!(target_bit_depth >= 8 || target_bit_depth <= 16);
        let mut new_image = GainImageMut::<u16, N>::alloc(self.width, self.height);
        let dst_stride = new_image.row_stride();
        let shift_left = target_bit_depth.saturating_sub(8) as u16;
        let shift_right = 8u16.saturating_sub(shift_left);
        for (src, dst) in self
            .data
            .as_ref()
            .chunks_exact(self.row_stride())
            .zip(new_image.data.borrow_mut().chunks_exact_mut(dst_stride))
        {
            let src = &src[..self.width * N];
            let dst = &mut dst[..self.width * N];
            for (&src, dst) in src.iter().zip(dst.iter_mut()) {
                *dst = ((src as u16) << shift_left) | ((src as u16) >> shift_right);
            }
        }
        new_image
    }
}

impl<const N: usize> GainImageMut<'_, u8, N> {
    pub fn expand_to_u16(&self, target_bit_depth: usize) -> GainImageMut<'_, u16, N> {
        assert!(target_bit_depth >= 8 || target_bit_depth <= 16);
        let mut new_image = GainImageMut::<u16, N>::alloc(self.width, self.height);
        let dst_stride = new_image.row_stride();
        let shift_left = target_bit_depth.saturating_sub(8) as u16;
        let shift_right = 8u16.saturating_sub(shift_left);
        let width = self.width;
        for (src, dst) in self
            .data
            .borrow()
            .chunks_exact(self.row_stride())
            .zip(new_image.data.borrow_mut().chunks_exact_mut(dst_stride))
        {
            let src = &src[..width * N];
            let dst = &mut dst[..width * N];
            for (&src, dst) in src.iter().zip(dst.iter_mut()) {
                *dst = ((src as u16) << shift_left) | ((src as u16) >> shift_right);
            }
        }
        new_image
    }
}
