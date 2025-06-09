//SPDX-License-Identifier: MIT OR Apache-2.0
/*!
* f32
* f64
* u8
* u16
* u32
* u64
* i8
* i16
* i32
* i64
*/

use crate::types::sealed::{Constants, Float};

impl Constants for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl Constants for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl Constants for u8 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Constants for u16 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Constants for u32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Constants for u64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Constants for i8 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Constants for i16 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Constants for i32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Constants for i64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}
pub(crate) mod sealed {
    pub trait Float {
        #[cfg(feature = "std")]
        fn sqrt(self) -> Self;
        fn eq_approx(self, other: Self, tolerance: Self) -> bool;

        #[cfg(feature = "std")]
        fn sin(self) -> Self;
        #[cfg(feature = "std")]
        fn cos(self) -> Self;
    }
    pub trait Constants {
        const ZERO: Self;
        const ONE: Self;
    }
}

impl Float for f32 {
    #[cfg(feature = "std")]
    #[inline]
    fn sqrt(self) -> Self {
        let sqrt_fn: fn(f32) -> f32 = f32::sqrt;
        sqrt_fn(self)
    }
    #[inline]
    fn eq_approx(self, other: Self, tolerance: Self) -> bool {
        (self - other).abs() < tolerance
    }

    #[cfg(feature = "std")]
    #[inline]
    fn sin(self) -> Self {
        let sin_fn: fn(f32) -> f32 = f32::sin;
        sin_fn(self)
    }
    #[cfg(feature = "std")]
    #[inline]
    fn cos(self) -> Self {
        let cos_fn: fn(f32) -> f32 = f32::cos;
        cos_fn(self)
    }
}
impl Float for f64 {
    #[cfg(feature = "std")]
    #[inline]
    fn sqrt(self) -> Self {
        let sqrt_fn: fn(f64) -> f64 = f64::sqrt;
        sqrt_fn(self)
    }
    #[inline]
    fn eq_approx(self, other: Self, tolerance: Self) -> bool {
        (self - other).abs() < tolerance
    }

    #[cfg(feature = "std")]
    #[inline]
    fn sin(self) -> Self {
        let sin_fn: fn(f64) -> f64 = f64::sin;
        sin_fn(self)
    }
    #[cfg(feature = "std")]
    #[inline]
    fn cos(self) -> Self {
        let cos_fn: fn(f64) -> f64 = f64::cos;
        cos_fn(self)
    }
}
