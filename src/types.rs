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
pub trait Constants {
    const ZERO: Self;
    const ONE: Self;
}

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

pub trait Float {
    fn sqrt(self) -> Self;
    fn eq_approx(self, other: Self, tolerance: Self) -> bool;

    fn sin(self) -> Self;
    fn cos(self) -> Self;
}

impl Float for f32 {
    #[inline] fn sqrt(self) -> Self {
        let sqrt_fn: fn(f32) -> f32 = f32::sqrt;
        sqrt_fn(self)
    }
    #[inline] fn eq_approx(self, other: Self, tolerance: Self) -> bool {
        (self - other).abs() < tolerance
    }

    #[inline] fn sin(self) -> Self {
        let sin_fn: fn(f32) -> f32 = f32::sin;
        sin_fn(self)
    }
    #[inline] fn cos(self) -> Self {
        let cos_fn: fn(f32) -> f32 = f32::cos;
        cos_fn(self)
    }
}
impl Float for f64 {
    #[inline] fn sqrt(self) -> Self {
        let sqrt_fn: fn(f64) -> f64 = f64::sqrt;
        sqrt_fn(self)
    }
    #[inline] fn eq_approx(self, other: Self, tolerance: Self) -> bool {
        (self - other).abs() < tolerance
    }

    #[inline] fn sin(self) -> Self {
        let sin_fn: fn(f64) -> f64 = f64::sin;
        sin_fn(self)
    }
    #[inline] fn cos(self) -> Self {
        let cos_fn: fn(f64) -> f64 = f64::cos;
        cos_fn(self)
    }

}
