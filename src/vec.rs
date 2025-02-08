use core::mem::MaybeUninit;
use crate::types::{Constants, Float};

/**
A vector type.
*/
#[derive(Debug,Copy,Clone,PartialEq)]
pub struct Vector<T, const N: usize>([T; N]);

impl <T, const N: usize> Vector<T, N> {
    /**
    Creates a new vector.
*/
    pub const fn new(value: [T; N]) -> Self {
        Self(value)
    }
}

impl<T: Constants, const N: usize> Vector<T, N> {
    /**
    Creates a new vector with all components set to zero.
*/
    pub const ZERO: Self = Self([T::ZERO; N]);
    pub const ONE: Self = Self([T::ONE; N]);

}

//getters

mod private {
    pub trait AtLeastOne {}
    impl<T> AtLeastOne for [T; 1] {}
    impl<T> AtLeastOne for [T; 2] {}
    impl<T> AtLeastOne for [T; 3] {}
    impl<T> AtLeastOne for [T; 4] {}
    pub trait AtLeastTwo{}
    impl<T> AtLeastTwo for [T; 2] {}
    impl<T> AtLeastTwo for [T; 3] {}
    impl<T> AtLeastTwo for [T; 4] {}

    pub trait AtLeastThree{}
    impl<T> AtLeastThree for [T; 3] {}
    impl<T> AtLeastThree for [T; 4] {}

    pub trait AtLeastFour{}
    impl<T> AtLeastFour for [T; 4] {}

}
impl<T, const N: usize> Vector<T, N>
where
    [T; N]: private::AtLeastOne
{
    /**
    Gets the first component.
*/
    #[inline] pub const fn x(&self) -> &T {
        &self.0[0]
    }
    /**
    Gets the first component.
    */
    #[inline] pub const fn x_mut(&mut self) -> &mut T {
        &mut self.0[0]
    }
}
impl<T, const N: usize> Vector<T, N>
where
    [T; N]: private::AtLeastTwo
{
    /**
    Gets the second component.
*/
    #[inline] pub const fn y(&self) -> &T {
        &self.0[1]
    }
    /**
    Gets the second component.
*/
    #[inline] pub const fn y_mut(&mut self) -> &mut T {
        &mut self.0[1]
    }
}

impl<T, const N: usize> Vector<T, N>
where
    [T; N]: private::AtLeastThree
{
    /**
    Gets the third component.
*/
    #[inline] pub const fn z(&self) -> &T {
        &self.0[2]
    }
    /**
    Gets the third component.
*/
    #[inline] pub const fn z_mut(&mut self) -> &mut T {
        &mut self.0[2]
    }
}

impl<T, const N: usize> Vector<T, N>
where
    [T; N]: private::AtLeastFour
{
    /**
    Gets the fourth component.
*/
    #[inline] pub const fn w(&self) -> &T {
        &self.0[3]
    }
    /**
    Gets the fourth component.
*/
    #[inline] pub const fn w_mut(&mut self) -> &mut T {
        &mut self.0[3]
    }
}

//elementwise ops

impl<T: core::ops::Add<Output=T> + Clone, const N: usize> Vector<T, N>
{
    #[inline] pub fn elementwise_add(self, other: Self) -> Self {
        let mut result = self.0.clone();
        for i in 0..N {
            result[i] = result[i].clone() + other.0[i].clone();
        }
        Self(result)
    }
}

impl<T: core::ops::Add<Output=T> + Clone, const N: usize>  core::ops::Add for Vector<T, N> {
    type Output = Self;
    #[inline] fn add(self, other: Self) -> Self {
        self.elementwise_add(other)
    }
}

impl<T: core::ops::Sub<Output=T> + Clone, const N: usize> Vector<T, N>
{
    #[inline] pub fn elementwise_sub(self, other: Self) -> Self {
        let mut result = self.0.clone();
        for i in 0..N {
            result[i] = result[i].clone() - other.0[i].clone();
        }
        Self(result)
    }
}

impl<T: core::ops::Sub<Output=T> + Clone, const N: usize>  core::ops::Sub for Vector<T, N> {
    type Output = Self;
    #[inline] fn sub(self, other: Self) -> Self {
        self.elementwise_sub(other)
    }
}

impl<T: core::ops::Mul<Output=T> + Clone, const N: usize> Vector<T, N> {
    #[inline] pub fn elementwise_mul(self, other: Self) -> Self {
        let mut result = self.0.clone();
        for i in 0..N {
            result[i] = result[i].clone() * other.0[i].clone();
        }
        Self(result)
    }
}

impl<T: core::ops::Mul<Output=T> + Clone, const N: usize>  core::ops::Mul for Vector<T, N> {
    type Output = Self;
    #[inline] fn mul(self, other: Self) -> Self {
        self.elementwise_mul(other)
    }
}

impl<T: core::ops::Div<Output=T> + Clone, const N: usize> Vector<T, N> {
    #[inline] pub fn elementwise_div(self, other: Self) -> Self {
        let mut result = self.0.clone();
        for i in 0..N {
            result[i] = result[i].clone() / other.0[i].clone();
        }
        Self(result)
    }
}

impl<T: core::ops::Div<Output=T> + Clone, const N: usize>  core::ops::Div for Vector<T, N> {
    type Output = Self;
    #[inline] fn div(self, other: Self) -> Self {
        self.elementwise_div(other)
    }
}

impl<T: core::ops::AddAssign + Clone, const N: usize>  Vector<T, N> {
    #[inline] pub fn add_elementwise_assign(&mut self, other: Self) {
        for i in 0..N {
            self.0[i] += other.0[i].clone();
        }
    }
}

impl<T: core::ops::AddAssign + Clone, const N: usize>  core::ops::AddAssign for Vector<T, N> {
    #[inline] fn add_assign(&mut self, other: Self) {
        self.add_elementwise_assign(other)
    }
}

impl<T: core::ops::SubAssign + Clone, const N: usize>  Vector<T, N> {
    #[inline] pub fn sub_elementwise_assign(&mut self, other: Self) {
        for i in 0..N {
            self.0[i] -= other.0[i].clone();
        }
    }
}

impl<T: core::ops::SubAssign + Clone, const N: usize>  core::ops::SubAssign for Vector<T, N> {
    #[inline] fn sub_assign(&mut self, other: Self) {
        self.sub_elementwise_assign(other)
    }
}

impl<T: core::ops::MulAssign + Clone, const N: usize>  Vector<T, N> {
    #[inline] fn mul_elementwise_assign(&mut self, other: Self) {
        for i in 0..N {
            self.0[i] *= other.0[i].clone();
        }
    }
}

impl<T: core::ops::MulAssign + Clone, const N: usize>  core::ops::MulAssign for Vector<T, N> {
    #[inline] fn mul_assign(&mut self, other: Self) {
        self.mul_elementwise_assign(other)
    }
}

impl<T: core::ops::DivAssign + Clone, const N: usize>  Vector<T, N> {
    #[inline] fn div_elementwise_assign(&mut self, other: Self) {
        for i in 0..N {
            self.0[i] /= other.0[i].clone();
        }
    }
}

impl<T: core::ops::DivAssign + Clone, const N: usize>  core::ops::DivAssign for Vector<T, N> {
    #[inline] fn div_assign(&mut self, other: Self) {
        self.div_elementwise_assign(other)
    }
}

//scalar ops
impl<T: core::ops::Add<Output=T> + Clone, const N: usize>  Vector<T, N> {
    #[inline] pub fn add_scalar(self, other: T) -> Self {
        let mut result = self.0.clone();
        for i in 0..N {
            result[i] = result[i].clone() + other.clone();
        }
        Self(result)
    }
}

impl<T: core::ops::Add<Output=T> + Clone, const N: usize>  core::ops::Add<T> for Vector<T, N> {
    type Output = Self;
    #[inline] fn add(self, other: T) -> Self {
        self.add_scalar(other)
    }
}

impl<T: core::ops::Sub<Output=T> + Clone, const N: usize>  Vector<T, N> {
    #[inline] pub fn sub_scalar(self, other: T) -> Self {
        let mut result = self.0.clone();
        for i in 0..N {
            result[i] = result[i].clone() - other.clone();
        }
        Self(result)
    }
}

impl<T: core::ops::Sub<Output=T> + Clone, const N: usize>  core::ops::Sub<T> for Vector<T, N> {
    type Output = Self;
    #[inline] fn sub(self, other: T) -> Self {
        self.sub_scalar(other)
    }
}

impl<T: core::ops::Mul<Output=T> + Clone, const N: usize>  Vector<T, N> {
    #[inline] pub fn mul_scalar(self, other: T) -> Self {
        let mut result = self.0.clone();
        for i in 0..N {
            result[i] = result[i].clone() * other.clone();
        }
        Self(result)
    }
}

impl<T: core::ops::Mul<Output=T> + Clone, const N: usize>  core::ops::Mul<T> for Vector<T, N> {
    type Output = Self;
    #[inline] fn mul(self, other: T) -> Self {
        self.mul_scalar(other)
    }
}

impl<T: core::ops::Div<Output=T> + Clone, const N: usize>  Vector<T, N> {
    #[inline] pub fn div_scalar(self, other: T) -> Self {
        let mut result = self.0.clone();
        for i in 0..N {
            result[i] = result[i].clone() / other.clone();
        }
        Self(result)
    }
}

impl<T: core::ops::Div<Output=T> + Clone, const N: usize>  core::ops::Div<T> for Vector<T, N> {
    type Output = Self;
    #[inline] fn div(self, other: T) -> Self {
        self.div_scalar(other)
    }
}

impl<T: core::ops::AddAssign + Clone, const N: usize>  Vector<T, N> {
    #[inline] pub fn add_assign_scalar(&mut self, other: T) {
        for i in 0..N {
            self.0[i] += other.clone();
        }
    }
}

impl<T: core::ops::AddAssign + Clone, const N: usize> core::ops::AddAssign<T> for Vector<T, N> {
    #[inline] fn add_assign(&mut self, other: T) {
        self.add_assign_scalar(other)
    }
}

impl<T: core::ops::SubAssign + Clone, const N: usize>  Vector<T, N> {
    #[inline] pub fn sub_assign_scalar(&mut self, other: T) {
        for i in 0..N {
            self.0[i] -= other.clone();
        }
    }
}

impl<T: core::ops::SubAssign + Clone, const N: usize> core::ops::SubAssign<T> for Vector<T, N> {
    #[inline] fn sub_assign(&mut self, other: T) {
        self.sub_assign_scalar(other)
    }
}

impl <T: core::ops::MulAssign + Clone, const N: usize>  Vector<T, N> {
    #[inline] pub fn mul_assign_scalar(&mut self, other: T) {
        for i in 0..N {
            self.0[i] *= other.clone();
        }
    }
}

impl<T: core::ops::MulAssign + Clone, const N: usize> core::ops::MulAssign<T> for Vector<T, N> {
    #[inline] fn mul_assign(&mut self, other: T) {
        self.mul_assign_scalar(other)
    }
}

impl <T: core::ops::DivAssign + Clone, const N: usize>  Vector<T, N> {
    #[inline] pub fn div_assign_scalar(&mut self, other: T) {
        for i in 0..N {
            self.0[i] /= other.clone();
        }
    }
}

impl<T: core::ops::DivAssign + Clone, const N: usize> core::ops::DivAssign<T> for Vector<T, N> {
    #[inline] fn div_assign(&mut self, other: T) {
        self.div_assign_scalar(other)
    }
}

//length squared
impl <T: core::ops::Mul<Output=T> + Clone + Copy + core::ops::Add<Output=T>, const N: usize>  Vector<T, N> {
    #[inline] pub fn length_squared(self) -> T {
        let mut result = self.0[0].clone() * self.0[0].clone();
        for i in 1..N {
            result = result + self.0[i].clone() * self.0[i].clone();
        }
        result
    }
}

impl <T: core::ops::Mul<Output=T> + Clone + Copy + core::ops::Add<Output=T> + Float, const N: usize> Vector<T,N> {
    #[inline] pub fn length(self) -> T {
        self.length_squared().sqrt()
    }
}

//euclid_distance_to

impl <T: core::ops::Sub<Output=T> + Clone + Copy + core::ops::Mul<Output=T> + core::ops::Add<Output=T> + Float, const N: usize>  Vector<T, N> {
    #[inline] pub fn euclid_distance_to(self, other: Self) -> T {
        let mut result = self.0[0].clone() - other.0[0].clone();
        result = result * result;
        for i in 1..N {
            let diff = self.0[i].clone() - other.0[i].clone();
            result = result + diff * diff;
        }
        result.sqrt()
    }
}

//norm
impl <T: core::ops::Div<Output=T> + Clone + Copy + core::ops::Mul<Output=T> + core::ops::Add<Output=T> + Float, const N: usize>  Vector<T, N> {
    #[inline] pub fn normalize(self) -> Self {
        self / self.length()
    }
}

impl<T, const N: usize> Vector<MaybeUninit<T>,N> where T: Sized {
    #[inline] pub unsafe fn assume_init(self) -> Vector<T,N> {
        let inner = self.0;
        let arr: [T; N] = inner.map(|maybe| unsafe { maybe.assume_init() });
        Vector(arr)
    }
}

//cross-product
impl <T: core::ops::Sub<Output=T> + Clone + Copy + core::ops::Mul<Output=T> + core::ops::Add<Output=T>>  Vector<T, 3>
{
    #[inline] pub fn cross(self, other: Self) -> Self {
        let x = self.y().clone() * other.z().clone() - self.z().clone() * other.y().clone();
        let y = self.z().clone() * other.x().clone() - self.x().clone() * other.z().clone();
        let z = self.x().clone() * other.y().clone() - self.y().clone() * other.x().clone();
        Vector::new([x,y,z])
    }
}


impl<T, const N: usize> Vector<MaybeUninit<T>, N> {
    ///A vector of [MaybeUninit::uninit] values.
    pub const UNINIT: Self = Self([const { MaybeUninit::uninit() }; N]);
}

//map
impl <T, const N: usize>  Vector<T, N>
where
    T: Clone,
{
    #[inline] pub fn map<F,U>(self, mut f: F) -> Vector<U, N>
    where
        F: FnMut(T) -> U
    {
        let mut result = Vector::UNINIT;

        for i in 0..N {
            result.0[i] = MaybeUninit::new(f(self.0[i].clone()));
        }
        unsafe { result.assume_init() }
    }
}

//map_in_place
impl <T, const N: usize>  Vector<T, N>
{
    #[inline] pub fn map_in_place<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T)
    {
        for i in 0..N {
            f(&mut self.0[i]);
        }
    }
}



//eq_approx
impl <T: Float + Clone, const N: usize>  Vector<T, N> {
    #[inline] pub fn eq_approx(self, other: Self, tolerance: T) -> bool {
        for i in 0..N {
            if !self.0[i].clone().eq_approx(other.0[i].clone(), tolerance.clone()) {
                return false;
            }
        }
        true
    }
}

//min, max
impl <T: core::cmp::PartialOrd + Clone, const N: usize>  Vector<T, N> {
    #[inline] pub fn max(self) -> T {
        let mut result = self.0[0].clone();
        for i in 1..N {
            if self.0[i].clone() > result {
                result = self.0[i].clone();
            }
        }
        result
    }

    #[inline] pub fn min(self) -> T {
        let mut result = self.0[0].clone();
        for i in 1..N {
            if self.0[i].clone() < result {
                result = self.0[i].clone();
            }
        }
        result
    }
}

//dot product
impl <T: core::ops::Mul<Output=T> + Clone + Copy + core::ops::Add<Output=T>, const N: usize>  Vector<T, N>
{
    #[inline] pub fn dot(self, other: Self) -> T {
        let mut result = self.0[0].clone() * other.0[0].clone();
        for i in 1..N {
            result = result + self.0[i].clone() * other.0[i].clone();
        }
        result
    }
}

//clamp
impl <T: core::cmp::PartialOrd + Clone, const N: usize>  Vector<T, N>
{
    #[inline] pub fn clamp(self, min: T, max: T) -> Self {
        let mut result = self.0.clone();
        for i in 0..N {
            if result[i].clone() < min.clone() {
                result[i] = min.clone();
            }
            else if result[i].clone() > max.clone() {
                result[i] = max.clone();
            }
        }
        Self(result)
    }
}

//mix
impl <T: Float + Constants + Clone + core::ops::Sub<Output=T> + core::ops::Mul<Output=T> + core::ops::Add<Output=T>, const N: usize>  Vector<T, N>
{
    #[inline] pub fn mix(self, other: Self, weight: T) -> Self {
        let mut result = self.0.clone();
        for i in 0..N {
            result[i] = self.0[i].clone() * (T::ONE - weight.clone()) + other.0[i].clone() * weight.clone();
        }
        Self(result)
    }
}

//index
impl <T, const N: usize>  core::ops::Index<usize> for Vector<T, N>
{
    type Output = T;
    #[inline] fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

//index_mut
impl <T, const N: usize>  core::ops::IndexMut<usize> for Vector<T, N>
{
    #[inline] fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[cfg(test)] mod tests {
    use crate::vec::Vector;

    #[test] fn elementwise() {
        let a = Vector::new([1,2,3]);
        let b = Vector::new([4,5,6]);
        let c = a + b;
        assert_eq!(c.0, [5,7,9]);

        let d = a - b;
        assert_eq!(d.0, [-3,-3,-3]);

        let e = a * b;
        assert_eq!(e.0, [4,10,18]);

        let f = a / b;
        assert_eq!(f.0, [0,0,0]);
    }

    #[test] fn elementwise_assign() {
        let mut a = Vector::new([1,2,3]);
        let b = Vector::new([4,5,6]);
        a += b;
        assert_eq!(a.0, [5,7,9]);

        let mut c = Vector::new([1,2,3]);
        let d = Vector::new([4,5,6]);
        c -= d;
        assert_eq!(c.0, [-3,-3,-3]);

        let mut e = Vector::new([1,2,3]);
        let f = Vector::new([4,5,6]);
        e *= f;
        assert_eq!(e.0, [4,10,18]);

        let mut g = Vector::new([1,2,3]);
        let h = Vector::new([4,5,6]);
        g /= h;
        assert_eq!(g.0, [0,0,0]);
    }

    #[test] fn scalar() {
        let a = Vector::new([1,2,3]);
        let b = a + 1;
        assert_eq!(b.0, [2,3,4]);

        let c = a - 1;
        assert_eq!(c.0, [0,1,2]);

        let d = a * 2;
        assert_eq!(d.0, [2,4,6]);

        let e = a / 2;
        assert_eq!(e.0, [0,1,1]);
    }

    #[test] fn scalar_assign() {
        let mut a = Vector::new([1,2,3]);
        a += 1;
        assert_eq!(a.0, [2,3,4]);

        let mut b = Vector::new([1,2,3]);
        b -= 1;
        assert_eq!(b.0, [0,1,2]);

        let mut c = Vector::new([1,2,3]);
        c *= 2;
        assert_eq!(c.0, [2,4,6]);

        let mut d = Vector::new([1,2,3]);
        d /= 2;
        assert_eq!(d.0, [0,1,1]);
    }

    #[test] fn length() {
        let a = Vector::new([3.0,4.0]);
        assert_eq!(a.length(), 5.0);
    }

    #[test] fn euclid_distance() {
        let a = Vector::new([1.0,2.0]);
        let b = Vector::new([4.0,6.0]);
        assert_eq!(a.euclid_distance_to(b), 5.0);
    }

    #[test] fn normalize() {
        let a = Vector::new([3.0,4.0]);
        let b = a.normalize();
        assert_eq!(b.length(), 1.0);
    }

    #[test] fn map() {
        let a = Vector::new([1i32,-2,3]);
        let b = a.map(|x| x.abs());
        assert_eq!(b.0, [1,2,3]);
    }

    #[test] fn min_max() {
        let a = Vector::new([1,2,3]);
        assert_eq!(a.min(), 1);
        assert_eq!(a.max(), 3);
    }

    #[test] fn eq_approx() {
        let a = Vector::new([1.0,2.0,3.0]);
        let b = Vector::new([1.0,2.005,3.0]);
        assert!(a.eq_approx(b, 0.01));
        assert!(!a.eq_approx(b, 0.001));

    }
    #[test] fn test_cross() {
        let a = Vector::new([1,2,3]);
        let b = Vector::new([4,5,6]);
        let c = a.cross(b);
        assert_eq!(c.0, [-3,6,-3]);
    }

    #[test] fn test_dot() {
        let a = Vector::new([1,2,3]);
        let b = Vector::new([4,5,6]);
        let c = a.dot(b);
        assert_eq!(c, 32);
    }

    #[test] fn test_clamp() {
        let a = Vector::new([1,2,3,4]);
        let b = a.clamp(2, 3);
        assert_eq!(b.0, [2,2,3,3]);
    }

    #[test] fn test_mix() {
        let a = Vector::new([1.0f32,2.0,3.0,4.0]);
        let b = a.mix(Vector::new([4.0f32,3.0,2.0,1.0]), 0.5);
        assert_eq!(b.0, [2.5,2.5,2.5,2.5]);
    }


}