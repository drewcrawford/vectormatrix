//SPDX-License-Identifier: MIT OR Apache-2.0
//! Mathematical vector types for linear algebra operations.
//!
//! This module provides a generic `Vector<T, N>` type that represents an N-dimensional
//! mathematical vector with elements of type `T`. It also includes a `NormalizedVector<T, N>`
//! type that guarantees unit length through the type system.
//!
//! # Features
//!
//! - Generic over element type and dimension
//! - Stack-allocated with const generics
//! - Comprehensive arithmetic operations
//! - Type-safe normalized vectors
//! - Conversion to/from arrays and tuples
//! - Integration with matrix operations
//!
//! # Examples
//!
//! ```
//! use vectormatrix::vector::Vector;
//!
//! // Create vectors from arrays
//! let v1 = Vector::new([1.0, 2.0, 3.0]);
//! let v2 = Vector::new([4.0, 5.0, 6.0]);
//!
//! // Basic arithmetic
//! let sum = v1 + v2;
//! let scaled = v1 * 2.0;
//!
//! // Vector operations
//! let dot_product = v1.dot(v2);
//! let cross_product = v1.cross(v2);
//! ```

use crate::matrix::Matrix;
use crate::types::sealed::{Constants, Float};
use core::mem::MaybeUninit;

/// A mathematical vector with `N` elements of type `T`.
///
/// `Vector` represents a column vector in N-dimensional space. It supports
/// standard vector operations like addition, scalar multiplication, dot product,
/// and cross product (for 3D vectors).
///
/// # Type Parameters
///
/// - `T`: The type of each element in the vector
/// - `N`: The number of dimensions (compile-time constant)
///
/// # Examples
///
/// ```
/// use vectormatrix::vector::Vector;
///
/// // Create a 3D vector
/// let v = Vector::new([1.0, 2.0, 3.0]);
///
/// // Access components
/// assert_eq!(*v.x(), 1.0);
/// assert_eq!(*v.y(), 2.0);
/// assert_eq!(*v.z(), 3.0);
///
/// // Convert from tuple
/// let v2: Vector<f64, 3> = (4.0, 5.0, 6.0).into();
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Vector<T, const N: usize>([T; N]);

impl<T, const N: usize> Vector<T, N> {
    /// Creates a new vector from an array of elements.
    ///
    /// # Arguments
    ///
    /// * `value` - An array containing the vector elements
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([1, 2, 3, 4]);
    /// assert_eq!(v.into_inner(), [1, 2, 3, 4]);
    /// ```
    pub const fn new(value: [T; N]) -> Self {
        Self(value)
    }

    /// Consumes the vector and returns the underlying array.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([1.0, 2.0, 3.0]);
    /// let array = v.into_inner();
    /// assert_eq!(array, [1.0, 2.0, 3.0]);
    /// ```
    pub fn into_inner(self) -> [T; N] {
        self.0
    }
}

impl<T: Constants, const N: usize> Vector<T, N> {
    /// A vector with all components set to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v: Vector<f32, 3> = Vector::ZERO;
    /// assert_eq!(v.into_inner(), [0.0, 0.0, 0.0]);
    /// ```
    pub const ZERO: Self = Self([T::ZERO; N]);

    /// A vector with all components set to one.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v: Vector<i32, 4> = Vector::ONE;
    /// assert_eq!(v.into_inner(), [1, 1, 1, 1]);
    /// ```
    pub const ONE: Self = Self([T::ONE; N]);
}

/// A vector that is guaranteed to have unit length.
///
/// `NormalizedVector` is a wrapper around `Vector` that ensures the vector
/// always has a length of 1. This property is enforced by the type system -
/// you can only create a `NormalizedVector` by normalizing an existing vector.
///
/// # Type Parameters
///
/// - `T`: The type of each element (must implement `Float` for normalization)
/// - `N`: The number of dimensions
///
/// # Examples
///
/// ```rust
///# #[cfg(feature = "std")] {
/// use vectormatrix::vector::Vector;
///
/// let v = Vector::new([3.0f64, 4.0]);
/// let normalized = v.normalize();
///
/// // The normalized vector has length 1
/// let length = normalized.as_vector().length();
/// assert!((length - 1.0f64).abs() < 0.0001);
/// # }
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NormalizedVector<T, const N: usize>(Vector<T, N>);

impl<T, const N: usize> NormalizedVector<T, N> {
    /// Consumes the normalized vector and returns the underlying vector.
    ///
    /// # Examples
    ///
    /// ```
    /// #[cfg(feature = "std")]
    /// # {
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([3.0, 4.0]);
    /// let normalized = v.normalize();
    /// let vector_back = normalized.into_vector();
    /// # }
    /// ```
    #[inline]
    pub fn into_vector(self) -> Vector<T, N> {
        self.0
    }

    /// Returns a reference to the underlying vector.
    ///
    /// # Examples
    ///
    /// ```
    /// #[cfg(feature = "std")]
    /// # {
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([3.0, 4.0]);
    /// let normalized = v.normalize();
    /// let vec_ref = normalized.as_vector();
    /// assert_eq!(vec_ref, &normalized.into_vector());
    /// # }
    /// ```
    #[inline]
    pub const fn as_vector(&self) -> &Vector<T, N> {
        &self.0
    }
}

//getters

mod private {
    pub trait AtLeastOne {}
    impl<T> AtLeastOne for [T; 1] {}
    impl<T> AtLeastOne for [T; 2] {}
    impl<T> AtLeastOne for [T; 3] {}
    impl<T> AtLeastOne for [T; 4] {}
    pub trait AtLeastTwo {}
    impl<T> AtLeastTwo for [T; 2] {}
    impl<T> AtLeastTwo for [T; 3] {}
    impl<T> AtLeastTwo for [T; 4] {}

    pub trait AtLeastThree {}
    impl<T> AtLeastThree for [T; 3] {}
    impl<T> AtLeastThree for [T; 4] {}

    pub trait AtLeastFour {}
    impl<T> AtLeastFour for [T; 4] {}
}
impl<T, const N: usize> Vector<T, N>
where
    [T; N]: private::AtLeastOne,
{
    /// Returns a reference to the x (first) component.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([1.0, 2.0, 3.0]);
    /// assert_eq!(*v.x(), 1.0);
    /// ```
    #[inline]
    pub const fn x(&self) -> &T {
        &self.0[0]
    }

    /// Returns a mutable reference to the x (first) component.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let mut v = Vector::new([1.0, 2.0, 3.0]);
    /// *v.x_mut() = 5.0;
    /// assert_eq!(*v.x(), 5.0);
    /// ```
    #[inline]
    pub const fn x_mut(&mut self) -> &mut T {
        &mut self.0[0]
    }
}
impl<T, const N: usize> Vector<T, N>
where
    [T; N]: private::AtLeastTwo,
{
    /// Returns a reference to the y (second) component.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([1.0, 2.0, 3.0]);
    /// assert_eq!(*v.y(), 2.0);
    /// ```
    #[inline]
    pub const fn y(&self) -> &T {
        &self.0[1]
    }

    /// Returns a mutable reference to the y (second) component.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let mut v = Vector::new([1.0, 2.0, 3.0]);
    /// *v.y_mut() = 5.0;
    /// assert_eq!(*v.y(), 5.0);
    /// ```
    #[inline]
    pub const fn y_mut(&mut self) -> &mut T {
        &mut self.0[1]
    }
}

impl<T, const N: usize> Vector<T, N>
where
    [T; N]: private::AtLeastThree,
{
    /// Returns a reference to the z (third) component.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([1.0, 2.0, 3.0]);
    /// assert_eq!(*v.z(), 3.0);
    /// ```
    #[inline]
    pub const fn z(&self) -> &T {
        &self.0[2]
    }

    /// Returns a mutable reference to the z (third) component.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let mut v = Vector::new([1.0, 2.0, 3.0]);
    /// *v.z_mut() = 5.0;
    /// assert_eq!(*v.z(), 5.0);
    /// ```
    #[inline]
    pub const fn z_mut(&mut self) -> &mut T {
        &mut self.0[2]
    }
}

impl<T, const N: usize> Vector<T, N>
where
    [T; N]: private::AtLeastFour,
{
    /// Returns a reference to the w (fourth) component.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(*v.w(), 4.0);
    /// ```
    #[inline]
    pub const fn w(&self) -> &T {
        &self.0[3]
    }

    /// Returns a mutable reference to the w (fourth) component.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let mut v = Vector::new([1.0, 2.0, 3.0, 4.0]);
    /// *v.w_mut() = 5.0;
    /// assert_eq!(*v.w(), 5.0);
    /// ```
    #[inline]
    pub const fn w_mut(&mut self) -> &mut T {
        &mut self.0[3]
    }
}

//elementwise ops

impl<T: core::ops::Add<Output = T> + Clone, const N: usize> Vector<T, N> {
    /// Performs element-wise addition with another vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v1 = Vector::new([1, 2, 3]);
    /// let v2 = Vector::new([4, 5, 6]);
    /// let result = v1.elementwise_add(v2);
    /// assert_eq!(result.into_inner(), [5, 7, 9]);
    /// ```
    #[inline]
    pub fn elementwise_add(self, other: Self) -> Self {
        let mut result = [const { MaybeUninit::<T>::uninit() }; N];
        let mut n = 0;
        while n < N {
            result[n] = MaybeUninit::new(self.0[n].clone() + other.0[n].clone());
            n += 1;
        }
        // SAFETY: All elements are initialized by the loop above
        let result: [T; N] = unsafe { core::ptr::read(result.as_ptr() as *const [T; N]) };
        Self(result)
    }
}

impl<T: core::ops::Add<Output = T> + Clone, const N: usize> core::ops::Add for Vector<T, N> {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        self.elementwise_add(other)
    }
}

impl<T: core::ops::Sub<Output = T> + Clone, const N: usize> Vector<T, N> {
    /// Performs element-wise subtraction with another vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v1 = Vector::new([5, 7, 9]);
    /// let v2 = Vector::new([1, 2, 3]);
    /// let result = v1.elementwise_sub(v2);
    /// assert_eq!(result.into_inner(), [4, 5, 6]);
    /// ```
    #[inline]
    pub fn elementwise_sub(self, other: Self) -> Self {
        let mut result = self.0.clone();
        for (i, result_item) in result.iter_mut().enumerate().take(N) {
            *result_item = result_item.clone() - other.0[i].clone();
        }
        Self(result)
    }
}

impl<T: core::ops::Sub<Output = T> + Clone, const N: usize> core::ops::Sub for Vector<T, N> {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        self.elementwise_sub(other)
    }
}

impl<T: core::ops::Mul<Output = T> + Clone, const N: usize> Vector<T, N> {
    /// Performs element-wise multiplication with another vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v1 = Vector::new([2, 3, 4]);
    /// let v2 = Vector::new([5, 6, 7]);
    /// let result = v1.elementwise_mul(v2);
    /// assert_eq!(result.into_inner(), [10, 18, 28]);
    /// ```
    #[inline]
    pub fn elementwise_mul(self, other: Self) -> Self {
        let mut result = [const { MaybeUninit::uninit() }; N];
        let mut n = 0;
        while n < N {
            result[n] = MaybeUninit::new(self.0[n].clone() * other.0[n].clone());
            n += 1;
        }
        // SAFETY: All elements are initialized by the loop above
        let result: [T; N] = unsafe { core::ptr::read(result.as_ptr() as *const [T; N]) };
        Self(result)
    }
}

impl<T: core::ops::Mul<Output = T> + Clone, const N: usize> core::ops::Mul for Vector<T, N> {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        self.elementwise_mul(other)
    }
}

impl<T: core::ops::Div<Output = T> + Clone, const N: usize> Vector<T, N> {
    /// Performs element-wise division with another vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v1 = Vector::new([10.0, 18.0, 28.0]);
    /// let v2 = Vector::new([2.0, 3.0, 4.0]);
    /// let result = v1.elementwise_div(v2);
    /// assert_eq!(result.into_inner(), [5.0, 6.0, 7.0]);
    /// ```
    #[inline]
    pub fn elementwise_div(self, other: Self) -> Self {
        let mut result = self.0.clone();
        for (i, result_item) in result.iter_mut().enumerate().take(N) {
            *result_item = result_item.clone() / other.0[i].clone();
        }
        Self(result)
    }
}

impl<T: core::ops::Div<Output = T> + Clone, const N: usize> core::ops::Div for Vector<T, N> {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self {
        self.elementwise_div(other)
    }
}

impl<T: core::ops::AddAssign + Clone, const N: usize> Vector<T, N> {
    /// Adds another vector to this one in place (element-wise).
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let mut v1 = Vector::new([1, 2, 3]);
    /// let v2 = Vector::new([4, 5, 6]);
    /// v1.add_elementwise_assign(v2);
    /// assert_eq!(v1.into_inner(), [5, 7, 9]);
    /// ```
    #[inline]
    pub fn add_elementwise_assign(&mut self, other: Self) {
        for i in 0..N {
            self.0[i] += other.0[i].clone();
        }
    }
}

impl<T: core::ops::AddAssign + Clone, const N: usize> core::ops::AddAssign for Vector<T, N> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.add_elementwise_assign(other)
    }
}

impl<T: core::ops::SubAssign + Clone, const N: usize> Vector<T, N> {
    #[inline]
    pub fn sub_elementwise_assign(&mut self, other: Self) {
        for i in 0..N {
            self.0[i] -= other.0[i].clone();
        }
    }
}

impl<T: core::ops::SubAssign + Clone, const N: usize> core::ops::SubAssign for Vector<T, N> {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.sub_elementwise_assign(other)
    }
}

impl<T: core::ops::MulAssign + Clone, const N: usize> Vector<T, N> {
    #[inline]
    fn mul_elementwise_assign(&mut self, other: Self) {
        for i in 0..N {
            self.0[i] *= other.0[i].clone();
        }
    }
}

impl<T: core::ops::MulAssign + Clone, const N: usize> core::ops::MulAssign for Vector<T, N> {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.mul_elementwise_assign(other)
    }
}

impl<T: core::ops::DivAssign + Clone, const N: usize> Vector<T, N> {
    #[inline]
    fn div_elementwise_assign(&mut self, other: Self) {
        for i in 0..N {
            self.0[i] /= other.0[i].clone();
        }
    }
}

impl<T: core::ops::DivAssign + Clone, const N: usize> core::ops::DivAssign for Vector<T, N> {
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.div_elementwise_assign(other)
    }
}

//scalar ops
impl<T: core::ops::Add<Output = T> + Clone, const N: usize> Vector<T, N> {
    /// Adds a scalar value to each element of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([1, 2, 3]);
    /// let result = v.add_scalar(5);
    /// assert_eq!(result.into_inner(), [6, 7, 8]);
    /// ```
    #[inline]
    pub fn add_scalar(self, other: T) -> Self {
        Self(self.0.map(|x| x + other.clone()))
    }
}

impl<T: core::ops::Add<Output = T> + Clone, const N: usize> core::ops::Add<T> for Vector<T, N> {
    type Output = Self;
    #[inline]
    fn add(self, other: T) -> Self {
        self.add_scalar(other)
    }
}

impl<T: core::ops::Sub<Output = T> + Clone, const N: usize> Vector<T, N> {
    /// Subtracts a scalar value from each element of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([5, 6, 7]);
    /// let result = v.sub_scalar(2);
    /// assert_eq!(result.into_inner(), [3, 4, 5]);
    /// ```
    #[inline]
    pub fn sub_scalar(self, other: T) -> Self {
        Self(self.0.map(|x| x - other.clone()))
    }
}

impl<T: core::ops::Sub<Output = T> + Clone, const N: usize> core::ops::Sub<T> for Vector<T, N> {
    type Output = Self;
    #[inline]
    fn sub(self, other: T) -> Self {
        self.sub_scalar(other)
    }
}

impl<T: core::ops::Mul<Output = T> + Clone, const N: usize> Vector<T, N> {
    /// Multiplies each element of the vector by a scalar value.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([2, 3, 4]);
    /// let result = v.mul_scalar(3);
    /// assert_eq!(result.into_inner(), [6, 9, 12]);
    /// ```
    #[inline]
    pub fn mul_scalar(self, other: T) -> Self {
        let mut result = [const { MaybeUninit::uninit() }; N];
        let mut n = 0;
        while n < N {
            result[n] = MaybeUninit::new(self.0[n].clone() * other.clone());
            n += 1;
        }
        // SAFETY: All elements are initialized by the loop above
        let result: [T; N] = unsafe { core::ptr::read(result.as_ptr() as *const [T; N]) };
        Self(result)
    }
}

impl<T: core::ops::Mul<Output = T> + Clone, const N: usize> core::ops::Mul<T> for Vector<T, N> {
    type Output = Self;
    #[inline]
    fn mul(self, other: T) -> Self {
        self.mul_scalar(other)
    }
}

impl<T: core::ops::Div<Output = T> + Clone, const N: usize> Vector<T, N> {
    /// Divides each element of the vector by a scalar value.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([6.0, 9.0, 12.0]);
    /// let result = v.div_scalar(3.0);
    /// assert_eq!(result.into_inner(), [2.0, 3.0, 4.0]);
    /// ```
    #[inline]
    pub fn div_scalar(self, other: T) -> Self {
        Self(self.0.map(|x| x / other.clone()))
    }
}

impl<T: core::ops::Div<Output = T> + Clone, const N: usize> core::ops::Div<T> for Vector<T, N> {
    type Output = Self;
    #[inline]
    fn div(self, other: T) -> Self {
        self.div_scalar(other)
    }
}

impl<T: core::ops::AddAssign + Clone, const N: usize> Vector<T, N> {
    /// Adds a scalar value to each element in place.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let mut v = Vector::new([1, 2, 3]);
    /// v.add_assign_scalar(5);
    /// assert_eq!(v.into_inner(), [6, 7, 8]);
    /// ```
    #[inline]
    pub fn add_assign_scalar(&mut self, other: T) {
        for i in 0..N {
            self.0[i] += other.clone();
        }
    }
}

impl<T: core::ops::AddAssign + Clone, const N: usize> core::ops::AddAssign<T> for Vector<T, N> {
    #[inline]
    fn add_assign(&mut self, other: T) {
        self.add_assign_scalar(other)
    }
}

impl<T: core::ops::SubAssign + Clone, const N: usize> Vector<T, N> {
    #[inline]
    pub fn sub_assign_scalar(&mut self, other: T) {
        for i in 0..N {
            self.0[i] -= other.clone();
        }
    }
}

impl<T: core::ops::SubAssign + Clone, const N: usize> core::ops::SubAssign<T> for Vector<T, N> {
    #[inline]
    fn sub_assign(&mut self, other: T) {
        self.sub_assign_scalar(other)
    }
}

impl<T: core::ops::MulAssign + Clone, const N: usize> Vector<T, N> {
    /// Multiplies each element by a scalar value in place.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let mut v = Vector::new([2, 4, 6]);
    /// v.mul_assign_scalar(3);
    /// assert_eq!(v.into_inner(), [6, 12, 18]);
    /// ```
    #[inline]
    pub fn mul_assign_scalar(&mut self, other: T) {
        for i in 0..N {
            self.0[i] *= other.clone();
        }
    }
}

impl<T: core::ops::MulAssign + Clone, const N: usize> core::ops::MulAssign<T> for Vector<T, N> {
    #[inline]
    fn mul_assign(&mut self, other: T) {
        self.mul_assign_scalar(other)
    }
}

impl<T: core::ops::DivAssign + Clone, const N: usize> Vector<T, N> {
    #[inline]
    pub fn div_assign_scalar(&mut self, other: T) {
        for i in 0..N {
            self.0[i] /= other.clone();
        }
    }
}

impl<T: core::ops::DivAssign + Clone, const N: usize> core::ops::DivAssign<T> for Vector<T, N> {
    #[inline]
    fn div_assign(&mut self, other: T) {
        self.div_assign_scalar(other)
    }
}

//length squared
impl<T: core::ops::Mul<Output = T> + Clone + core::ops::Add<Output = T>, const N: usize>
    Vector<T, N>
{
    /// Calculates the squared length (magnitude squared) of the vector.
    ///
    /// This is more efficient than `length()` when you only need to compare
    /// magnitudes, as it avoids the square root calculation.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([3.0, 4.0]);
    /// assert_eq!(v.length_squared(), 25.0);
    /// ```
    #[inline]
    pub fn length_squared(self) -> T {
        let mut result = self.0[0].clone() * self.0[0].clone();
        for i in 1..N {
            result = result + self.0[i].clone() * self.0[i].clone();
        }
        result
    }
}

impl<T: core::ops::Mul<Output = T> + Clone + core::ops::Add<Output = T> + Float, const N: usize>
    Vector<T, N>
{
    /// Calculates the length (magnitude) of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([3.0, 4.0]);
    /// assert_eq!(v.length(), 5.0);
    /// ```
    #[inline]
    #[cfg(feature = "std")]
    pub fn length(self) -> T {
        self.length_squared().sqrt()
    }
}

//euclid_distance_to

impl<
    T: core::ops::Sub<Output = T>
        + Clone
        + Copy
        + core::ops::Mul<Output = T>
        + core::ops::Add<Output = T>
        + Float,
    const N: usize,
> Vector<T, N>
{
    /// Calculates the Euclidean distance between two vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v1 = Vector::new([1.0f32, 2.0, 3.0]);
    /// let v2 = Vector::new([4.0f32, 6.0, 8.0]);
    /// let distance = v1.euclid_distance_to(v2);
    /// assert!((distance - 7.0710678f32).abs() < 0.0001);
    /// ```
    #[inline]
    #[cfg(feature = "std")]
    pub fn euclid_distance_to(self, other: Self) -> T {
        let mut result = self.0[0] - other.0[0];
        result = result * result;
        for i in 1..N {
            let diff = self.0[i] - other.0[i];
            result = result + diff * diff;
        }
        result.sqrt()
    }
}

//norm
impl<
    T: core::ops::Div<Output = T>
        + Clone
        + core::ops::Mul<Output = T>
        + core::ops::Add<Output = T>
        + Float,
    const N: usize,
> Vector<T, N>
{
    /// Normalizes the vector to unit length.
    ///
    /// Returns a `NormalizedVector` which is guaranteed to have length 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([3.0f64, 4.0]);
    /// let normalized = v.normalize();
    /// let length = normalized.as_vector().length();
    /// assert!((length - 1.0f64).abs() < 0.0001);
    /// ```
    #[inline]
    #[cfg(feature = "std")]
    pub fn normalize(self) -> NormalizedVector<T, N> {
        NormalizedVector(self.clone() / self.length())
    }
}

impl<T, const N: usize> Vector<MaybeUninit<T>, N>
where
    T: Sized,
{
    /// Converts a vector of `MaybeUninit<T>` to `Vector<T>`.
    ///
    /// # Safety
    ///
    /// The caller must ensure all elements have been properly initialized
    /// before calling this method.
    #[inline]
    pub unsafe fn assume_init(self) -> Vector<T, N> {
        let inner = self.0;
        //transmute to T
        //This is faster than using `MaybeUninit::assume_init` on each element
        // SAFETY: All elements of `columns` have been initialized.
        let arr = unsafe { core::ptr::read(inner.as_ptr() as *const [T; N]) };
        Vector(arr)
    }
}

//cross-product
impl<
    T: core::ops::Sub<Output = T>
        + Clone
        + Copy
        + core::ops::Mul<Output = T>
        + core::ops::Add<Output = T>,
> Vector<T, 3>
{
    /// Computes the cross product of two 3D vectors.
    ///
    /// The cross product produces a vector perpendicular to both input vectors.
    /// This operation is only defined for 3D vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v1 = Vector::new([1, 0, 0]);
    /// let v2 = Vector::new([0, 1, 0]);
    /// let cross = v1.cross(v2);
    /// assert_eq!(cross.into_inner(), [0, 0, 1]);
    /// ```
    #[inline]
    pub fn cross(self, other: Self) -> Self {
        let x = *self.y() * *other.z() - *self.z() * *other.y();
        let y = *self.z() * *other.x() - *self.x() * *other.z();
        let z = *self.x() * *other.y() - *self.y() * *other.x();
        Vector::new([x, y, z])
    }
}

impl<T, const N: usize> Vector<MaybeUninit<T>, N> {
    /// A vector of uninitialized values.
    ///
    /// This is useful for creating temporary storage that will be initialized later.
    /// The vector must be initialized before use.
    ///
    /// # Safety
    ///
    /// The vector contains uninitialized memory and must be properly initialized
    /// before being converted to a regular `Vector<T, N>`.
    pub const UNINIT: Self = Self([const { MaybeUninit::uninit() }; N]);
}

//map
impl<T, const N: usize> Vector<T, N>
where
    T: Clone,
{
    /// Applies a function to each element of the vector, producing a new vector.
    ///
    /// # Arguments
    ///
    /// * `f` - The function to apply to each element
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([1, 2, 3, 4]);
    /// let doubled = v.map(|x| x * 2);
    /// assert_eq!(doubled.into_inner(), [2, 4, 6, 8]);
    ///
    /// // Convert types
    /// let floats = v.map(|x| x as f32);
    /// assert_eq!(floats.into_inner(), [1.0, 2.0, 3.0, 4.0]);
    /// ```
    #[inline]
    pub fn map<F, U>(self, mut f: F) -> Vector<U, N>
    where
        F: FnMut(T) -> U,
    {
        let mut result = Vector::UNINIT;

        for i in 0..N {
            result.0[i] = MaybeUninit::new(f(self.0[i].clone()));
        }
        unsafe { result.assume_init() }
    }
}

//map_in_place
impl<T, const N: usize> Vector<T, N> {
    /// Applies a function to each element of the vector in place.
    ///
    /// # Arguments
    ///
    /// * `f` - The function to apply to each element
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let mut v = Vector::new([1, 2, 3, 4]);
    /// v.map_in_place(|x| *x *= 2);
    /// assert_eq!(v.into_inner(), [2, 4, 6, 8]);
    /// ```
    #[inline]
    pub fn map_in_place<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T),
    {
        for i in 0..N {
            f(&mut self.0[i]);
        }
    }
}

//eq_approx
impl<T: Float + Clone, const N: usize> Vector<T, N> {
    /// Checks if two vectors are approximately equal within a tolerance.
    ///
    /// This is useful for floating-point comparisons where exact equality
    /// may not be reliable due to rounding errors.
    ///
    /// # Arguments
    ///
    /// * `other` - The vector to compare with
    /// * `tolerance` - The maximum allowed difference between elements
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v1 = Vector::new([1.0, 2.0, 3.0]);
    /// let v2 = Vector::new([1.001, 2.001, 3.001]);
    /// assert!(v1.eq_approx(v2, 0.01));
    /// assert!(!v1.eq_approx(v2, 0.0001));
    /// ```
    #[inline]
    pub fn eq_approx(self, other: Self, tolerance: T) -> bool {
        for i in 0..N {
            if !self.0[i]
                .clone()
                .eq_approx(other.0[i].clone(), tolerance.clone())
            {
                return false;
            }
        }
        true
    }
}

//min, max
impl<T: core::cmp::PartialOrd + Clone, const N: usize> Vector<T, N> {
    /// Returns the maximum element in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([3, 1, 4, 1, 5]);
    /// assert_eq!(v.max(), 5);
    /// ```
    #[inline]
    pub fn max(self) -> T {
        let mut result = self.0[0].clone();
        for i in 1..N {
            if self.0[i].clone() > result {
                result = self.0[i].clone();
            }
        }
        result
    }

    /// Returns the minimum element in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([3, 1, 4, 1, 5]);
    /// assert_eq!(v.min(), 1);
    /// ```
    #[inline]
    pub fn min(self) -> T {
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
impl<T: core::ops::Mul<Output = T> + Clone + core::ops::Add<Output = T>, const N: usize>
    Vector<T, N>
{
    /// Calculates the dot product (scalar product) of two vectors.
    ///
    /// The dot product is the sum of the products of corresponding elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v1 = Vector::new([1, 2, 3]);
    /// let v2 = Vector::new([4, 5, 6]);
    /// assert_eq!(v1.dot(v2), 32); // 1*4 + 2*5 + 3*6
    /// ```
    #[inline]
    pub fn dot(self, other: Self) -> T {
        let mut result = self.0[0].clone() * other.0[0].clone();
        for i in 1..N {
            result = result + self.0[i].clone() * other.0[i].clone();
        }
        result
    }
}

//clamp
impl<T: core::cmp::PartialOrd + Clone, const N: usize> Vector<T, N> {
    /// Clamps each element of the vector between minimum and maximum values.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum allowed value
    /// * `max` - The maximum allowed value
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([0, 5, 10, 15]);
    /// let clamped = v.clamp(2, 12);
    /// assert_eq!(clamped.into_inner(), [2, 5, 10, 12]);
    /// ```
    #[inline]
    pub fn clamp(self, min: T, max: T) -> Self {
        Self(self.0.map(|x| {
            if x < min {
                min.clone()
            } else if x > max {
                max.clone()
            } else {
                x
            }
        }))
    }
}

//mix
impl<
    T: Float
        + Constants
        + Clone
        + core::ops::Sub<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Add<Output = T>,
    const N: usize,
> Vector<T, N>
{
    /// Performs linear interpolation between two vectors.
    ///
    /// # Arguments
    ///
    /// * `other` - The target vector to interpolate towards
    /// * `weight` - The interpolation factor (0.0 returns self, 1.0 returns other)
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v1 = Vector::new([0.0, 0.0, 0.0]);
    /// let v2 = Vector::new([10.0, 10.0, 10.0]);
    /// let interpolated = v1.mix(v2, 0.5);
    /// assert_eq!(interpolated.into_inner(), [5.0, 5.0, 5.0]);
    /// ```
    #[inline]
    pub fn mix(self, other: Self, weight: T) -> Self {
        let mut result = self.0.clone();
        for (i, result_item) in result.iter_mut().enumerate().take(N) {
            *result_item =
                self.0[i].clone() * (T::ONE - weight.clone()) + other.0[i].clone() * weight.clone();
        }
        Self(result)
    }
}

//index
impl<T, const N: usize> core::ops::Index<usize> for Vector<T, N> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

//index_mut
impl<T, const N: usize> core::ops::IndexMut<usize> for Vector<T, N> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

//to_row, to_col
impl<T, const N: usize> Vector<T, N>
where
    T: Clone,
{
    /// Converts the vector to a row matrix (1×N matrix).
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([1, 2, 3]);
    /// let row_matrix = v.to_row();
    /// // Creates a 1×3 matrix
    /// ```
    #[inline]
    pub fn to_row(self) -> Matrix<T, 1, N> {
        Matrix::new_rows([self])
    }

    /// Creates a vector from a row matrix (1×N matrix).
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    /// use vectormatrix::matrix::Matrix;
    ///
    /// let row = Matrix::new_rows([Vector::new([1, 2, 3])]);
    /// let v = Vector::from_row(row);
    /// assert_eq!(v.into_inner(), [1, 2, 3]);
    /// ```
    #[inline]
    pub fn from_row(row: Matrix<T, 1, N>) -> Self {
        row.rows()[0].clone()
    }

    /// Creates a vector from a column matrix (N×1 matrix).
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    /// use vectormatrix::matrix::Matrix;
    ///
    /// let col = Matrix::new_columns([Vector::new([1, 2, 3])]);
    /// let v = Vector::from_col(col);
    /// assert_eq!(v.into_inner(), [1, 2, 3]);
    /// ```
    #[inline]
    pub fn from_col(col: Matrix<T, N, 1>) -> Self {
        col.columns()[0].clone()
    }

    /// Converts the vector to a column matrix (N×1 matrix).
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::vector::Vector;
    ///
    /// let v = Vector::new([1, 2, 3]);
    /// let col_matrix = v.to_col();
    /// // Creates a 3×1 matrix
    /// ```
    #[inline]
    pub fn to_col(self) -> Matrix<T, N, 1> {
        Matrix::new_columns([self])
    }
}

//boilerplate

impl<T, const N: usize> Default for Vector<T, N>
where
    T: Default + Copy,
{
    fn default() -> Self {
        Vector::new([T::default(); N])
    }
}

//boilerplate: tuple constructors
impl<T> From<T> for Vector<T, 1> {
    fn from(value: T) -> Self {
        Vector::new([value])
    }
}

impl<T> From<(T,)> for Vector<T, 1> {
    fn from(value: (T,)) -> Self {
        Vector::new([value.0])
    }
}

impl<T> From<(T, T)> for Vector<T, 2> {
    fn from(value: (T, T)) -> Self {
        Vector::new([value.0, value.1])
    }
}

impl<T> From<(T, T, T)> for Vector<T, 3> {
    fn from(value: (T, T, T)) -> Self {
        Vector::new([value.0, value.1, value.2])
    }
}

impl<T> From<(T, T, T, T)> for Vector<T, 4> {
    fn from(value: (T, T, T, T)) -> Self {
        Vector::new([value.0, value.1, value.2, value.3])
    }
}

impl<T> From<Vector<T, 1>> for (T,) {
    fn from(value: Vector<T, 1>) -> Self {
        let a = value.0;
        let mut iter = a.into_iter();
        (iter.next().unwrap(),)
    }
}

impl<T> From<Vector<T, 2>> for (T, T) {
    fn from(value: Vector<T, 2>) -> Self {
        let a = value.0;
        let mut iter = a.into_iter();
        (iter.next().unwrap(), iter.next().unwrap())
    }
}

impl<T> From<Vector<T, 3>> for (T, T, T) {
    fn from(value: Vector<T, 3>) -> Self {
        let a = value.0;
        let mut iter = a.into_iter();
        (
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
        )
    }
}

impl<T> From<Vector<T, 4>> for (T, T, T, T) {
    fn from(value: Vector<T, 4>) -> Self {
        let a = value.0;
        let mut iter = a.into_iter();
        (
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
        )
    }
}

impl<T, const N: usize> From<Vector<T, N>> for [T; N] {
    fn from(value: Vector<T, N>) -> Self {
        value.0
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(value: [T; N]) -> Self {
        Vector::new(value)
    }
}

//asref/asmut

impl<T, const N: usize> core::convert::AsRef<[T; N]> for Vector<T, N> {
    fn as_ref(&self) -> &[T; N] {
        &self.0
    }
}

impl<T, const N: usize> core::convert::AsMut<[T; N]> for Vector<T, N> {
    fn as_mut(&mut self) -> &mut [T; N] {
        &mut self.0
    }
}

impl<T, const N: usize, const M: usize> core::ops::Mul<Matrix<T, M, N>> for Vector<T, M>
where
    T: Clone + core::ops::Mul<Output = T> + core::ops::Add<Output = T>,
{
    type Output = Matrix<T, 1, N>;

    fn mul(self, rhs: Matrix<T, M, N>) -> Self::Output {
        self.to_row().mul_matrix(rhs)
    }
}

//NormalizedVector boilerplate

// From/Into implementations
#[cfg(feature = "std")]
impl<T, const N: usize> From<Vector<T, N>> for NormalizedVector<T, N>
where
    T: core::ops::Add<Output = T>
        + Clone
        + core::ops::Div<Output = T>
        + core::ops::Mul<Output = T>
        + Float,
{
    ///Converts to NormalizedVector by normalizing the vector.
    fn from(value: Vector<T, N>) -> Self {
        value.normalize()
    }
}

impl<T, const N: usize> From<NormalizedVector<T, N>> for Vector<T, N> {
    ///Converts from NormalizedVector by returning the underlying vector.
    fn from(value: NormalizedVector<T, N>) -> Self {
        value.into_vector()
    }
}

// AsRef implementation (no AsMut since this would break normalization)
impl<T, const N: usize> AsRef<Vector<T, N>> for NormalizedVector<T, N> {
    ///Returns a reference to the underlying vector.
    fn as_ref(&self) -> &Vector<T, N> {
        self.as_vector()
    }
}

// Deref for convenient access to Vector methods (no DerefMut to preserve normalization)
impl<T, const N: usize> core::ops::Deref for NormalizedVector<T, N> {
    type Target = Vector<T, N>;

    fn deref(&self) -> &Self::Target {
        self.as_vector()
    }
}

#[cfg(test)]
mod tests {
    use crate::vector::Vector;

    #[test]
    fn elementwise() {
        let a = Vector::new([1, 2, 3]);
        let b = Vector::new([4, 5, 6]);
        let c = a + b;
        assert_eq!(c.0, [5, 7, 9]);

        let d = a - b;
        assert_eq!(d.0, [-3, -3, -3]);

        let e = a * b;
        assert_eq!(e.0, [4, 10, 18]);

        let f = a / b;
        assert_eq!(f.0, [0, 0, 0]);
    }

    #[test]
    fn elementwise_assign() {
        let mut a = Vector::new([1, 2, 3]);
        let b = Vector::new([4, 5, 6]);
        a += b;
        assert_eq!(a.0, [5, 7, 9]);

        let mut c = Vector::new([1, 2, 3]);
        let d = Vector::new([4, 5, 6]);
        c -= d;
        assert_eq!(c.0, [-3, -3, -3]);

        let mut e = Vector::new([1, 2, 3]);
        let f = Vector::new([4, 5, 6]);
        e *= f;
        assert_eq!(e.0, [4, 10, 18]);

        let mut g = Vector::new([1, 2, 3]);
        let h = Vector::new([4, 5, 6]);
        g /= h;
        assert_eq!(g.0, [0, 0, 0]);
    }

    #[test]
    fn scalar() {
        let a = Vector::new([1, 2, 3]);
        let b = a + 1;
        assert_eq!(b.0, [2, 3, 4]);

        let c = a - 1;
        assert_eq!(c.0, [0, 1, 2]);

        let d = a * 2;
        assert_eq!(d.0, [2, 4, 6]);

        let e = a / 2;
        assert_eq!(e.0, [0, 1, 1]);
    }

    #[test]
    fn scalar_assign() {
        let mut a = Vector::new([1, 2, 3]);
        a += 1;
        assert_eq!(a.0, [2, 3, 4]);

        let mut b = Vector::new([1, 2, 3]);
        b -= 1;
        assert_eq!(b.0, [0, 1, 2]);

        let mut c = Vector::new([1, 2, 3]);
        c *= 2;
        assert_eq!(c.0, [2, 4, 6]);

        let mut d = Vector::new([1, 2, 3]);
        d /= 2;
        assert_eq!(d.0, [0, 1, 1]);
    }

    #[test]
    #[cfg(feature = "std")]
    fn length() {
        let a = Vector::new([3.0, 4.0]);
        assert_eq!(a.length(), 5.0);
    }

    #[test]
    #[cfg(feature = "std")]
    fn euclid_distance() {
        let a = Vector::new([1.0, 2.0]);
        let b = Vector::new([4.0, 6.0]);
        assert_eq!(a.euclid_distance_to(b), 5.0);
    }

    #[test]
    #[cfg(feature = "std")]
    fn normalize() {
        let a = Vector::new([3.0, 4.0]);
        let b = a.normalize();
        assert_eq!(b.as_vector().length(), 1.0);
    }

    #[test]
    fn map() {
        let a = Vector::new([1i32, -2, 3]);
        let b = a.map(|x| x.abs());
        assert_eq!(b.0, [1, 2, 3]);
    }

    #[test]
    fn min_max() {
        let a = Vector::new([1, 2, 3]);
        assert_eq!(a.min(), 1);
        assert_eq!(a.max(), 3);
    }

    #[test]
    fn eq_approx() {
        let a = Vector::new([1.0, 2.0, 3.0]);
        let b = Vector::new([1.0, 2.005, 3.0]);
        assert!(a.eq_approx(b, 0.01));
        assert!(!a.eq_approx(b, 0.001));
    }
    #[test]
    fn test_cross() {
        let a = Vector::new([1, 2, 3]);
        let b = Vector::new([4, 5, 6]);
        let c = a.cross(b);
        assert_eq!(c.0, [-3, 6, -3]);
    }

    #[test]
    fn test_dot() {
        let a = Vector::new([1, 2, 3]);
        let b = Vector::new([4, 5, 6]);
        let c = a.dot(b);
        assert_eq!(c, 32);
    }

    #[test]
    fn test_clamp() {
        let a = Vector::new([1, 2, 3, 4]);
        let b = a.clamp(2, 3);
        assert_eq!(b.0, [2, 2, 3, 3]);
    }

    #[test]
    fn test_mix() {
        let a = Vector::new([1.0f32, 2.0, 3.0, 4.0]);
        let b = a.mix(Vector::new([4.0f32, 3.0, 2.0, 1.0]), 0.5);
        assert_eq!(b.0, [2.5, 2.5, 2.5, 2.5]);
    }

    #[test]
    fn test_convert() {
        let a = Vector::new([1, 2, 3]);
        let _b: (i32, i32, i32) = a.into();
    }
}
