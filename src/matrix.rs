//SPDX-License-Identifier: MIT OR Apache-2.0
//! Mathematical matrix types for linear algebra operations.
//!
//! This module provides a generic `Matrix<T, R, C>` type that represents an R×C
//! matrix with elements of type `T`. Matrices are stored in column-major order
//! for efficient linear algebra operations.
//!
//! # Features
//!
//! - Generic over element type and dimensions (rows × columns)
//! - Stack-allocated with const generics
//! - Column-major storage with row accessors
//! - Comprehensive arithmetic operations
//! - Matrix multiplication
//! - Special operations for 2×2, 3×3, and 4×4 matrices (determinant, inverse, transformations)
//! - Type-safe dimension checking at compile time
//! - `no_std` compatible
//!
//! # Examples
//!
//! ```
//! use vectormatrix::matrix::Matrix;
//! use vectormatrix::vector::Vector;
//!
//! // Create a 2×3 matrix from rows
//! let m1 = Matrix::<f32, 2, 3>::new_rows([
//!     Vector::new([1.0, 2.0, 3.0]),
//!     Vector::new([4.0, 5.0, 6.0]),
//! ]);
//!
//! // Create a 3×2 matrix from columns  
//! let m2 = Matrix::<f32, 3, 2>::new_columns([
//!     Vector::new([7.0, 9.0, 11.0]),
//!     Vector::new([8.0, 10.0, 12.0]),
//! ]);
//!
//! // Matrix multiplication (2×3 × 3×2 = 2×2)
//! let result = m1 * m2;
//!
//! // Element-wise operations
//! let m3 = Matrix::<f32, 2, 2>::IDENTITY;
//! let scaled = m3 * 2.0;  // Scalar multiplication
//! let sum = result + scaled;  // Element-wise addition
//! ```
//!
//! # Storage Layout
//!
//! Matrices use column-major storage internally. This means that columns are 
//! contiguous in memory, which is optimal for many linear algebra operations.
//! Despite column-major storage, both row and column accessors are provided
//! for convenience.

mod m3x3;
mod m4x4;
mod m2x2;

use core::fmt::Debug;
use core::mem::MaybeUninit;
use crate::types::sealed::{Constants};
use crate::vector::Vector;

/// A mathematical matrix with `R` rows and `C` columns.
///
/// `Matrix<T, R, C>` represents a matrix in R×C dimensional space. It supports
/// standard matrix operations like addition, scalar multiplication, matrix
/// multiplication, and transposition.
///
/// # Type Parameters
///
/// - `T`: The type of each element in the matrix
/// - `R`: The number of rows (compile-time constant)
/// - `C`: The number of columns (compile-time constant)
///
/// # Storage
///
/// The matrix uses column-major storage internally, meaning columns are stored
/// contiguously. This layout is efficient for many linear algebra operations.
/// Despite the internal storage, the API provides both row and column access.
///
/// # Examples
///
/// ```
/// use vectormatrix::matrix::Matrix;
/// use vectormatrix::vector::Vector;
///
/// // Create a 2×2 matrix from columns
/// let m = Matrix::new_columns([
///     Vector::new([1.0, 3.0]),
///     Vector::new([2.0, 4.0]),
/// ]);
///
/// // Access elements
/// assert_eq!(*m.element_at(0, 0), 1.0);
/// assert_eq!(*m.element_at(1, 0), 3.0);
///
/// // Use identity matrix
/// let identity = Matrix::<f32, 3, 3>::IDENTITY;
/// assert_eq!(*identity.element_at(0, 0), 1.0);
/// assert_eq!(*identity.element_at(1, 1), 1.0);
/// assert_eq!(*identity.element_at(0, 1), 0.0);
/// ```
#[derive(Copy,Clone,PartialEq,Eq,PartialOrd, Ord, Hash)]
pub struct Matrix<T, const R: usize, const C: usize> {
    columns: [Vector<T,R>; C],
}

impl<T, const R: usize, const C: usize> Matrix<T,R,C> {

    /// An uninitialized matrix with uninitialized elements.
    ///
    /// This is useful for creating matrices that will be filled in later,
    /// avoiding unnecessary initialization overhead.
    ///
    /// # Safety
    ///
    /// The returned matrix contains uninitialized memory. You must initialize
    /// all elements before calling `assume_init()`.
    pub const UNINIT: Matrix<MaybeUninit<T>,R,C> = Matrix {
        columns: [Vector::UNINIT; C],
    };

    /// Creates a new matrix from the given columns.
    ///
    /// This is the most efficient constructor since the matrix uses column-major
    /// storage internally. The columns are directly stored without transformation.
    ///
    /// # Arguments
    ///
    /// * `columns` - An array of column vectors
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    /// use vectormatrix::vector::Vector;
    ///
    /// let m = Matrix::new_columns([
    ///     Vector::new([1.0, 4.0]),  // First column
    ///     Vector::new([2.0, 5.0]),  // Second column
    ///     Vector::new([3.0, 6.0]),  // Third column
    /// ]);
    ///
    /// // The resulting matrix is:
    /// // | 1.0  2.0  3.0 |
    /// // | 4.0  5.0  6.0 |
    /// ```
    #[inline] pub const fn new_columns(columns: [Vector<T, R>; C]) -> Self {
        Self {
            columns,
        }
    }

    /// Creates a new matrix from the given rows.
    ///
    /// Since matrices use column-major storage internally, this constructor
    /// transposes the input rows into columns. This makes it slightly less
    /// efficient than `new_columns`, but more intuitive for row-based initialization.
    ///
    /// # Arguments
    ///
    /// * `rows` - An array of row vectors
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    /// use vectormatrix::vector::Vector;
    ///
    /// let m = Matrix::new_rows([
    ///     Vector::new([1.0, 2.0, 3.0]),  // First row
    ///     Vector::new([4.0, 5.0, 6.0]),  // Second row
    /// ]);
    ///
    /// // The resulting matrix is:
    /// // | 1.0  2.0  3.0 |
    /// // | 4.0  5.0  6.0 |
    /// ```
    #[inline] pub fn new_rows(rows: [Vector<T, C>; R]) -> Self
    where
        T: Clone
    {
        Matrix::new_columns(rows).transpose()
    }

    /// Creates a new matrix from a flat array of elements in row-major order.
    ///
    /// Elements are read from the array in row-major order (row by row) and
    /// stored in the matrix's column-major format.
    ///
    /// # Arguments
    ///
    /// * `arr` - A flat array containing R×C elements in row-major order
    ///
    /// # Panics
    ///
    /// Panics if the array length `L` does not equal `R * C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    ///
    /// // Create a 2×3 matrix from a flat array
    /// let m = Matrix::<f32, 2, 3>::new_from_array([
    ///     1.0, 2.0, 3.0,  // First row
    ///     4.0, 5.0, 6.0,  // Second row
    /// ]);
    ///
    /// // The resulting matrix is:
    /// // | 1.0  2.0  3.0 |
    /// // | 4.0  5.0  6.0 |
    ///
    /// assert_eq!(*m.element_at(0, 0), 1.0);
    /// assert_eq!(*m.element_at(1, 2), 6.0);
    /// ```
    #[inline] pub fn new_from_array<const L: usize>(arr: [T; L]) -> Self
    where
        T: Clone
    {
        assert_eq!(L, R * C);
        let mut columns = [const { MaybeUninit::uninit() }; C];
        for c in 0..C {
            let mut column = Vector::UNINIT;
            for r in 0..R {
                column[r] = MaybeUninit::new(arr[r * C + c].clone());
            }
            let column = unsafe { column.assume_init() };
            columns[c] = MaybeUninit::new(column);
        }
        let arr: [Vector<T, R>; C] = columns.map(|maybe| unsafe { maybe.assume_init() });
        Self {
            columns: arr,
        }
    }

    /// Returns the transpose of this matrix.
    ///
    /// The transpose of an R×C matrix is a C×R matrix where the rows and columns
    /// are swapped. That is, element (i,j) in the original matrix becomes element
    /// (j,i) in the transposed matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    /// use vectormatrix::vector::Vector;
    ///
    /// let m = Matrix::new_rows([
    ///     Vector::new([1.0, 2.0, 3.0]),
    ///     Vector::new([4.0, 5.0, 6.0]),
    /// ]);
    ///
    /// let transposed = m.transpose();
    ///
    /// // Original: 2×3 matrix
    /// // | 1.0  2.0  3.0 |
    /// // | 4.0  5.0  6.0 |
    ///
    /// // Transposed: 3×2 matrix  
    /// // | 1.0  4.0 |
    /// // | 2.0  5.0 |
    /// // | 3.0  6.0 |
    ///
    /// assert_eq!(*transposed.element_at(0, 0), 1.0);
    /// assert_eq!(*transposed.element_at(1, 0), 2.0);
    /// assert_eq!(*transposed.element_at(2, 1), 6.0);
    /// ```
    pub fn transpose(self) -> Matrix<T, C, R>
    where
        T: Clone
    {
        let mut columns = [const { MaybeUninit::uninit() }; R];
        for r in 0..R {
            let mut column = Vector::UNINIT;
            for c in 0..C {
                column[c] = MaybeUninit::new(self.columns[c][r].clone());
            }
            let column = unsafe { column.assume_init() };
            columns[r] = MaybeUninit::new(column);
        }
        let arr: [Vector<T, C>; R] = columns.map(|maybe| unsafe { maybe.assume_init() });
        Matrix {
            columns: arr,
        }
    }


}

//assume_init
impl<T, const R: usize, const C: usize> Matrix<MaybeUninit<T>,R,C> {
    /// Assumes that all elements in the matrix have been initialized.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all `R * C` elements in the matrix have been
    /// properly initialized before calling this method. Calling this on a matrix
    /// with uninitialized elements results in undefined behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    /// use vectormatrix::vector::Vector;
    /// use core::mem::MaybeUninit;
    ///
    /// // Create an uninitialized matrix
    /// let mut m: Matrix<MaybeUninit<f32>, 2, 2> = Matrix::UNINIT;
    ///
    /// // Initialize all elements
    /// for c in 0..2 {
    ///     for r in 0..2 {
    ///         m.columns_mut()[c][r] = MaybeUninit::new((r * 2 + c) as f32);
    ///     }
    /// }
    ///
    /// // Now safe to assume initialization
    /// let initialized = unsafe { m.assume_init() };
    /// assert_eq!(*initialized.element_at(0, 0), 0.0);
    /// assert_eq!(*initialized.element_at(1, 1), 3.0);
    /// ```
    pub unsafe fn assume_init(self) -> Matrix<T,R,C> {
        let columns = self.columns.map(|maybe| unsafe { maybe.assume_init() });
        Matrix {
            columns
        }
    }
}

//elementwise add
impl<T: Constants + Clone + core::ops::Add<Output=T> , const R: usize, const C: usize> Matrix<T,R,C> {
    /// Performs element-wise addition with another matrix.
    ///
    /// Each element in the result is the sum of the corresponding elements
    /// from the two input matrices.
    ///
    /// # Arguments
    ///
    /// * `other` - The matrix to add
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    ///
    /// let a = Matrix::<f32, 2, 2>::new_from_array([
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    /// ]);
    ///
    /// let b = Matrix::<f32, 2, 2>::new_from_array([
    ///     5.0, 6.0,
    ///     7.0, 8.0,
    /// ]);
    ///
    /// let result = a.elementwise_add(b);
    /// // Result:
    /// // | 6.0   8.0 |
    /// // | 10.0  12.0 |
    ///
    /// assert_eq!(*result.element_at(0, 0), 6.0);
    /// assert_eq!(*result.element_at(1, 1), 12.0);
    /// ```
    #[inline] pub fn elementwise_add(self, other: Self) -> Self {
        let mut columns = [const { MaybeUninit::uninit() }; C];
        for c in 0..C {
            columns[c] = MaybeUninit::new(self.columns[c].clone().elementwise_add(other.columns[c].clone()));
        }
        let c = unsafe { columns.map(|c| c.assume_init()) };
        Self {
            columns: c
        }
    }
}

impl<T: Constants + Clone + core::ops::Add<Output=T>, const R: usize, const C: usize> core::ops::Add for Matrix<T,R,C> {
    type Output = Self;
    #[inline] fn add(self, other: Self) -> Self {
        self.elementwise_add(other)
    }
}

impl<T: Constants + Clone + core::ops::Sub<Output=T> , const R: usize, const C: usize> Matrix<T,R,C> {
    /// Performs element-wise subtraction with another matrix.
    ///
    /// Each element in the result is the difference of the corresponding elements
    /// from the two input matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    ///
    /// let a = Matrix::<f32, 2, 2>::new_from_array([
    ///     10.0, 20.0,
    ///     30.0, 40.0,
    /// ]);
    ///
    /// let b = Matrix::<f32, 2, 2>::new_from_array([
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    /// ]);
    ///
    /// let result = a.elementwise_sub(b);
    /// assert_eq!(*result.element_at(0, 0), 9.0);
    /// assert_eq!(*result.element_at(1, 1), 36.0);
    /// ```
    #[inline] pub fn elementwise_sub(self, other: Self) -> Self {
        let mut columns = [const { MaybeUninit::uninit() }; C];
        for c in 0..C {
            columns[c] = MaybeUninit::new(self.columns[c].clone().elementwise_sub(other.columns[c].clone()));
        }
        let c = unsafe { columns.map(|c| c.assume_init()) };
        Self {
            columns: c
        }
    }
}

impl<T: Constants + Clone + core::ops::Sub<Output=T> , const R: usize, const C: usize> core::ops::Sub for  Matrix<T,R,C> {
    type Output = Self;
    #[inline] fn sub(self, other: Self) -> Self {
        self.elementwise_sub(other)
    }
}

impl<T: Constants + Clone + core::ops::Mul<Output=T>, const R: usize, const C: usize> Matrix<T,R,C> {
    ///Multiply all elements in the matrix by the corresponding elements in the other matrix.
    ///
    /// This has no Mul implementation -> in Rust that would conflict with matrix multiply.
    #[inline] pub fn elementwise_mul(self, other: Self) -> Self {
        let mut columns = [const { MaybeUninit::uninit() }; C];
        for c in 0..C {
            columns[c] = MaybeUninit::new(self.columns[c].clone().elementwise_mul(other.columns[c].clone()));
        }
        let c = unsafe { columns.map(|c| c.assume_init()) };
        Self {
            columns: c
        }
    }
}


impl <T: Constants + Clone + core::ops::Div<Output=T>, const R: usize, const C: usize> Matrix<T,R,C> {
    /**
    Divide all elements in the matrix by the corresponding elements in the other matrix.

    This has no Div implementation, there is not a neat mathematical definition of matrix division.
    */
    #[inline]
    pub fn elementwise_div(self, other: Self) -> Self {
        let mut columns = [const { MaybeUninit::uninit() }; C];
        for c in 0..C {
            columns[c] = MaybeUninit::new(self.columns[c].clone().elementwise_div(other.columns[c].clone()));
        }
        let c = unsafe { columns.map(|c| c.assume_init()) };
        Self {
            columns: c
        }
    }
}


//constants
impl <T: Constants, const R: usize, const C: usize> Matrix<T,R,C> {
    /// A matrix where all elements are zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    ///
    /// let zero: Matrix<f32, 2, 3> = Matrix::ZERO;
    ///
    /// // All elements are 0.0
    /// for r in 0..2 {
    ///     for c in 0..3 {
    ///         assert_eq!(*zero.element_at(r, c), 0.0);
    ///     }
    /// }
    /// ```
    pub const ZERO: Self = Self {
        columns: [Vector::ZERO; C],
    };
    
    /// A matrix where all elements are one.
    ///
    /// Note: This is different from an identity matrix. For identity matrices,
    /// use the `IDENTITY` constant available on square matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    ///
    /// let ones: Matrix<f32, 2, 3> = Matrix::ONE;
    ///
    /// // All elements are 1.0
    /// for r in 0..2 {
    ///     for c in 0..3 {
    ///         assert_eq!(*ones.element_at(r, c), 1.0);
    ///     }
    /// }
    /// ```
    pub const ONE: Self = Self {
        columns: [Vector::ONE; C],
    };
}

//identities
impl <T: Constants> Matrix<T,1,1> {
    pub const IDENTITY: Self = Self {
        columns: [Vector::ONE],
    };
}

impl <T: Constants> Matrix<T,1,2> {
    pub const IDENTITY: Self = Self {
        columns: [Vector::ONE, Vector::ZERO],
    };
}

impl<T: Constants> Matrix<T,1,3> {
    pub const IDENTITY: Self = Self {
        columns: [Vector::ONE, Vector::ZERO, Vector::ZERO],
    };
}

impl <T: Constants> Matrix<T,1,4> {
    pub const IDENTITY: Self = Self {
        columns: [Vector::ONE, Vector::ZERO, Vector::ZERO, Vector::ZERO],
    };
}

impl<T: Constants> Matrix<T,2,1> {
    pub const IDENTITY: Self = Self {
        columns: [Vector::new([T::ONE, T::ZERO])],
    };
}

impl<T: Constants> Matrix<T,2,2> {
    pub const IDENTITY: Self = Self {
        columns: [Vector::new([T::ONE, T::ZERO]), Vector::new([T::ZERO, T::ONE])],
    };
}

impl<T: Constants> Matrix<T,2,3> {
    pub const IDENTITY: Self = Self {
        columns: [Vector::new([T::ONE, T::ZERO]), Vector::new([T::ZERO, T::ONE]), Vector::new([T::ZERO, T::ZERO])],
    };
}

impl<T: Constants> Matrix<T,2,4> {
    pub const IDENTITY: Self = Self {
        columns: [Vector::new([T::ONE, T::ZERO]), Vector::new([T::ZERO, T::ONE]), Vector::new([T::ZERO, T::ZERO]), Vector::new([T::ZERO, T::ZERO])],
    };
}

impl <T: Constants> Matrix<T,3,1> {
    pub const IDENTITY: Self = Self {
        columns: [Vector::new([T::ONE, T::ZERO, T::ZERO])],
    };
}

impl <T: Constants> Matrix<T,3,2> {
    pub const IDENTITY: Self = Self {
        columns: [Vector::new([T::ONE, T::ZERO, T::ZERO]), Vector::new([T::ZERO, T::ONE, T::ZERO])],
    };
}

impl <T: Constants> Matrix<T,3,3> {
    pub const IDENTITY: Self = Self {
        columns: [Vector::new([T::ONE, T::ZERO, T::ZERO]), Vector::new([T::ZERO, T::ONE, T::ZERO]), Vector::new([T::ZERO, T::ZERO, T::ONE])],
    };
}

impl <T: Constants> Matrix<T,3,4> {
    pub const IDENTITY: Self = Self {
        columns: [Vector::new([T::ONE, T::ZERO, T::ZERO]), Vector::new([T::ZERO, T::ONE, T::ZERO]), Vector::new([T::ZERO, T::ZERO, T::ONE]), Vector::new([T::ZERO, T::ZERO, T::ZERO])],
    };
}

impl<T: Constants> Matrix<T,4,1> {
    pub const IDENTITY: Self = Self {
        columns: [Vector::new([T::ONE, T::ZERO, T::ZERO, T::ZERO])],
    };
}

impl<T: Constants> Matrix<T,4,2> {
    pub const IDENTITY: Self = Self {
        columns: [Vector::new([T::ONE, T::ZERO, T::ZERO, T::ZERO]), Vector::new([T::ZERO, T::ONE, T::ZERO, T::ZERO])],
    };
}

impl<T: Constants> Matrix<T,4,3> {
    pub const IDENTITY: Self = Self {
        columns: [Vector::new([T::ONE, T::ZERO, T::ZERO, T::ZERO]), Vector::new([T::ZERO, T::ONE, T::ZERO, T::ZERO]), Vector::new([T::ZERO, T::ZERO, T::ONE, T::ZERO])],
    };
}

impl<T: Constants> Matrix<T,4,4> {
    pub const IDENTITY: Self = Self {
        columns: [Vector::new([T::ONE, T::ZERO, T::ZERO, T::ZERO]), Vector::new([T::ZERO, T::ONE, T::ZERO, T::ZERO]), Vector::new([T::ZERO, T::ZERO, T::ONE, T::ZERO]), Vector::new([T::ZERO, T::ZERO, T::ZERO, T::ONE])],
    };
}

//AddAssign, etc.
impl <T: Clone + core::ops::AddAssign, const R: usize, const C: usize> Matrix<T,R,C> {
    #[inline] pub fn add_elementwise_assign(&mut self, other: Self) {
        for c in 0..C {
            self.columns[c] += other.columns[c].clone();
        }
    }
}

impl <T: Clone + core::ops::AddAssign, const R: usize, const C: usize> core::ops::AddAssign for Matrix<T,R,C> {
    #[inline] fn add_assign(&mut self, other: Self) {
        self.add_elementwise_assign(other);
    }
}

//SubAssign
impl <T: Clone + core::ops::SubAssign, const R: usize, const C: usize> Matrix<T,R,C> {
    #[inline] pub fn sub_elementwise_assign(&mut self, other: Self) {
        for c in 0..C {
            self.columns[c] -= other.columns[c].clone();
        }
    }
}
impl <T: Clone + core::ops::SubAssign, const R: usize, const C: usize> core::ops::SubAssign for Matrix<T,R,C> {
    #[inline] fn sub_assign(&mut self, other: Self) {
        self.sub_elementwise_assign(other);
    }
}

//MulAssign
impl <T: Clone + core::ops::MulAssign, const R: usize, const C: usize> Matrix<T,R,C> {
    #[inline] pub fn mul_elementwise_assign(&mut self, other: Self) {
        for c in 0..C {
            self.columns[c] *= other.columns[c].clone();
        }
    }
}

impl <T: Clone + core::ops::MulAssign, const R: usize, const C: usize> core::ops::MulAssign for Matrix<T,R,C> {
    #[inline] fn mul_assign(&mut self, other: Self) {
        self.mul_elementwise_assign(other);
    }
}

//DivAssign

impl <T: Clone + core::ops::DivAssign, const R: usize, const C: usize> Matrix<T,R,C> {
    #[inline] pub fn div_elementwise_assign(&mut self, other: Self) {
        for c in 0..C {
            self.columns[c] /= other.columns[c].clone();
        }
    }
}

impl <T: Clone + core::ops::DivAssign, const R: usize, const C: usize> core::ops::DivAssign for Matrix<T,R,C> {
    #[inline] fn div_assign(&mut self, other: Self) {
        self.div_elementwise_assign(other);
    }
}

//scalar ops

impl<T: Clone + core::ops::Add<Output=T>, const R: usize, const C: usize> Matrix<T,R,C> {
    /// Adds a scalar value to each element in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    ///
    /// let m = Matrix::<f32, 2, 2>::new_from_array([
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    /// ]);
    ///
    /// let result = m.add_scalar(10.0);
    ///
    /// assert_eq!(*result.element_at(0, 0), 11.0);
    /// assert_eq!(*result.element_at(1, 1), 14.0);
    /// ```
    #[inline] pub fn add_scalar(self, other: T) -> Self {
        let mut columns = [const { MaybeUninit::uninit() }; C];
        for c in 0..C {
            columns[c] = MaybeUninit::new(self.columns[c].clone().add_scalar(other.clone()));
        }
        let c = unsafe { columns.map(|c| c.assume_init()) };
        Self {
            columns: c
        }
    }
}

impl<T: Clone + core::ops::Sub<Output=T>, const R: usize, const C: usize> Matrix<T,R,C> {
    /// Subtracts a scalar value from each element in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    ///
    /// let m = Matrix::<f32, 2, 2>::new_from_array([
    ///     10.0, 20.0,
    ///     30.0, 40.0,
    /// ]);
    ///
    /// let result = m.sub_scalar(5.0);
    ///
    /// assert_eq!(*result.element_at(0, 0), 5.0);
    /// assert_eq!(*result.element_at(1, 1), 35.0);
    /// ```
    #[inline] pub fn sub_scalar(self, other: T) -> Self {
        let mut columns = [const { MaybeUninit::uninit() }; C];
        for c in 0..C {
            columns[c] = MaybeUninit::new(self.columns[c].clone().sub_scalar(other.clone()));
        }
        let c = unsafe { columns.map(|c| c.assume_init()) };
        Self {
            columns: c
        }
    }
}

impl<T: Clone + core::ops::Mul<Output=T>, const R: usize, const C: usize> Matrix<T,R,C> {
    /// Multiplies each element in the matrix by a scalar value.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    ///
    /// let m = Matrix::<f32, 2, 2>::new_from_array([
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    /// ]);
    ///
    /// let result = m.mul_scalar(3.0);
    ///
    /// assert_eq!(*result.element_at(0, 0), 3.0);
    /// assert_eq!(*result.element_at(1, 1), 12.0);
    /// ```
    #[inline] pub fn mul_scalar(self, other: T) -> Self {
        let mut columns = [const { MaybeUninit::uninit() }; C];
        for c in 0..C {
            columns[c] = MaybeUninit::new(self.columns[c].clone().mul_scalar(other.clone()));
        }
        let c = unsafe { columns.map(|c| c.assume_init()) };
        Self {
            columns: c
        }
    }
}

impl<T: Clone + core::ops::Div<Output=T>, const R: usize, const C: usize> Matrix<T,R,C> {
    /// Divides each element in the matrix by a scalar value.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    ///
    /// let m = Matrix::<f32, 2, 2>::new_from_array([
    ///     10.0, 20.0,
    ///     30.0, 40.0,
    /// ]);
    ///
    /// let result = m.div_scalar(10.0);
    ///
    /// assert_eq!(*result.element_at(0, 0), 1.0);
    /// assert_eq!(*result.element_at(1, 1), 4.0);
    /// ```
    #[inline] pub fn div_scalar(self, other: T) -> Self {
        let mut columns = [const { MaybeUninit::uninit() }; C];
        for c in 0..C {
            columns[c] = MaybeUninit::new(self.columns[c].clone().div_scalar(other.clone()));
        }
        let c = unsafe { columns.map(|c| c.assume_init()) };
        Self {
            columns: c
        }
    }
}

//trait implementations
impl <T, const R: usize, const C: usize> core::ops::Add<T> for Matrix<T,R,C>
where
    T: core::ops::Add<Output=T> + Clone,
{
    type Output = Self;
    #[inline] fn add(self, other: T) -> Self {
        self.add_scalar(other)
    }
}

impl <T, const R: usize, const C: usize> core::ops::Sub<T> for Matrix<T,R,C>
where
    T: core::ops::Sub<Output=T> + Clone,
{
    type Output = Self;
    #[inline] fn sub(self, other: T) -> Self {
        self.sub_scalar(other)
    }
}

impl <T, const R: usize, const C: usize> core::ops::Mul<T> for Matrix<T,R,C>
where
    T: core::ops::Mul<Output=T> + Clone,
{
    type Output = Self;
    #[inline] fn mul(self, other: T) -> Self {
        self.mul_scalar(other)
    }
}

impl <T, const R: usize, const C: usize> core::ops::Div<T> for Matrix<T,R,C>
where
    T: core::ops::Div<Output=T> + Clone,
{
    type Output = Self;
    #[inline] fn div(self, other: T) -> Self {
        self.div_scalar(other)
    }
}

//addassign, scalar
impl <T, const R: usize, const C: usize> Matrix<T,R,C>
where
    T: core::ops::AddAssign + Clone,
{
    #[inline] fn add_scalar_assign(&mut self, other: T) {
        for c in 0..C {
            self.columns[c] += other.clone();
        }
    }
}

impl <T, const R: usize, const C: usize> core::ops::AddAssign<T> for Matrix<T,R,C>
where
    T: core::ops::AddAssign + Clone,
{
    #[inline] fn add_assign(&mut self, other: T) {
        self.add_scalar_assign(other);
    }
}

//subassign, scalar

impl <T, const R: usize, const C: usize> Matrix<T,R,C>
where
    T: core::ops::SubAssign + Clone,
{
    #[inline] pub fn sub_scalar_assign(&mut self, other: T) {
        for c in 0..C {
            self.columns[c] -= other.clone();
        }
    }
}

impl <T, const R: usize, const C: usize> core::ops::SubAssign<T> for Matrix<T,R,C>
where
    T: core::ops::SubAssign + Clone,
{
    fn sub_assign(&mut self, other: T) {
        self.sub_scalar_assign(other);
    }
}

//map
impl<T: Clone, const R: usize, const C: usize> Matrix<T,R,C> {
    /// Applies a function to each element of the matrix, returning a new matrix.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that transforms each element
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    ///
    /// let m = Matrix::<f32, 2, 2>::new_from_array([
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    /// ]);
    ///
    /// // Square each element
    /// let squared = m.map(|x| x * x);
    ///
    /// assert_eq!(*squared.element_at(0, 0), 1.0);
    /// assert_eq!(*squared.element_at(0, 1), 4.0);
    /// assert_eq!(*squared.element_at(1, 0), 9.0);
    /// assert_eq!(*squared.element_at(1, 1), 16.0);
    /// ```
    pub fn map<F: FnMut(T) -> T>(self, mut f: F) -> Self {
        let mut columns = [const { MaybeUninit::uninit() }; C];
        for c in 0..C {
            columns[c] = MaybeUninit::new(self.columns[c].clone().map(&mut f));
        }
        let c = unsafe { columns.map(|c| c.assume_init()) };
        Self {
            columns: c
        }
    }
}

//map_in_place
impl<T, const R: usize, const C: usize> Matrix<T,R,C> {
    /// Applies a function to each element of the matrix in place.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that mutates each element
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    ///
    /// let mut m = Matrix::<f32, 2, 2>::new_from_array([
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    /// ]);
    ///
    /// // Double each element in place
    /// m.map_in_place(|x| *x *= 2.0);
    ///
    /// assert_eq!(*m.element_at(0, 0), 2.0);
    /// assert_eq!(*m.element_at(1, 1), 8.0);
    /// ```
    pub fn map_in_place<F: FnMut(&mut T)>(&mut self, mut f: F) {
        for c in 0..C {
            self.columns[c].map_in_place(|v| f(v));
        }
    }
}

//eq_approx
impl<T, const R: usize, const C: usize> Matrix<T,R,C>
where T: Clone + crate::types::sealed::Float
{
    /// Tests if two matrices are approximately equal within a tolerance.
    ///
    /// This is useful for comparing floating-point matrices where exact
    /// equality may not hold due to rounding errors.
    ///
    /// # Arguments
    ///
    /// * `other` - The matrix to compare with
    /// * `tolerance` - The maximum allowed difference between corresponding elements
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    ///
    /// let a = Matrix::<f32, 2, 2>::new_from_array([
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    /// ]);
    ///
    /// let b = Matrix::<f32, 2, 2>::new_from_array([
    ///     1.0001, 2.0001,
    ///     3.0001, 4.0001,
    /// ]);
    ///
    /// assert!(a.eq_approx(b, 0.001));
    /// assert!(!a.eq_approx(b, 0.00001));
    /// ```
    pub fn eq_approx(self, other: Self, tolerance: T) -> bool {
        for c in 0..C {
            if !self.columns[c].clone().eq_approx(other.columns[c].clone(), tolerance.clone()) {
                return false;
            }
        }
        true
    }
}

//columns accessors
impl<T, const R: usize, const C: usize> Matrix<T,R,C> {
    /// Returns a reference to the column vectors of the matrix.
    ///
    /// Since the matrix uses column-major storage, this gives direct access
    /// to the underlying storage without any transformation.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    /// use vectormatrix::vector::Vector;
    ///
    /// let m = Matrix::new_columns([
    ///     Vector::new([1.0, 4.0]),
    ///     Vector::new([2.0, 5.0]),
    ///     Vector::new([3.0, 6.0]),
    /// ]);
    ///
    /// let columns = m.columns();
    /// assert_eq!(columns[0], Vector::new([1.0, 4.0]));
    /// assert_eq!(columns[1], Vector::new([2.0, 5.0]));
    /// assert_eq!(columns[2], Vector::new([3.0, 6.0]));
    /// ```
    #[inline] pub fn columns(&self) -> &[Vector<T,R>; C] {
        &self.columns
    }

    /// Returns a mutable reference to the column vectors of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    /// use vectormatrix::vector::Vector;
    ///
    /// let mut m = Matrix::new_columns([
    ///     Vector::new([1.0, 2.0]),
    ///     Vector::new([3.0, 4.0]),
    /// ]);
    ///
    /// // Modify the second column
    /// m.columns_mut()[1] = Vector::new([5.0, 6.0]);
    ///
    /// assert_eq!(*m.element_at(0, 1), 5.0);
    /// assert_eq!(*m.element_at(1, 1), 6.0);
    /// ```
    #[inline] pub fn columns_mut(&mut self) -> &mut [Vector<T,R>; C] {
        &mut self.columns
    }

    /// Returns a reference to the element at the specified row and column.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index (0-based)
    /// * `col` - The column index (0-based)
    ///
    /// # Panics
    ///
    /// Panics if `row >= R` or `col >= C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    ///
    /// let m = Matrix::<f32, 2, 3>::new_from_array([
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    /// ]);
    ///
    /// assert_eq!(*m.element_at(0, 0), 1.0);
    /// assert_eq!(*m.element_at(1, 2), 6.0);
    /// ```
    #[inline] pub fn element_at(&self, row: usize, col: usize) -> &T {
        &self.columns[col][row]
    }

    /// Returns a mutable reference to the element at the specified row and column.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index (0-based)
    /// * `col` - The column index (0-based)
    ///
    /// # Panics
    ///
    /// Panics if `row >= R` or `col >= C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    ///
    /// let mut m = Matrix::<f32, 2, 2>::new_from_array([
    ///     1.0, 2.0,
    ///     3.0, 4.0,
    /// ]);
    ///
    /// *m.element_at_mut(0, 1) = 10.0;
    /// assert_eq!(*m.element_at(0, 1), 10.0);
    /// ```
    #[inline] pub fn element_at_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.columns[col][row]
    }
}

//row accessor
impl<T, const R: usize, const C: usize> Matrix<T,R,C>
where T: Clone {
    /// Returns the rows of the matrix as an array of vectors.
    ///
    /// Since the matrix uses column-major storage internally, this method
    /// creates a transposed copy to extract the rows. For direct access to
    /// the underlying storage, use `columns()` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    /// use vectormatrix::vector::Vector;
    ///
    /// let m = Matrix::new_rows([
    ///     Vector::new([1.0, 2.0, 3.0]),
    ///     Vector::new([4.0, 5.0, 6.0]),
    /// ]);
    ///
    /// let rows = m.rows();
    /// assert_eq!(rows[0], Vector::new([1.0, 2.0, 3.0]));
    /// assert_eq!(rows[1], Vector::new([4.0, 5.0, 6.0]));
    /// ```
    #[inline] pub fn rows(self) -> [Vector<T, C>; R] {
        self.transpose().columns
    }
}


impl <T, const R: usize, const C: usize> Debug for Matrix<T,R,C>
where
    T: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        //pretty-print the matrix
        for row in 0..R {
            for col in 0..C {
                // Right-align in a width of 8 for demonstration:
                write!(f, "{:>8.3?}", self.columns[col][row])?;
            }
            // End the row
            writeln!(f)?;
        }

        Ok(())
    }
}

//matrix multiplication

impl <T, const M: usize, const N: usize> Matrix<T,M,N>
where T: Clone + core::ops::Mul<Output=T> + core::ops::Add<Output=T>
{
    /// Performs matrix multiplication with another matrix.
    ///
    /// Computes the product of this M×N matrix with an N×P matrix, resulting
    /// in an M×P matrix. Each element (i,j) in the result is the dot product
    /// of row i from this matrix and column j from the other matrix.
    ///
    /// # Type Parameters
    ///
    /// * `P` - The number of columns in the other matrix
    ///
    /// # Arguments
    ///
    /// * `other` - An N×P matrix to multiply with
    ///
    /// # Examples
    ///
    /// ```
    /// use vectormatrix::matrix::Matrix;
    /// use vectormatrix::vector::Vector;
    ///
    /// // 2×3 matrix
    /// let a = Matrix::new_rows([
    ///     Vector::new([1.0, 2.0, 3.0]),
    ///     Vector::new([4.0, 5.0, 6.0]),
    /// ]);
    ///
    /// // 3×2 matrix
    /// let b = Matrix::new_rows([
    ///     Vector::new([7.0, 8.0]),
    ///     Vector::new([9.0, 10.0]),
    ///     Vector::new([11.0, 12.0]),
    /// ]);
    ///
    /// // Result is 2×2
    /// let result = a.mul_matrix(b);
    ///
    /// // | 1 2 3 |   | 7  8  |   | 58  64 |
    /// // | 4 5 6 | × | 9  10 | = | 139 154|
    /// //             | 11 12 |
    ///
    /// assert_eq!(*result.element_at(0, 0), 58.0);
    /// assert_eq!(*result.element_at(0, 1), 64.0);
    /// assert_eq!(*result.element_at(1, 0), 139.0);
    /// assert_eq!(*result.element_at(1, 1), 154.0);
    /// ```
    #[inline]
    pub fn mul_matrix<const P: usize>(self, other: Matrix<T, N, P>) -> Matrix<T, M, P> {
        let mut out = Matrix::UNINIT;
        let a_rows = self.transpose();
        for c in 0..P { //columns of output
            let b_column = other.columns[c].clone();
            for r in 0..M {
                out.columns[c][r] = MaybeUninit::new(a_rows.columns[r].clone().dot(b_column.clone()));
            }
        }

        unsafe{out.assume_init()}

    }
}

impl<T, const M: usize, const N: usize,const P: usize> core::ops::Mul<Matrix<T, N, P>> for Matrix<T, M, N>
where
    T: Clone + core::ops::Mul<Output=T> + core::ops::Add<Output=T>
{
    type Output = Matrix<T, M, P>;
    #[inline]
    fn mul(self, other: Matrix<T, N, P>) -> Self::Output {
        self.mul_matrix(other)
    }
}

impl<T, const R: usize, const C: usize> core::ops::Mul<Vector<T, C>> for Matrix<T, R, C>
where
    T: Clone + core::ops::Mul<Output=T> + core::ops::Add<Output=T>
{
    type Output = Matrix<T, R, 1>;
    #[inline]
    fn mul(self, other: Vector<T, C>) -> Self::Output {
        self.mul_matrix(other.to_col())
    }
}

//boilerplate

impl<T: Default, const R: usize, const C: usize> Default for Matrix<T,R,C> where T: Copy {
    fn default() -> Self {
        Self::new_columns([Default::default(); C])
    }
}

//from/into
impl<T, const R: usize, const C: usize> From<[Vector<T,R>; C]> for Matrix<T,R,C> {
    fn from(arr: [Vector<T,R>; C]) -> Self {
        Self::new_columns(arr)
    }
}

impl<T, const R: usize, const C: usize> From<Matrix<T,R,C>> for [Vector<T,R>; C] {
    fn from(m: Matrix<T,R,C>) -> Self {
        m.columns
    }
}

//asref / asmut
impl<T, const R: usize, const C: usize> AsRef<[Vector<T,R>; C]> for Matrix<T,R,C> {
    fn as_ref(&self) -> &[Vector<T,R>; C] {
        &self.columns
    }
}

impl<T, const R: usize, const C: usize> AsMut<[Vector<T,R>; C]> for Matrix<T,R,C> {
    fn as_mut(&mut self) -> &mut [Vector<T,R>; C] {
        &mut self.columns
    }
}





#[cfg(test)]
mod tests {
    #[test] fn test() {
        use crate::vector::Vector;
        use crate::matrix::Matrix;
        let m = Matrix::<f32, 2, 3>::new_rows([
            Vector::new([1.0, 2.0, 3.0]),
            Vector::new([4.0, 5.0, 6.0]),
        ]);
        assert_eq!(m.columns[0], Vector::new([1.0, 4.0]));
        assert_eq!(m.columns[1], Vector::new([2.0, 5.0]));
        assert_eq!(m.columns[2], Vector::new([3.0, 6.0]));
    }

    #[test] fn test_from_array() {
        use crate::vector::Vector;
        use crate::matrix::Matrix;
        let m = Matrix::<f32, 2, 3>::new_from_array([
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);
        assert_eq!(m.columns[0], Vector::new([1.0, 4.0]));
        assert_eq!(m.columns[1], Vector::new([2.0, 5.0]));
        assert_eq!(m.columns[2], Vector::new([3.0, 6.0]));
    }
    #[test] fn test_debug() {
        use crate::vector::Vector;
        use crate::matrix::Matrix;
        let m = Matrix::<f32, 2, 3>::new_rows([
            Vector::new([1.0, 2.0, 3.0]),
            Vector::new([4.0, 5.0, 6.0]),
        ]);
        use alloc::format;
        // println!("{:?}", m);
        assert_eq!(format!("{:?}", m), "   1.000   2.000   3.000\n   4.000   5.000   6.000\n");
    }

    #[test] fn test_elementwise() {
        use crate::vector::Vector;
        use crate::matrix::Matrix;
        let m1 = Matrix::<f32, 2, 3>::new_rows([
            Vector::new([1.0, 2.0, 3.0]),
            Vector::new([4.0, 5.0, 6.0]),
        ]);
        let m2 = Matrix::<f32, 2, 3>::new_rows([
            Vector::new([7.0, 8.0, 9.0]),
            Vector::new([10.0, 11.0, 12.0]),
        ]);
        let m3 = Matrix::<f32, 2, 3>::new_rows([
            Vector::new([8.0, 10.0, 12.0]),
            Vector::new([14.0, 16.0, 18.0]),
        ]);
        assert_eq!(m1 + m2, m3);
        assert_eq!(m3 - m2, m1);
        assert_eq!(m1.elementwise_mul(m2), Matrix::new_rows([
            Vector::new([7.0, 16.0, 27.0]),
            Vector::new([40.0, 55.0, 72.0]),
        ]));

        assert_eq!(m3.elementwise_div(m2), Matrix::new_rows([
            Vector::new([8.0/7.0, 10.0/8.0, 12.0/9.0]),
            Vector::new([14.0/10.0, 16.0/11.0, 18.0/12.0]),
        ]));
    }

    #[test] fn map() {
        use crate::vector::Vector;
        use crate::matrix::Matrix;
        let m1 = Matrix::<f32, 2, 3>::new_rows([
            Vector::new([1.0, 2.0, 3.0]),
            Vector::new([4.0, 5.0, 6.0]),
        ]);
        let m2 = Matrix::<f32, 2, 3>::new_rows([
            Vector::new([2.0, 4.0, 6.0]),
            Vector::new([8.0, 10.0, 12.0]),
        ]);
        assert_eq!(m1.map(|v| v * 2.0), m2);
    }

    #[test] fn rows_cols() {
        use crate::vector::Vector;
        use crate::matrix::Matrix;
        let m1 = Matrix::<f32, 2, 3>::new_rows([
            Vector::new([1.0, 2.0, 3.0]),
            Vector::new([4.0, 5.0, 6.0]),
        ]);
        assert_eq!(m1.columns(), &[Vector::new([1.0, 4.0]), Vector::new([2.0, 5.0]), Vector::new([3.0, 6.0])]);
        assert_eq!(m1.rows(), [Vector::new([1.0, 2.0, 3.0]), Vector::new([4.0, 5.0, 6.0])]);
    }

    #[test] fn test_mul() {
        use crate::vector::Vector;
        use crate::matrix::Matrix;
        let m1 = Matrix::<f32, 2, 3>::new_rows([
            Vector::new([1.0, 2.0, 3.0]),
            Vector::new([4.0, 5.0, 6.0]),
        ]);
        let m2 = Matrix::<f32, 3, 2>::new_rows([
            Vector::new([7.0, 8.0]),
            Vector::new([9.0, 10.0]),
            Vector::new([11.0, 12.0]),
        ]);
        let m3 = Matrix::<f32, 2, 2>::new_rows([
            Vector::new([58.0, 64.0]),
            Vector::new([139.0, 154.0]),
        ]);
        assert_eq!(m1.clone() * m2.clone(), m3);
        assert_eq!(m1.clone().mul_matrix(m2), m3);
    }
}