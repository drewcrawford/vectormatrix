mod m3x3;
mod m4x4;
mod m2x2;

use core::fmt::Debug;
use core::mem::MaybeUninit;
use crate::types::Constants;
use crate::vec::Vector;

#[derive(Copy,Clone,PartialEq)]
pub struct Matrix<T, const R: usize, const C: usize> {
    columns: [Vector<T,R>; C],
}

impl<T, const R: usize, const C: usize> Matrix<T,R,C> {

    pub const UNINIT: Matrix<MaybeUninit<T>,R,C> = Matrix {
        columns: [Vector::UNINIT; C],
    };

    pub const fn new_columns(columns: [Vector<T, R>; C]) -> Self {
        Self {
            columns,
        }
    }

    pub fn new_rows(rows: [Vector<T, C>; R]) -> Self
    where
        T: Clone
    {
        Matrix::new_columns(rows).transpose()
    }

    pub fn new_from_array<const L: usize>(arr: [T; L]) -> Self
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
    pub unsafe fn assume_init(self) -> Matrix<T,R,C> {
        let columns = self.columns.map(|maybe| unsafe { maybe.assume_init() });
        Matrix {
            columns
        }
    }
}

//elementwise add
impl<T: Constants + Clone + core::ops::Add<Output=T> , const R: usize, const C: usize> Matrix<T,R,C> {
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
    pub const ZERO: Self = Self {
        columns: [Vector::ZERO; C],
    };
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
    pub fn map<F: FnMut(T) -> T>(self, mut f: F) -> Self {
        let mut columns = [const { MaybeUninit::uninit() }; C];
        for c in 0..C {
            columns[c] = MaybeUninit::new(self.columns[c].clone().map(|v| f(v)));
        }
        let c = unsafe { columns.map(|c| c.assume_init()) };
        Self {
            columns: c
        }
    }
}

//map_in_place
impl<T, const R: usize, const C: usize> Matrix<T,R,C> {
    pub fn map_in_place<F: FnMut(&mut T)>(&mut self, mut f: F) {
        for c in 0..C {
            self.columns[c].map_in_place(|v| f(v));
        }
    }
}

//eq_approx
impl<T, const R: usize, const C: usize> Matrix<T,R,C>
where T: Clone + crate::types::Float
{
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
    /**
    Access the columns of the underlying matrix.

    Since the matrix storage is column-major, we return an inner slice of the columns.
    */
    #[inline] pub fn columns(&self) -> &[Vector<T,R>; C] {
        &self.columns
    }

    #[inline] pub fn columns_mut(&mut self) -> &mut [Vector<T,R>; C] {
        &mut self.columns
    }

    /**
    Returns a reference to the element at the given row and column.
    */
    #[inline] pub fn element_at(&self, row: usize, col: usize) -> &T {
        &self.columns[col][row]
    }

    /**
    Returns a mutable reference to the element at the given row and column.
    */
    #[inline] pub fn element_at_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.columns[col][row]
    }
}

//row accessor
impl<T, const R: usize, const C: usize> Matrix<T,R,C>
where T: Clone {
    /**
    Access the rows of the underlying matrix.

    Since the matrix storage is column-major, we return the transpose of the columns.
    */
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




#[cfg(test)]
mod tests {
    #[test] fn test() {
        use crate::vec::Vector;
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
        use crate::vec::Vector;
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
        use crate::vec::Vector;
        use crate::matrix::Matrix;
        let m = Matrix::<f32, 2, 3>::new_rows([
            Vector::new([1.0, 2.0, 3.0]),
            Vector::new([4.0, 5.0, 6.0]),
        ]);
        use alloc::format;
        println!("{:?}", m);
        assert_eq!(format!("{:?}", m), "   1.000   2.000   3.000\n   4.000   5.000   6.000\n");
    }

    #[test] fn test_elementwise() {
        use crate::vec::Vector;
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
        use crate::vec::Vector;
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
        use crate::vec::Vector;
        use crate::matrix::Matrix;
        let m1 = Matrix::<f32, 2, 3>::new_rows([
            Vector::new([1.0, 2.0, 3.0]),
            Vector::new([4.0, 5.0, 6.0]),
        ]);
        assert_eq!(m1.columns(), &[Vector::new([1.0, 4.0]), Vector::new([2.0, 5.0]), Vector::new([3.0, 6.0])]);
        assert_eq!(m1.rows(), [Vector::new([1.0, 2.0, 3.0]), Vector::new([4.0, 5.0, 6.0])]);
    }

    #[test] fn test_mul() {
        use crate::vec::Vector;
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