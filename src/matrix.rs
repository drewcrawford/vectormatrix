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

    pub fn new_columns(columns: [Vector<T, R>; C]) -> Self {
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
    pub fn elementwise_add(self, other: Self) -> Self {
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

impl<T: Constants + Clone + core::ops::Add<Output=T>, const R: usize, const c: usize> core::ops::Add for Matrix<T,R,c> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.elementwise_add(other)
    }
}

impl<T: Constants + Clone + core::ops::Sub<Output=T> , const R: usize, const C: usize> Matrix<T,R,C> {
    pub fn elementwise_sub(self, other: Self) -> Self {
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
    fn sub(self, other: Self) -> Self {
        self.elementwise_sub(other)
    }
}

impl<T: Constants + Clone + core::ops::Mul<Output=T>, const R: usize, const C: usize> Matrix<T,R,C> {
    pub fn elementwise_mul(self, other: Self) -> Self {
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

impl <T: Constants + Clone + core::ops::Mul<Output=T>, const R: usize, const C: usize> core::ops::Mul for Matrix<T,R,C> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.elementwise_mul(other)
    }
}

impl <T: Constants + Clone + core::ops::Div<Output=T>, const R: usize, const C: usize> Matrix<T,R,C> {
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

impl <T: Constants + Clone + core::ops::Div<Output=T>, const R: usize, const C: usize> core::ops::Div for Matrix<T,R,C> {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self.elementwise_div(other)
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
impl <T: Constants + Clone + core::ops::AddAssign, const R: usize, const C: usize> Matrix<T,R,C> {
    fn add_assign(&mut self, other: Self) {
        for c in 0..C {
            self.columns[c] += other.columns[c].clone();
        }
    }
}

impl <T: Constants + Clone + core::ops::AddAssign, const R: usize, const C: usize> core::ops::AddAssign for Matrix<T,R,C> {
    fn add_assign(&mut self, other: Self) {
        self.add_assign(other);
    }
}

//SubAssign
impl <T: Constants + Clone + core::ops::SubAssign, const R: usize, const C: usize> Matrix<T,R,C> {
    fn sub_assign(&mut self, other: Self) {
        for c in 0..C {
            self.columns[c] -= other.columns[c].clone();
        }
    }
}
impl <T: Constants + Clone + core::ops::SubAssign, const R: usize, const C: usize> core::ops::SubAssign for Matrix<T,R,C> {
    fn sub_assign(&mut self, other: Self) {
        self.sub_assign(other);
    }
}

//MulAssign
impl <T: Constants + Clone + core::ops::MulAssign, const R: usize, const C: usize> Matrix<T,R,C> {
    fn mul_assign(&mut self, other: Self) {
        for c in 0..C {
            self.columns[c] *= other.columns[c].clone();
        }
    }
}

impl <T: Constants + Clone + core::ops::MulAssign, const R: usize, const C: usize> core::ops::MulAssign for Matrix<T,R,C> {
    fn mul_assign(&mut self, other: Self) {
        self.mul_assign(other);
    }
}

//DivAssign

impl <T: Constants + Clone + core::ops::DivAssign, const R: usize, const C: usize> Matrix<T,R,C> {
    fn div_assign(&mut self, other: Self) {
        for c in 0..C {
            self.columns[c] /= other.columns[c].clone();
        }
    }
}

impl <T: Constants + Clone + core::ops::DivAssign, const R: usize, const C: usize> core::ops::DivAssign for Matrix<T,R,C> {
    fn div_assign(&mut self, other: Self) {
        self.div_assign(other);
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
        assert_eq!(m1 * m2, Matrix::new_rows([
            Vector::new([7.0, 16.0, 27.0]),
            Vector::new([40.0, 55.0, 72.0]),
        ]));

        assert_eq!(m3 / m2, Matrix::new_rows([
            Vector::new([8.0/7.0, 10.0/8.0, 12.0/9.0]),
            Vector::new([14.0/10.0, 16.0/11.0, 18.0/12.0]),
        ]));
    }
}