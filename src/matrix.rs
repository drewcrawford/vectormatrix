use core::fmt::Debug;
use core::mem::MaybeUninit;
use crate::types::Constants;
use crate::vec::Vector;

pub struct Matrix<T, const R: usize, const C: usize> {
    columns: [Vector<T,R>; C],
}

impl<T, const R: usize, const C: usize> Matrix<T,R,C> {
    pub fn new_columns(columns: [Vector<T, R>; C]) -> Self {
        Self {
            columns,
        }
    }

    pub fn new_rows(rows: [Vector<T, C>; R]) -> Self
    where
        T: Clone
    {
        let mut columns = [const { MaybeUninit::uninit() }; C];
        for c in 0..C {
            let mut column = Vector::UNINIT;
            for r in 0..R {
                column[r] = MaybeUninit::new(rows[r][c].clone());
            }
            let column = unsafe { column.assume_init() };
            columns[c] = MaybeUninit::new(column);
        }
        let arr: [Vector<T, R>; C] = columns.map(|maybe| unsafe { maybe.assume_init() });
        Self {
            columns: arr,
        }
    }

    pub fn new_from_array<const L: usize>(arr: [T; L]) -> Self
    where T: Clone
    {
        assert_eq!(L,R * C);
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
}