use alloc::vec::Vec;
use core::mem::MaybeUninit;
use crate::vec::Vector;

pub struct Matrix<T, const R: usize, const C: usize> {
    data: [Vector<T,R>; C],
}

impl<T, const R: usize, const C: usize> Matrix<T,R,C> {
    pub fn new_columns(columns: [Vector<T, R>; C]) -> Self {
        Self {
            data: columns,
        }
    }

    pub fn new_rows(rows: [Vector<T, C>; R]) -> Self where T: Clone {
        let mut columns = [const { MaybeUninit::uninit() }; C];
        for c in 0..C {
            let mut column = Vector::UNINIT;
            for r in 0..R {
                column[r] = MaybeUninit::new(rows[r][c].clone());
            }
            let column = unsafe { column.assume_init() };
            columns[c] = MaybeUninit::new(column);
        }
        let arr: [Vector<T,R>; C] = columns.map(|maybe| unsafe { maybe.assume_init() });
        Self {
            data: arr,
        }

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
        assert_eq!(m.data[0], Vector::new([1.0, 4.0]));
        assert_eq!(m.data[1], Vector::new([2.0, 5.0]));
        assert_eq!(m.data[2], Vector::new([3.0, 6.0]));
    }
}