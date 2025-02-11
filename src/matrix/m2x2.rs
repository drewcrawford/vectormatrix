/*!
2x2 matrix algorithms
*/

use crate::matrix::Matrix;
use crate::types::Constants;
use crate::vector::Vector;

impl<T> Matrix<T,2,2> {
    #[inline] pub fn determinant(self) -> T where T: Clone + core::ops::Sub<Output=T> + core::ops::Mul<Output=T> {
        let a = self.columns()[0].x().clone();
        let b = self.columns()[1].x().clone();
        let c = self.columns()[0].y().clone();
        let d = self.columns()[1].y().clone();
        a * d - b * c
    }

    /**
    Finds the inverse of the matrix.
*/
    pub fn inverse(self) -> Option<Self> where T: Constants + core::ops::Div<Output=T> + core::ops::Sub<Output=T> + core::ops::Mul<Output=T> + core::cmp::PartialEq + core::ops::Neg<Output=T> + Clone {
        let det = self.clone().determinant();
        if det == T::ZERO {
            None
        } else {
            let det_inverse = T::ONE / det;
            let a = self.columns()[0].x().clone();
            let b = self.columns()[1].x().clone();
            let c = self.columns()[0].y().clone();
            let d = self.columns()[1].y().clone();
            let mat = Matrix::new_columns([
                Vector::new([d, -c]),
                Vector::new([-b, a]),
            ]);
            Some(mat * det_inverse)
        }
    }
}

#[cfg(test)] mod tests {
    use crate::matrix::Matrix;
    use crate::vector::Vector;

    #[test] fn determinant() {
        let m = Matrix::new_rows([
            Vector::new([3, 7]),
            Vector::new([1, -4]),
        ]);
        assert_eq!(m.determinant(), -19);
    }
    #[test] fn inverse() {
        let m = Matrix::new_rows([
            Vector::new([3.0f32, 7.0]),
            Vector::new([1.0, -4.0]),
        ]);
        let inv = m.inverse().unwrap();
        assert_eq!(inv, Matrix::new_rows([
            Vector::new([4.0/19.0, 7.0/19.0]),
            Vector::new([1.0/19.0, -3.0/19.0]),
        ]));
    }
}