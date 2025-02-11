/*!
2x2 matrix algorithms
*/

use crate::matrix::Matrix;

impl<T> Matrix<T,2,2> {
    #[inline] pub fn determinant(self) -> T where T: Clone + core::ops::Sub<Output=T> + core::ops::Mul<Output=T> {
        let a = self.columns()[0].x().clone();
        let b = self.columns()[1].x().clone();
        let c = self.columns()[0].y().clone();
        let d = self.columns()[1].y().clone();
        a * d - b * c
    }
}

#[cfg(test)] mod tests {
    use crate::matrix::Matrix;
    use crate::vec::Vector;

    #[test] fn determinant() {
        let m = Matrix::new_rows([
            Vector::new([3, 7]),
            Vector::new([1, -4]),
        ]);
        assert_eq!(m.determinant(), -19);
    }
}