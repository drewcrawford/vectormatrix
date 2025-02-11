/*!
3x3 matrix algorithms
*/

use crate::matrix::Matrix;
use crate::types::{Constants, Float};
use crate::vector::Vector;

impl<T: Constants> Matrix<T, 3, 3> {
    /**
    Constructs a 3x3 matrix that performs a translation operation.

    This returns the matrix

    ```ignore
    | 1 0 dx |
    | 0 1 dy |
    | 0 0 1  |
    ```
    */
    #[inline] pub const fn translation_matrix(dx: T, dy: T) -> Self {
        Matrix::new_columns([
            Vector::new([T::ONE, T::ZERO, T::ZERO]),
            Vector::new([T::ZERO, T::ONE, T::ZERO]),
            Vector::new([dx, dy, T::ONE]),
        ])
    }



    /**
    Constructs a 3x3 matrix that applies a counterclockwise rotation of `theta` radians.

    This returns the matrix
    ```ignore
    | cos(theta) -sin(theta) 0 |
    | sin(theta)  cos(theta) 0 |
    | 0           0          1 |
    ```
    */

    pub fn rotation_matrix(theta: T) -> Self where T: Float + core::ops::Neg<Output=T> + Clone {
        let s = theta.clone().sin();
        let c = theta.cos();
        Matrix::new_columns([
            Vector::new([c.clone(), -s.clone(), T::ZERO]),
            Vector::new([s, c, T::ZERO]),
            Vector::new([T::ZERO, T::ZERO, T::ONE]),
        ])
    }



    /**
    Constructs a 3x3 matrix that applies a scaling operation.

    This returns the matrix
    ```ignore
    | sx 0 0 |
    | 0 sy 0 |
    | 0 0 1 |
    ```
    */
    #[inline] pub const fn scaling_matrix(sx: T, sy: T) -> Self {
        Matrix::new_columns([
            Vector::new([sx, T::ZERO, T::ZERO]),
            Vector::new([T::ZERO, sy, T::ZERO]),
            Vector::new([T::ZERO, T::ZERO, T::ONE]),
        ])
    }


    /**
    Constructs a 3x3 matrix that applies a shear operation.

    This returns the matrix
    ```ignore
    | 1  sx 0 |
    | sy 1  0 |
    | 0  0  1 |
    ```
    */
    #[inline] pub const fn shear_matrix(sx: T, sy: T) -> Self {
        Matrix::new_columns([
            Vector::new([T::ONE, sy, T::ZERO]),
            Vector::new([sx, T::ONE, T::ZERO]),
            Vector::new([T::ZERO, T::ZERO, T::ONE]),
        ])
    }

    /**
    Finds the determinant of the matrix.
    */
    #[inline] pub fn determinant(self) -> T where T: Clone + core::ops::Sub<Output=T> + core::ops::Mul<Output=T> + core::ops::Add<Output=T> {
        let a = self.columns()[0].x().clone();
        let b = self.columns()[1].x().clone();
        let c = self.columns()[2].x().clone();
        let d = self.columns()[0].y().clone();
        let e = self.columns()[1].y().clone();
        let f = self.columns()[2].y().clone();
        let g = self.columns()[0].z().clone();
        let h = self.columns()[1].z().clone();
        let i = self.columns()[2].z().clone();
        a.clone() * e.clone() * i.clone() + b.clone() * f.clone() * g.clone() + c.clone() * d.clone() * h.clone() - c * e * g - b * d * i - a * f * h
    }

    /**
    Finds the inverse of the matrix.
    */
    pub fn inverse(self) -> Option<Self> where T: Clone + core::ops::Div<Output=T> + core::ops::Mul<Output=T> + core::ops::Sub<Output=T> + core::ops::Add<Output=T> + core::ops::Neg<Output=T> + PartialEq  {
        let det = self.clone().determinant();
        if det == T::ZERO {
            None
        }
        else {
            let det_inverse = T::ONE / det;
            let a = self.columns()[0].x().clone();
            let b = self.columns()[1].x().clone();
            let c = self.columns()[2].x().clone();
            let d = self.columns()[0].y().clone();
            let e = self.columns()[1].y().clone();
            let f = self.columns()[2].y().clone();
            let g = self.columns()[0].z().clone();
            let h = self.columns()[1].z().clone();
            let i = self.columns()[2].z().clone();
            let m_a = e.clone() * i.clone() - f.clone() * h.clone();
            let m_b = -(d.clone() * i.clone() - f.clone() * g.clone());
            let m_c = d.clone() * h.clone() - e.clone() * g.clone();
            let m_d = -(b.clone() * i.clone() - c.clone() * h.clone());
            let m_e = a.clone() * i - c.clone() * g.clone();
            let m_f = -(a.clone() * h - b.clone() * g.clone());
            let m_g = b.clone() * f.clone() - c.clone() * e.clone();
            let m_h = -(a.clone() * f - c * d.clone());
            let m_i = a * e - b * d;
            let m  = Matrix::new_columns([
                                             Vector::new([m_a, m_b, m_c]),
                                             Vector::new([m_d, m_e, m_f]),
                                             Vector::new([m_g, m_h, m_i]),
                                                         ]);
            Some(m * det_inverse)

        }
    }

}

#[cfg(test)] mod tests {
    #[test] fn test_determinant() {
        let m = crate::matrix::Matrix::new_columns([
            crate::vector::Vector::new([3, 7, 0]),
            crate::vector::Vector::new([1, -4, 0]),
            crate::vector::Vector::new([0, 0, 1]),
        ]);
        assert_eq!(m.determinant(), -19);
    }

    #[test] fn inverse() {
        let m = crate::matrix::Matrix::new_columns([
            crate::vector::Vector::new([3.0f32, 7.0, 0.0]),
            crate::vector::Vector::new([1.0, -4.0, 0.0]),
            crate::vector::Vector::new([0.0, 0.0, 1.0]),
        ]);
        let inv = m.inverse().unwrap();
        assert_eq!(inv, crate::matrix::Matrix::new_columns([
            crate::vector::Vector::new([4.0/19.0, 7.0/19.0, 0.0]),
            crate::vector::Vector::new([1.0/19.0, -3.0/19.0, 0.0]),
            crate::vector::Vector::new([0.0, 0.0, 1.0]),
        ]));
    }


}