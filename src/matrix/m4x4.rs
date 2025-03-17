//SPDX-License-Identifier: MIT OR Apache-2.0
/*!
4x4 matrix algorithms
 */

use crate::matrix::Matrix;
use crate::types::sealed::{Constants, Float};
use crate::vector::Vector;

impl<T> Matrix<T,4,4> {
    /**
    Constructs a 4x4 matrix that performs a translation operation.

    This returns the matrix

    ```ignore
    | 1 0 0 dx |
    | 0 1 0 dy |
    | 0 0 1 dz |
    | 0 0 0 1  |
    ```
    */
    #[inline] pub const fn translation_matrix(dx: T, dy: T, dz: T) -> Self where T: Constants {
        Matrix::new_columns([
            Vector::new([T::ONE, T::ZERO, T::ZERO, T::ZERO]),
            Vector::new([T::ZERO, T::ONE, T::ZERO, T::ZERO]),
            Vector::new([T::ZERO, T::ZERO, T::ONE, T::ZERO]),
            Vector::new([dx, dy, dz, T::ONE]),
        ])
    }

    /**
    Constructs a 4x4 matrix that applies a counterclockwise rotation of `theta` radians around the x axis.

    This returns the matrix
    ```ignore
    | 1 0 0 0 |
    | 0 cos(theta) -sin(theta) 0 |
    | 0 sin(theta)  cos(theta) 0 |
    | 0 0 0 1 |
    ```
    */

    #[inline] pub fn rotation_matrix_x(theta: T) -> Self where T: Float + core::ops::Neg<Output=T> + Clone + Constants {
        let s = theta.clone().f_sin();
        let c = theta.f_cos();
        Matrix::new_columns([
            Vector::new([T::ONE, T::ZERO, T::ZERO, T::ZERO]),
            Vector::new([T::ZERO, c.clone(), s.clone(), T::ZERO]),
            Vector::new([T::ZERO, -s, c, T::ZERO]),
            Vector::new([T::ZERO, T::ZERO, T::ZERO, T::ONE]),
        ])
    }

    /**
    Constructs a 4x4 matrix that applies a counterclockwise rotation of `theta` radians around the y axis.

    This returns the matrix
    ```ignore
    | cos(theta) 0 sin(theta) 0 |
    | 0 1 0 0 |
    | -sin(theta) 0 cos(theta) 0 |
    | 0 0 0 1 |
    ```
    */

    #[inline] pub fn rotation_matrix_y(theta: T) -> Self where T: Float + core::ops::Neg<Output=T> + Clone + Constants {
        let s = theta.clone().f_sin();
        let c = theta.f_cos();
        Matrix::new_columns([
            Vector::new([c.clone(), T::ZERO, -s.clone(), T::ZERO]),
            Vector::new([T::ZERO, T::ONE, T::ZERO, T::ZERO]),
            Vector::new([s, T::ZERO, c, T::ZERO]),
            Vector::new([T::ZERO, T::ZERO, T::ZERO, T::ONE]),
        ])
    }

    /**
    Constructs a 4x4 matrix that applies a counterclockwise rotation of `theta` radians around the z axis.

    This returns the matrix
    ```ignore
    | cos(theta) -sin(theta) 0 0 |
    | sin(theta)  cos(theta) 0 0 |
    | 0 0 1 0 |
    | 0 0 0 1 |
    ```
    */

    #[inline] pub fn rotation_matrix_z(theta: T) -> Self where T: Float + core::ops::Neg<Output=T> + Clone + Constants {
        let s = theta.clone().f_sin();
        let c = theta.f_cos();
        Matrix::new_columns([
            Vector::new([c.clone(), s.clone(), T::ZERO, T::ZERO]),
            Vector::new([-s, c, T::ZERO, T::ZERO]),
            Vector::new([T::ZERO, T::ZERO, T::ONE, T::ZERO]),
            Vector::new([T::ZERO, T::ZERO, T::ZERO, T::ONE]),
        ])
    }

    /**
    Constructs a 4x4 matrix that applies a scaling operation.

    This returns the matrix
    ```ignore
    | sx 0 0 0 |
    | 0 sy 0 0 |
    | 0 0 sz 0 |
    | 0 0 0 1 |
    ```
    */

    #[inline] pub const fn scaling_matrix(sx: T, sy: T, sz: T) -> Self where T: Constants {
        Matrix::new_columns([
            Vector::new([sx, T::ZERO, T::ZERO, T::ZERO]),
            Vector::new([T::ZERO, sy, T::ZERO, T::ZERO]),
            Vector::new([T::ZERO, T::ZERO, sz, T::ZERO]),
            Vector::new([T::ZERO, T::ZERO, T::ZERO, T::ONE]),
        ])
    }

    /**
    Constructs a 4x4 matrix that applies a shear operation.

    This returns the matrix
    ```ignore
    |1 sxy sxz 0 |
    |syx 1 syz 0 |
    |szx szy 1 0 |
    |0 0 0 1 |
    ```
    */

    #[inline] pub const fn shear_matrix(sxy: T, sxz: T, syx: T, syz: T, szx: T, szy: T) -> Self where T: Constants {
        Matrix::new_columns([
            Vector::new([T::ONE, syx, szx, T::ZERO]),
            Vector::new([sxy, T::ONE, szy, T::ZERO]),
            Vector::new([sxz, syz, T::ONE, T::ZERO]),
            Vector::new([T::ZERO, T::ZERO, T::ZERO, T::ONE]),
        ])
    }

    /**
    Calculates the determinant of the matrix.
    */
    pub fn determinant(self) -> T where T: Clone + core::ops::Mul<Output=T> + core::ops::Sub<Output=T> + core::ops::Add<Output=T> + core::ops::Neg<Output=T> {
        let m0 = self.columns()[0].x().clone();
        let m1 = self.columns()[1].x().clone();
        let m2 = self.columns()[2].x().clone();
        let m3 = self.columns()[3].x().clone();
        let m4 = self.columns()[0].y().clone();
        let m5 = self.columns()[1].y().clone();
        let m6 = self.columns()[2].y().clone();
        let m7 = self.columns()[3].y().clone();
        let m8 = self.columns()[0].z().clone();
        let m9 = self.columns()[1].z().clone();
        let m10 = self.columns()[2].z().clone();
        let m11 = self.columns()[3].z().clone();
        let m12 = self.columns()[0].w().clone();
        let m13 = self.columns()[1].w().clone();
        let m14 = self.columns()[2].w().clone();
        let m15 = self.columns()[3].w().clone();

        //these inv values represent the adjugate matrix (ie, cofactor matrix transposed).
        let inv0 = m5.clone()  * m10.clone() * m15.clone() -
            m5.clone()  * m11.clone() * m14.clone() -
            m9.clone()  * m6.clone() * m15.clone() +
            m9.clone()  * m7.clone()  * m14.clone() +
            m13.clone() * m6.clone()  * m11.clone() -
            m13.clone() * m7.clone()  * m10.clone();
        let inv4 = -m4.clone()  * m10.clone()  * m15.clone() +
            m4.clone()  * m11.clone() * m14.clone() +
            m8.clone()  * m6.clone()  * m15.clone() -
            m8.clone()  * m7.clone()  * m14.clone() -
            m12.clone() * m6.clone()  * m11.clone() +
            m12.clone() * m7.clone() * m10.clone();

        let inv8 = m4.clone()  * m9.clone() * m15.clone() -
            m4.clone()  * m11.clone() * m13.clone() -
            m8.clone()  * m5.clone() * m15.clone() +
            m8.clone()  * m7.clone() * m13.clone() +
            m12.clone() * m5.clone() * m11.clone() -
            m12.clone() * m7.clone() * m9.clone();

        let inv12 = -m4.clone()  * m9.clone() * m14.clone() +
            m4.clone()  * m10.clone() * m13.clone() +
            m8.clone()  * m5.clone() * m14.clone() -
            m8.clone()  * m6.clone() * m13.clone() -
            m12.clone() * m5.clone() * m10.clone() +
            m12.clone() * m6.clone() * m9.clone();


        m0 * inv0 + m1 * inv4 + m2 * inv8 + m3 * inv12
    }

    /**
    Finds the inverse of the matrix.
    */
    pub fn inverse(self) -> Option<Self> where T: Constants + Clone + core::ops::Mul<Output=T> + core::ops::Sub<Output=T> + core::ops::Add<Output=T> + core::ops::Neg<Output=T> + core::ops::Div<Output=T> + PartialEq {
        let determinant = self.clone().determinant();
        if determinant == T::ZERO {
            None
        }
        else {
            let m0 = self.columns()[0].x().clone();
            let m1 = self.columns()[1].x().clone();
            let m2 = self.columns()[2].x().clone();
            let m3 = self.columns()[3].x().clone();
            let m4 = self.columns()[0].y().clone();
            let m5 = self.columns()[1].y().clone();
            let m6 = self.columns()[2].y().clone();
            let m7 = self.columns()[3].y().clone();
            let m8 = self.columns()[0].z().clone();
            let m9 = self.columns()[1].z().clone();
            let m10 = self.columns()[2].z().clone();
            let m11 = self.columns()[3].z().clone();
            let m12 = self.columns()[0].w().clone();
            let m13 = self.columns()[1].w().clone();
            let m14 = self.columns()[2].w().clone();
            let m15 = self.columns()[3].w().clone();

            let det_inverse = T::ONE / determinant;
            //these inv values represent the adjugate matrix (ie, cofactor matrix transposed).
            let inv0 = m5.clone()  * m10.clone() * m15.clone() -
                m5.clone()  * m11.clone() * m14.clone() -
                m9.clone()  * m6.clone() * m15.clone() +
                m9.clone()  * m7.clone()  * m14.clone() +
                m13.clone() * m6.clone()  * m11.clone() -
                m13.clone() * m7.clone()  * m10.clone();
            let inv4 = -m4.clone()  * m10.clone()  * m15.clone() +
                m4.clone()  * m11.clone() * m14.clone() +
                m8.clone()  * m6.clone()  * m15.clone() -
                m8.clone()  * m7.clone()  * m14.clone() -
                m12.clone() * m6.clone()  * m11.clone() +
                m12.clone() * m7.clone() * m10.clone();
            let inv8 = m4.clone()  * m9.clone() * m15.clone() -
                m4.clone()  * m11.clone() * m13.clone() -
                m8.clone()  * m5.clone() * m15.clone() +
                m8.clone()  * m7.clone() * m13.clone() +
                m12.clone() * m5.clone() * m11.clone() -
                m12.clone() * m7.clone() * m9.clone();
            let inv12 = -m4.clone()  * m9.clone() * m14.clone() +
                m4.clone()  * m10.clone() * m13.clone() +
                m8.clone()  * m5.clone() * m14.clone() -
                m8.clone()  * m6.clone() * m13.clone() -
                m12.clone() * m5.clone() * m10.clone() +
                m12.clone() * m6.clone() * m9.clone();

            let inv1 = -m1.clone()  * m10.clone() * m15.clone() +
                m1.clone()  * m11.clone() * m14.clone() +
                m9.clone()  * m2.clone() * m15.clone() -
                m9.clone()  * m3.clone() * m14.clone() -
                m13.clone() * m2.clone() * m11.clone() +
                m13.clone() * m3.clone() * m10.clone();

            let inv5 = m0.clone()  * m10.clone() * m15.clone() -
                m0.clone()  * m11.clone() * m14.clone() -
                m8.clone()  * m2.clone() * m15.clone() +
                m8.clone()  * m3.clone() * m14.clone() +
                m12.clone() * m2.clone() * m11.clone() -
                m12.clone() * m3.clone() * m10.clone();

            let inv9 = -m0.clone()  * m9.clone() * m15.clone() +
                m0.clone()  * m11.clone() * m13.clone() +
                m8.clone()  * m1.clone() * m15.clone() -
                m8.clone()  * m3.clone() * m13.clone() -
                m12.clone() * m1.clone() * m11.clone() +
                m12.clone() * m3.clone() * m9.clone();

            let inv13 = m0.clone()  * m9.clone() * m14.clone() -
                m0.clone()  * m10.clone() * m13.clone() -
                m8.clone()  * m1.clone() * m14.clone() +
                m8.clone()  * m2.clone() * m13.clone() +
                m12.clone() * m1.clone() * m10.clone() -
                m12.clone() * m2.clone() * m9.clone();

            let inv2 = m1.clone()  * m6.clone() * m15.clone() -
                m1.clone()  * m7.clone() * m14.clone() -
                m5.clone()  * m2.clone() * m15.clone() +
                m5.clone()  * m3.clone() * m14.clone() +
                m13.clone() * m2.clone() * m7.clone() -
                m13.clone() * m3.clone() * m6.clone();

            let inv6 = -m0.clone()  * m6.clone() * m15.clone() +
                m0.clone()  * m7.clone() * m14.clone() +
                m4.clone()  * m2.clone() * m15.clone() -
                m4.clone()  * m3.clone() * m14.clone() -
                m12.clone() * m2.clone() * m7.clone() +
                m12.clone() * m3.clone() * m6.clone();

            let inv10 = m0.clone()  * m5.clone() * m15.clone() -
                m0.clone()  * m7.clone() * m13.clone() -
                m4.clone()  * m1.clone() * m15.clone() +
                m4.clone()  * m3.clone() * m13.clone() +
                m12.clone() * m1.clone() * m7.clone() -
                m12.clone() * m3.clone() * m5.clone();

            let inv14 = -m0.clone()  * m5.clone() * m14.clone() +
                m0.clone()  * m6.clone() * m13.clone() +
                m4.clone()  * m1.clone() * m14.clone() -
                m4.clone()  * m2.clone() * m13.clone() -
                m12.clone() * m1.clone() * m6.clone() +
                m12.clone() * m2.clone() * m5.clone();

            let inv3 = -m1.clone() * m6.clone() * m11.clone() +
                m1.clone() * m7.clone() * m10.clone() +
                m5.clone() * m2.clone() * m11.clone() -
                m5.clone() * m3.clone() * m10.clone() -
                m9.clone() * m2.clone() * m7.clone() +
                m9.clone() * m3.clone() * m6.clone();

            let inv7 = m0.clone() * m6.clone() * m11.clone() -
                m0.clone() * m7.clone() * m10.clone() -
                m4.clone() * m2.clone() * m11.clone()+
                m4.clone() * m3.clone() * m10.clone() +
                m8.clone() * m2.clone() * m7.clone() -
                m8.clone() * m3.clone() * m6.clone();

            let inv11 = -m0.clone() * m5.clone() * m11.clone() +
                m0.clone() * m7.clone() * m9.clone() +
                m4.clone() * m1.clone() * m11 -
                m4.clone() * m3.clone() * m9.clone() -
                m8.clone() * m1.clone() * m7 +
                m8.clone() * m3.clone() * m5.clone();

            let inv15 = m0.clone() * m5.clone() * m10.clone() -
                m0.clone() * m6.clone() * m9.clone() -
                m4.clone() * m1.clone() * m10 +
                m4 * m2.clone() * m9 +
                m8.clone() * m1.clone() * m6 -
                m8 * m2.clone() * m5;

            let mtx = Self::new_columns([
                Vector::new([inv0, inv4, inv8, inv12]),
                Vector::new([inv1, inv5, inv9, inv13]),
                Vector::new([inv2, inv6, inv10, inv14]),
                Vector::new([inv3, inv7, inv11, inv15])

            ]);
            Some(mtx * det_inverse)
        }
    }

}

#[cfg(test)] mod tests {
    use crate::matrix::Matrix;
    use crate::vector::Vector;

    #[test] fn determinant() {
        let m = Matrix::new_rows([
            Vector::new([4, 3, 2, 2]),
            Vector::new([0, 1, -3, 3]),
            Vector::new([0, -1, 3, 3]),
            Vector::new([0, 3, 1, 1]),
        ]);
        assert_eq!(m.determinant(), -240);
    }

    #[test] fn inverse() {
        let m = Matrix::new_rows([
            Vector::new([4.0f32, 3.0, 2.0, 2.0]),
            Vector::new([0.0, 1.0, -3.0, 3.0]),
            Vector::new([0.0, -1.0, 3.0, 3.0]),
            Vector::new([0.0, 3.0, 1.0, 1.0]),
        ]);
        let inv = m.inverse().unwrap();
        assert!(inv.eq_approx(Matrix::new_rows([
            Vector::new([0.25, 0.0, -3.0/40.0, -11.0/40.0]),
            Vector::new([0.0, 0.0, -1.0/10.0, 3.0/10.0]),
            Vector::new([0.0, -1.0/6.0, 2.0/15.0, 1.0/10.0]),
            Vector::new([0.0, 1.0/6.0, 1.0/6.0, 0.0]),
        ]), 0.001));
    }
}