/*!
4x4 matrix algorithms
 */

use crate::matrix::Matrix;
use crate::types::{Constants, Float};
use crate::vec::Vector;

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
        let s = theta.clone().sin();
        let c = theta.cos();
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
        let s = theta.clone().sin();
        let c = theta.cos();
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
        let s = theta.clone().sin();
        let c = theta.cos();
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

}