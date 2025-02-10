/*!
3x3 matrix algorithms
*/

use crate::matrix::Matrix;
use crate::types::{Constants, Float};
use crate::vec::Vector;

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

}

#[cfg(test)] mod tests {



}