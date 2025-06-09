//SPDX-License-Identifier: MIT OR Apache-2.0
/*!
Idiomatic Matrix and Vector types for Rust.

![logo](../../../art/logo.png)

This crate implements small stack-allocated Vector and Matrix types with the obvious semantics and operations.
`no_std` support and zero dependencies.

# Operations

Most obvious matrix and vector operations including:
* scalar addition, subtraction, and multiplication, including operator overloading
* elementwise addition, subtraction, and multiplication, including operator overloading wherever applicable
* map operations
* approximate equality for floating point types

## vector operations
* dot and cross product
* length, square length
* norm
* convert to/from row and column matrix

## matrix operations
* matrix multiplication
* transpose
* determinant (for 2x2, 3x3, and 4x4)
* inverse (for 2x2, 3x3, and 4x4)
* common affine transformations: translation, rotation, scaling, shear

# Type design

* Use generics to support any element type
* Use const generics to encode the matrix size in the type system and avoid heap allocation
* Where practical, use generics for optimized code at any matrix size
* Where mathematically appropriate, implement particular algorithms for particular size
* Allow inline of most operations
* Use `repr(Rust)` so that the compiler may optimize memory for simd, etc.
  (Convert to your own `repr(C)` type if you need specific memory layout).
* API design should allow for simd or hardware accelerated ops 'in the future'.

*/
#![no_std] // Always applies to the crate

#[cfg(feature = "std")]
extern crate std; // Explicitly import std when enabled

extern crate alloc;

pub mod vector;
mod types;
pub mod matrix;


