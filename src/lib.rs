//! Low-level manipulations of IEEE754 floating-point numbers.
//!
//! # Installation
//!
//! Add this to your Cargo.toml:
//!
//! ```toml
//! [dependencies]
//! ieee754 = "0.2"
//! ```
//!
//! To enable `rayon` parallel iteration, activate the optional
//! `rayon` feature:
//!
//! ```toml
//! [dependencies]
//! ieee754 = { version = "0.2", features = ["rayon"] }
//! ```
//!
//! # Examples
//!
//! ```rust
//! use ieee754::Ieee754;
//!
//! // there are 840 single-precision floats between 1.0 and 1.0001
//! // (inclusive).
//! assert_eq!(1_f32.upto(1.0001).count(), 840);
//! ```
//!
//! If `rayon` is enabled, this can be performed in parallel:
//!
//! ```rust
//! extern crate ieee754;
//! # #[cfg(feature = "rayon")]
//! extern crate rayon;
//!
//! # #[cfg(feature = "rayon")]
//! # fn main() {
//! use ieee754::Ieee754;
//! use rayon::prelude::*;
//!
//! // there are 840 single-precision floats between 1.0 and 1.0001
//! // (inclusive).
//! assert_eq!(1_f32.upto(1.0001).into_par_iter().count(), 840);
//! # }
//! # #[cfg(not(feature = "rayon"))] fn main() {}
//! ```

#![no_std]
#![cfg_attr(nightly, feature(try_trait))]
#[cfg(test)] #[macro_use] extern crate std;

mod iter;
mod impls;
mod traits;

pub use traits::{Bits, Ieee754};
pub use iter::Iter;

#[cfg(feature = "rayon")]
pub use iter::rayon::ParIter;

#[inline]
#[doc(hidden)]
pub fn abs<F: Ieee754>(x: F) -> F {
    x.abs()
}
