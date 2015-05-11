//! Low-level manipulations of IEEE754 floating-point numbers.
//!
//! # Installation
//!
//! Add this to your Cargo.toml
//!
//! ```toml
//! [dependencies]
//! ieee754 = "0.1"
//! ```
//!
//! # Examples
//!
//! ```rust
//! use ieee754::Ieee754;
//!
//! // there are 840 single-precision floats in between 1.0 and 1.0001
//! // (inclusive).
//! assert_eq!(1_f32.upto(1.0001).count(), 840);
//! ```

#![cfg_attr(all(test, feature = "unstable"), feature(test))]
#[cfg(all(test, feature = "unstable"))] extern crate test;

use std::mem;

/// An iterator over floating point numbers.
pub struct Iter<T: Ieee754> {
    from: T,
    to: T,
    done: bool
}
impl<T: Ieee754> Iterator for Iter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.done { return None }

        let x = self.from;
        let y = x.next();
        if x == self.to {
            self.done = true;
        }
        self.from = y;
        return Some(x)
    }
}
impl<T: Ieee754> DoubleEndedIterator for Iter<T> {
    fn next_back(&mut self) -> Option<T> {
        if self.done { return None }

        let x = self.to;
        let y = x.prev();
        if x == self.from {
            self.done = true;
        }
        self.to = y;
        return Some(x)
    }
}

/// Types that are IEEE754 floating point numbers.
pub trait Ieee754: Copy + PartialEq + PartialOrd {
    /// Iterate over each value of `T` in `[self, lim]`.
    ///
    /// # Panics
    ///
    /// Panics if `self > lim`, or if either are NaN.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ieee754::Ieee754;
    ///
    /// // there are 840 single-precision floats in between 1.0 and 1.0001
    /// // (inclusive).
    /// assert_eq!(1_f32.upto(1.0001).count(), 840);
    /// ```
    fn upto(self, lim: Self) -> Iter<Self> {
        assert!(self <= lim);
        Iter {
            from: self,
            to: lim,
            done: false,
        }
    }

    /// A type that represents the raw bits of `Self`.
    type Bits;
    /// A type large enough to store the exponent of `Self`.
    type Exponent;
    /// A type large enough to store the significand of `Self`.
    type Signif;

    /// Return the next value after `self`.
    ///
    /// Calling this on NaN will yield nonsense.
    fn next(self) -> Self;

    /// Return the previous value before `self`.
    ///
    /// Calling this on NaN will yield nonsense.
    fn prev(self) -> Self;
    /// View `self` as a collection of bits.
    fn bits(self) -> Self::Bits;
    /// View a collections of bits as a floating point number.
    fn from_bits(x: Self::Bits) -> Self;
    /// Get the bias of the stored exponent.
    fn exponent_bias(self) -> Self::Exponent;
    /// Break `self` into the three constituent parts of an IEEE754 float.
    ///
    /// The exponent returned is the raw bits, use `exponent_bias` to
    /// compute the offset required.
    fn decompose(self) -> (bool, Self::Exponent, Self::Signif);
    /// Create a `Self` out of the three constituent parts of an IEEE754 float.
    ///
    /// The exponent should be the raw bits, use `exponent_bias` to
    /// compute the offset required.
    fn recompose(sign: bool, expn: Self::Exponent, signif: Self::Signif) -> Self;
}

macro_rules! mask{
    ($bits: expr; $current: expr => $($other: expr),*) => {
        ($bits >> (0 $(+ $other)*)) & ((1 << $current) - 1)
    }
}
macro_rules! unmask {
    ($x: expr => $($other: expr),*) => {
        $x << (0 $(+ $other)*)
    }
}

macro_rules! mk_impl {
    ($f: ty, $bits: ty, $expn: ty, $signif: ty,
     $expn_n: expr, $signif_n: expr) => {
        impl Ieee754 for $f {
            type Bits = $bits;
            type Exponent = $expn;
            type Signif = $signif;

            #[inline]
            fn next(self) -> Self {
                let abs_mask = (!(0 as Self::Bits)) >> 1;
                let (sign, _expn, _signif) = self.decompose();
                let mut bits = self.bits();
                if bits & abs_mask == 0 {
                    bits = 1;
                } else if sign {
                    // neg
                    bits -= 1;
                } else {
                    // pos
                    bits += 1;
                }
                Self::from_bits(bits)
            }
            #[inline]
            fn prev(self) -> Self {
                let abs_mask = (!(0 as Self::Bits)) >> 1;
                let (sign, _expn, _signif) = self.decompose();
                let mut bits = self.bits();
                if bits & abs_mask == 0 {
                    bits = 1 | !abs_mask;
                } else if sign {
                    bits += 1;
                } else {
                    bits -= 1;
                }
                Self::from_bits(bits)
            }

            fn exponent_bias(self) -> Self::Exponent {
                1 << ($expn_n - 1) - 1
            }

            #[inline]
            fn bits(self) -> Self::Bits {
                unsafe {mem::transmute(self)}
            }
            #[inline]
            fn from_bits(bits: Self::Bits) -> Self {
                unsafe {mem::transmute(bits)}
            }
            #[inline]
            fn decompose(self) -> (bool, Self::Exponent, Self::Signif) {
                let bits = self.bits();

                (mask!(bits; 1 => $expn_n, $signif_n) != 0,
                 mask!(bits; $expn_n => $signif_n) as Self::Exponent,
                 mask!(bits; $signif_n => ) as Self::Signif)

            }
            #[inline]
            fn recompose(sign: bool, expn: Self::Exponent, signif: Self::Signif) -> Self {
                Self::from_bits(
                    unmask!(sign as Self::Bits => $expn_n, $signif_n) |
                    unmask!(expn as Self::Bits => $signif_n) |
                    unmask!(signif as Self::Bits => ))
            }
        }

    }
}

mk_impl!(f32, u32, u8, u32, 8, 23);
mk_impl!(f64, u64, u16, u64, 11, 52);

#[cfg(test)]
mod tests {
    use super::Ieee754;

    #[test]
    fn all() {
        assert_eq!(0.0_f32.upto(0.0_f32).collect::<Vec<_>>(),
                   &[0.0]);

        assert_eq!(f32::recompose(false, 1, 1).upto(f32::recompose(false, 1, 10)).count(),
                   10);

        assert_eq!(f32::recompose(true, 0, 10).upto(f32::recompose(false, 0, 10)).count(),
                   21);

    }
}
#[cfg(all(test, feature = "unstable"))]
mod benches {
    use test::{Bencher, black_box};
    use super::Ieee754;

    #[bench]
    fn f32_iter_pos(b: &mut Bencher) {
        let (_, expn, _) = 1_f32.decompose();
        let end = f32::recompose(false, expn, 100);
        b.iter(|| black_box(1_f32).upto(end).count())
    }
    #[bench]
    fn f32_iter_over_zero(b: &mut Bencher) {
        let x = f32::recompose(false, 0, 20);
        b.iter(|| black_box(-x).upto(x).count())
    }
    #[bench]
    fn f64_iter_pos(b: &mut Bencher) {
        let (_, expn, _) = 1_f64.decompose();
        let end = f64::recompose(false, expn, 100);
        b.iter(|| black_box(1_f64).upto(end).count())
    }
    #[bench]
    fn f64_iter_over_zero(b: &mut Bencher) {
        let x = f64::recompose(false, 0, 20);
        b.iter(|| black_box(-x).upto(x).count())
    }
}
