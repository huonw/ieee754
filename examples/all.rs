#![cfg_attr(feature = "unstable", feature(test))]

#[cfg(feature = "unstable")]
extern crate test;

extern crate ieee754;
use ieee754::Ieee754;
use std::f32 as f;

#[cfg(feature = "unstable")]
fn main() {
    let count = f::NEG_INFINITY.upto(f::INFINITY).map(test::black_box).count();
    let expected =
        /* bits */ (1u64 << 32)
        - /* NaNs */ (1 << 24)
        - /* -0.0 */ 1
        + /* infinities */ 2;

    assert_eq!(count, expected as usize);
    println!("there are {} non-NaN floats", count);
}

#[cfg(not(feature = "unstable"))]
fn main() {
    println!("compile with --features unstable")
}
