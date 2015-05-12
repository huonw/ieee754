#![cfg_attr(feature = "unstable", feature(test))]

#[cfg(feature = "unstable")]
extern crate test;

// base line comparison for the all example.
#[cfg(feature = "unstable")]
fn main() {
    let expected =
        /* bits */ (1u64 << 32)
        - /* NaNs */ (1 << 24)
        - /* -0.0 */ 1
        + /* infinities */ 2;

    println!("count {}", (0..expected as usize).map(test::black_box).count());
}

#[cfg(not(feature = "unstable"))]
fn main() {
    println!("compile with --features unstable")
}
