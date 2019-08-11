#[cfg(feature = "rayon")]
extern crate rayon;
#[cfg(feature = "rayon")]
extern crate ieee754;

#[cfg(feature = "rayon")]
fn main() {
    extern crate criterion;
    use std::f32 as f;

    use ieee754::Ieee754;
    use rayon::prelude::*;

    let count = f::NEG_INFINITY.upto(f::INFINITY).into_par_iter().map(criterion::black_box).count();
    let expected =
        /* bits */ (1u64 << 32)
        - /* NaNs */ (1 << 24)
        - /* -0.0 */ 1
        + /* infinities */ 2;

    assert_eq!(count, expected as usize);
    println!("there are {} non-NaN floats", count);
}

#[cfg(not(feature = "rayon"))]
fn main() {
    println!("all_par requires '--features rayon'");
}
