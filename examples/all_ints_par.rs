#[cfg(feature = "rayon")]
extern crate rayon;

// base line comparison for the all example.
#[cfg(feature = "rayon")]
fn main() {
    extern crate criterion;
    use rayon::prelude::*;

    let expected =
        /* bits */ (1u64 << 32)
        - /* NaNs */ (1 << 24)
        - /* -0.0 */ 1
        + /* infinities */ 2;

    println!("count {}",
             (0..expected as u32).into_par_iter().map(criterion::black_box).count());
}

#[cfg(not(feature = "rayon"))]
fn main() {
    println!("all_par requires '--feature rayon'");
}
