#[macro_use] extern crate criterion;
extern crate ieee754;

use criterion::{Criterion, Bencher, ParameterizedBenchmark, black_box};

use ieee754::Ieee754;

const SIZES: &[usize] = &[10, 100, 1_000, 1_000_000];

fn f32_iter(c: &mut Criterion) {
    type B = <f32 as Ieee754>::Bits;
    type S = <f32 as Ieee754>::Significand;

    let positive = |b: &mut Bencher, &i: &usize| {
        let end = f32::recompose(false, 0, i as S);
        b.iter(|| {
               assert_eq!(black_box(1_f32).upto(black_box(end))
                          .map(black_box).count(),
                          i + 1);
        })
    };
    let positive_ext = |b: &mut Bencher, &i: &usize| {
        let end = f32::recompose(false, 0, i as S);
        b.iter(|| {
            let mut count = 0;
            for val in black_box(1_f32).upto(black_box(end)) {
                black_box(val);
                count += 1;
            }
            assert_eq!(count, i + 1);
        })
    };
    let positive_find = |b: &mut Bencher, &i: &usize| {
        let end = f32::recompose(false, 0, i as S);
        let limit = black_box(1e9);
        b.iter(|| {
            assert_eq!(black_box(1_f32).upto(black_box(end))
                       .map(black_box).find(|a| *a > limit),
                       None);
        })
    };
    let over_zero = |b: &mut Bencher, &i: &usize| {
        let x = f32::recompose(false, -f32::exponent_bias(), (i / 2) as S);
        b.iter(|| {
               assert_eq!(black_box(-x).upto(black_box(x))
                          .map(black_box).count(),
                          i + 1);
        })
    };
    let over_zero_ext = |b: &mut Bencher, &i: &usize| {
        let x = f32::recompose(false, -f32::exponent_bias(), (i / 2) as S);
        b.iter(|| {
            let mut count = 0;
            for val in black_box(-x).upto(black_box(x)) {
                black_box(val);
                count += 1;
            }
            assert_eq!(count, i + 1);
        })
    };
    let over_zero_find = |b: &mut Bencher, &i: &usize| {
        let x = f32::recompose(false, -f32::exponent_bias(), (i / 2) as S);
        let limit = black_box(1e9);
        b.iter(|| {
            assert_eq!(black_box(-x).upto(black_box(x))
                       .map(black_box).find(|a| *a > limit),
                       None);
        })
    };
    let baseline = |b: &mut Bencher, &i: &usize| {
        b.iter(|| {
            assert_eq!((black_box(0 as B)..=black_box(i as B))
                       .map(black_box).count(),
                       i + 1);
        })
    };
    let baseline_ext = |b: &mut Bencher, &i: &usize| {
        b.iter(|| {
            let mut count = 0;

            for val in black_box(0 as B)..=black_box(i as B) {
                black_box(val);
                count += 1;
            }
            assert_eq!(count, i + 1);
        })
    };
    let baseline_find = |b: &mut Bencher, &i: &usize| {
        let limit = black_box(1_000_000_000);
        b.iter(|| {
            assert_eq!((black_box(0 as B)..=black_box(i as B))
                       .map(black_box).find(|a| *a > limit),
                       None);
        })
    };

    let internal = ParameterizedBenchmark::new("baseline", baseline, SIZES.to_owned())
        .with_function("positive", positive)
        .with_function("over_zero", over_zero);
    let external = ParameterizedBenchmark::new("baseline", baseline_ext, SIZES.to_owned())
        .with_function("positive", positive_ext)
        .with_function("over_zero", over_zero_ext);
    let find = ParameterizedBenchmark::new("baseline", baseline_find, SIZES.to_owned())
        .with_function("positive", positive_find)
        .with_function("over_zero", over_zero_find);

    c.bench("iter_f32_internal", internal);
    c.bench("iter_f32_external", external);
    c.bench("iter_f32_find", find);
}

fn f64_iter(c: &mut Criterion) {
    type B = <f64 as Ieee754>::Bits;
    type S = <f64 as Ieee754>::Significand;

    let positive = |b: &mut Bencher, &i: &usize| {
        let end = f64::recompose(false, 0, i as S);
        b.iter(|| {
               assert_eq!(black_box(1_f64).upto(black_box(end))
                          .map(black_box).count(),
                          i + 1);
        })
    };
    let positive_ext = |b: &mut Bencher, &i: &usize| {
        let end = f64::recompose(false, 0, i as S);
        b.iter(|| {
            let mut count = 0;
            for val in black_box(1_f64).upto(black_box(end)) {
                black_box(val);
                count += 1;
            }
            assert_eq!(count, i + 1);
        })
    };
    let positive_find = |b: &mut Bencher, &i: &usize| {
        let end = f64::recompose(false, 0, i as S);
        let limit = black_box(1e9);
        b.iter(|| {
            assert_eq!(black_box(1_f64).upto(black_box(end))
                       .find(|a| *a > limit),
                       None);
        })
    };
    let over_zero = |b: &mut Bencher, &i: &usize| {
        let x = f64::recompose(false, -f64::exponent_bias(), (i / 2) as S);
        b.iter(|| {
               assert_eq!(black_box(-x).upto(black_box(x))
                          .map(black_box).count(),
                          i + 1);
        })
    };
    let over_zero_ext = |b: &mut Bencher, &i: &usize| {
        let x = f64::recompose(false, -f64::exponent_bias(), (i / 2) as S);
        b.iter(|| {
            let mut count = 0;
            for val in black_box(-x).upto(black_box(x)) {
                black_box(val);
                count += 1;
            }
            assert_eq!(count, i + 1);
        })
    };
    let over_zero_find = |b: &mut Bencher, &i: &usize| {
        let x = f64::recompose(false, -f64::exponent_bias(), (i / 2) as S);
        let limit = black_box(1e9);
        b.iter(|| {
            assert_eq!(black_box(-x).upto(black_box(x))
                       .find(|a| *a > limit),
                       None);
        })
    };
    let baseline = |b: &mut Bencher, &i: &usize| {
        b.iter(|| {
            assert_eq!((black_box(0 as B)..=black_box(i as B))
                       .map(black_box).count(),
                       i + 1);
        })
    };
    let baseline_ext = |b: &mut Bencher, &i: &usize| {
        b.iter(|| {
            let mut count = 0;

            for val in black_box(0 as B)..=black_box(i as B) {
                black_box(val);
                count += 1;
            }
            assert_eq!(count, i + 1);
        })
    };
    let baseline_find = |b: &mut Bencher, &i: &usize| {
        let limit = black_box(1_000_000_000);
        b.iter(|| {
            assert_eq!((black_box(0 as B)..=black_box(i as B))
                       .find(|a| *a > limit),
                       None);
        })
    };

    let internal = ParameterizedBenchmark::new("baseline", baseline, SIZES.to_owned())
        .with_function("positive", positive)
        .with_function("over_zero", over_zero);
    let external = ParameterizedBenchmark::new("baseline", baseline_ext, SIZES.to_owned())
        .with_function("positive", positive_ext)
        .with_function("over_zero", over_zero_ext);
    let find = ParameterizedBenchmark::new("baseline", baseline_find, SIZES.to_owned())
        .with_function("positive", positive_find)
        .with_function("over_zero", over_zero_find);

    c.bench("iter_f64_internal", internal);
    c.bench("iter_f64_external", external);
    c.bench("iter_f64_find", find);
}

criterion_group!(benches, f32_iter, f64_iter);
criterion_main!(benches);
