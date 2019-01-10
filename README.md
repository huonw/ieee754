# `ieee754`

[![Build Status](https://travis-ci.org/huonw/ieee754.png)](https://travis-ci.org/huonw/ieee754) [![codecov](https://codecov.io/gh/huonw/ieee754/branch/master/graph/badge.svg)](https://codecov.io/gh/huonw/ieee754)

Low-level manipulations of IEEE754 floating-point numbers.

This library includes:

- ulp (units in the last place, representing the resolution of a
  float),
- `nextafter`,
- the IEEE-754 `totalOrder` predicate for doing `Ord::cmp`-like
  comparisons on floats,
- an iterator over every floating point value in a range

[Documentation](http://docs.rs/ieee754),
[crates.io](https://crates.io/crates/ieee754).
