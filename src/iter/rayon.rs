extern crate rayon;

use {Ieee754, Bits};
use super::{Iter, SingleSignIter};
use self::rayon::iter::*;
use self::rayon::iter::plumbing::*;

/// A parallel iterator over floating point numbers, created by
/// `into_par_iter` on `Iter`.
pub struct ParIter<T: Ieee754> {
    range: Iter<T>
}

struct IterProducer<T: Ieee754> {
    range: Iter<T>,
}

impl<T: Ieee754> IntoParallelIterator for Iter<T> {
    type Item = T;
    type Iter = ParIter<T>;

    fn into_par_iter(self) -> Self::Iter {
        ParIter { range: self }
    }
}

impl<T: Ieee754> ParallelIterator for ParIter<T> {
    type Item = T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge_unindexed(IterProducer { range: self.range }, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        self.range.size_hint().1
    }
}

impl<T: Ieee754> IterProducer<T> {
    // it's convenient to think of the split location separately to
    // actually doing the split.
    fn split_at(self, position: u64) -> (Self, Self) {
        // This splits the range in half by count, not by arithmetic
        // value.
        let len = self.range.len();
        assert!(position <= len);

        let Iter { neg, pos } = self.range;

        let (left, right) = if position < neg.len() {
            // the split happens within the negative range
            let mid = neg.from.offset(-(position as i64));
            (Iter {
                neg: SingleSignIter { to: mid, .. neg.clone() },
                // the positive range is empty
                pos: SingleSignIter { to: pos.from, .. pos.clone() },
            },
             Iter {
                 neg: SingleSignIter { from: mid, .. neg },
                 pos: pos
             })
        } else {
            // the split happens within the positive range (or at the boundary)
            let mid = pos.from.offset((position - neg.len()) as i64);
            (Iter {
                neg: neg.clone(),
                pos: SingleSignIter { to: mid, .. pos.clone() },
            },
             Iter {
                 // the negative range is empty
                 neg: SingleSignIter { from: neg.to, .. neg },
                 pos: SingleSignIter { from: mid, .. pos }
             })
        };
        // the ranges should be exactly touching, i.e. last of left is
        // immediately before first of right
        debug_assert!(left.clone().next_back().map(|x| x.next()) == right.clone().next());
        debug_assert_eq!(left.len(), position);
        debug_assert_eq!(right.len(), len - position);
        (IterProducer { range: left }, IterProducer { range: right })
    }
}

impl<T: Ieee754> UnindexedProducer for IterProducer<T> {
    type Item = T;

    fn split(self) -> (Self, Option<Self>) {
        if self.range.len() <= 1 {
            (self, None)
        } else {
            // left-bias by rounding up the position (e.g. if self.len()
            // == 5, we want (3, 2), not (2, 3)).
            let position = (self.range.len() + 1) / 2;
            let (left, right) = self.split_at(position);
            (left, Some(right))
        }
    }

    fn fold_with<F>(self, neg_folder: F) -> F
    where F: Folder<Self::Item> {
        // consume the two signs separately, to minimise branching
        let Iter { neg, pos } = self.range;

        let pos_folder = neg_folder.consume_iter(neg);
        if pos_folder.full() {
            pos_folder
        } else {
            pos_folder.consume_iter(pos)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::fmt::Debug;
    use core::{f32, f64};
    use std::vec::Vec;

    fn split_test<T: Ieee754 + Debug>(from: T, to: T, mid: T) {
        let range = from.upto(to);
        let len = range.len();
        let (left, right) = IterProducer { range: range }.split();
        let right = right.unwrap();
        assert_eq!(left.range.len(), (len + 1) / 2);
        assert_eq!(right.range.len(), len / 2);
        assert_eq!(left.range, from.upto(mid));
        assert_eq!(right.range, mid.next().upto(to));
    }
    #[test]
    fn test_split_zero() {
        split_test(0f32, 1.0, 8.131516e-20);
        split_test(-1f32, 0.0, -8.131516e-20);
        split_test(0f64, 1.0, 1.118751109680031e-154);
        split_test(-1f64, 0.0, -1.118751109680031e-154);

        split_test(0f32, f32::INFINITY, 1.5);
        split_test(f32::NEG_INFINITY, 0.0, -1.5);
        split_test(0f64, f64::INFINITY, 1.5);
        split_test(f64::NEG_INFINITY, 0.0, -1.5);
    }
    #[test]
    fn test_split_same_sign() {
        let x = f32::consts::PI;
        split_test(x, x.next(), x);
        split_test(x, x.next().next(), x.next());
        split_test(x, x.next().next().next(), x.next());
        split_test(-x.next(), -x, -x.next());
        split_test(-x.next().next(), -x, -x.next());
        split_test(-x.next().next().next(), -x, -x.next().next());

        let y = f64::consts::PI;
        split_test(y, y.next(), y);
        split_test(y, y.next().next(), y.next());
        split_test(y, y.next().next().next(), y.next());
        split_test(-y.next(), -y, -y.next());
        split_test(-y.next().next(), -y, -y.next());
        split_test(-y.next().next().next(), -y, -y.next().next());

        // one binade
        split_test(1f32, 2.0, 1.5);
        split_test(-2f32, -1.0, -1.5);
        split_test(1f64, 2.0, 1.5);
        split_test(-2f64, -1.0, -1.5);

        // cross binade (manually computed by average raw representations)
        split_test(1e-10_f32, 1e20, 100703.914);
        split_test(-1e20_f32, -1e-10, -100703.92);
        split_test(1e-10_f64, 1e20, 100703.91632713746);
        split_test(-1e20_f64, -1e-10, -100703.91632713747);
    }

    #[test]
    fn test_split_awkward_nonsplit() {
        fn t(x: f32) {
            split_test(x, x.next(), x);
        }
        t(-94618940000.0);
        t(-34652824.0);
        t(-34652790.0);
        t(-23.812748);
        t(-23.81274);
        t(-0.0000000000000069391306);
        t(-0.00000000000000000027740148);
        t(-0.00000000000000000027740024);
        t(-0.00000000000000000022259177);
        t(-0.00000000000000000022259156);
        t(-0.0000000000000000002195783);
        t(-0.0000000000000000000000000004100601);
        t(-0.00000000000000000000000000040458713);
        t(-0.0000000000000000000000000004045599);
    }

    #[test]
    fn test_split_different_signs_equal() {
        split_test(-1f32, 1.0, 0.0);
        split_test(-1f64, 1.0, 0.0);
        split_test(f32::NEG_INFINITY, f32::INFINITY, 0.0);
        split_test(f64::NEG_INFINITY, f64::INFINITY, 0.0);
    }

    #[test]
    fn test_split_different_signs_nonequal() {
        split_test(-2f32, 1.0, -5.877472e-39);
        split_test(-1f32, 2.0, 5.877472e-39);
        split_test(-2f64, 1.0, -1.1125369292536007e-308);
        split_test(-1f64, 2.0, 1.1125369292536007e-308);

        split_test(f32::NEG_INFINITY, 1.0, -1.0842022e-19);
        split_test(-1f32, f32::INFINITY, 1.0842022e-19);
        split_test(f64::NEG_INFINITY, 1.0, -1.4916681462400413e-154);
        split_test(-1.0, f64::INFINITY, 1.4916681462400413e-154);
    }
    fn split_fail_test<T: Ieee754 + Debug>(range: Iter<T>) {
        let (left, right) = IterProducer { range: range.clone() }.split();
        assert_eq!(left.range, range);
        assert!(right.is_none());
    }
    #[test]
    fn test_split_tiny() {
        fn t<T: Ieee754 + Debug>(x: T) {
            split_fail_test(x.upto(x));
        }
        t(0_f32);
        t(0_f64);

        t(1_f32);
        t(1_f64);

        t(-1_f32);
        t(-1_f64);
    }

    #[test]
    fn test_split_done() {
        fn t<T: Ieee754 + Debug>(from: T, to: T) {
            let mut range = from.upto(to);
            range.by_ref().for_each(|_| {});
            split_fail_test(range)
        }
        t(1_f32, 1.0001);
        t(1_f64, 1.00000000001);

        t(-1.0001_f32, -1.0);
        t(-1.00000000001_f64, -1.0);
    }

    #[test]
    fn test_iterate() {
        fn t<T: Ieee754 + Debug>(from: T, to: T) {
            // the elements yielded by the parallel iterate should be
            // the same as the non-parallel ones
            let mut parallel = from.upto(to).into_par_iter()
                .fold_with(
                    vec![],
                    |mut v, item| { v.push(item); v }
                )
                .reduce_with(|mut x, mut y| { x.append(&mut y); x })
                .unwrap();

            parallel.sort_by(|x, y| x.partial_cmp(&y).unwrap());

            let serial = from.upto(to).collect::<Vec<_>>();
            assert_eq!(parallel, serial);
        }

        t(1_f32, 1.0001);
        t(1_f64, 1.00000000001);
        t(-1.0001_f32, -1.0);
        t(-1.00000000001_f64, -1.0);
    }
}
