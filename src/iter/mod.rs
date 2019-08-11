use core::usize;
use core::fmt;
#[cfg(nightly)]
use core::ops::Try;
use {Bits, Ieee754};

#[cfg(feature = "rayon")]
pub mod rayon;

/// An iterator over floating point numbers, created by `Ieee754::upto`.
#[derive(Clone)]
pub struct Iter<T: Ieee754> {
    neg: SingleSignIter<T, Negative>,
    pos: SingleSignIter<T, Positive>
}
/// Create an iterator over the floating point values in [from, to]
/// (inclusive!)
pub fn new_iter<T: Ieee754>(from: T, to: T) -> Iter<T> {
    assert!(from <= to);

    let from_bits = from.bits();
    let to_bits = to.bits();
    let negative = from_bits.high();
    let positive = !to_bits.high();

    let neg_start = from_bits;
    let pos_end = to_bits.next();

    let (neg_end, pos_start) = match (negative, positive) {
        (true, true) => (T::Bits::imin(), T::Bits::zero()),
        // self is a range with just one sign, so one side is
        // empty (has start == end)
        (false, true) => (neg_start, from_bits),
        (true, false) => (to_bits.prev(), pos_end),
        // self is done, so both sides are empty
        (false, false) => (neg_start, pos_end),
    };

    Iter {
        neg: SingleSignIter { from: neg_start, to: neg_end, _sign: Negative },
        pos: SingleSignIter { from: pos_start, to: pos_end, _sign: Positive },
    }
}

fn u64_to_size_hint(x: u64) -> (usize, Option<usize>) {
    if x <= usize::MAX as u64 {
        let d = x as usize;
        (d, Some(d))
    } else {
        (usize::MAX, None)
    }
}

#[cfg(nightly)]
fn result_to_try<R: Try>(r: Result<R::Ok, R::Error>) -> R {
    match r {
        Ok(ok) => R::from_ok(ok),
        Err(error) => R::from_error(error)
    }
}

impl<T: Ieee754> Iter<T> {
    fn len(&self) -> u64 {
        self.neg.len() + self.pos.len()
    }

    fn done(&self) -> bool { self.neg.done() && self.pos.done() }

    /// A non-nightly only version of `Iterator::try_fold`.
    pub fn try_fold_result<B, F, E>(&mut self, init: B, mut f: F) -> Result<B, E>
    where F: FnMut(B, T) -> Result<B, E> {
        let next = self.neg.try_fold_result(init, &mut f)?;
        self.pos.try_fold_result(next, f)
    }

    /// A non-nightly only version of `Iterator::try_fold`.
    pub fn try_rfold_result<B, F, E>(&mut self, init: B, mut f: F) -> Result<B, E>
    where F: FnMut(B, T) -> Result<B, E> {
        let next = self.pos.try_rfold_result(init, &mut f)?;
        self.neg.try_fold_result(next, f)
    }
}

impl<T: Ieee754> Iterator for Iter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.neg.next().or_else(|| self.pos.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        u64_to_size_hint(self.len())
    }

    // internal iteration optimisations:
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where F: FnMut(B, Self::Item) -> B
    {
        let next = self.neg.fold(init, &mut f);
        self.pos.fold(next, f)
    }

    #[cfg(nightly)]
    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where F: FnMut(B, Self::Item) -> R,
          R: Try<Ok = B> {
        result_to_try(self.try_fold_result(init, |b, x| f(b, x).into_result()))
    }
}

impl<T: Ieee754> DoubleEndedIterator for Iter<T> {
    fn next_back(&mut self) -> Option<T> {
        self.pos.next_back().or_else(|| self.neg.next_back())
    }
}

// equal if they would iterate the same elements, even if the precise
// stored values are different.
impl<T: Ieee754> PartialEq for Iter<T> {
    fn eq(&self, other: &Self) -> bool {
        let mut self_ = self.clone();
        let mut other_ = other.clone();

        self_.next() == other_.next() && self_.next_back() == other_.next_back()
    }
}
impl<T: Ieee754> Eq for Iter<T> {}

impl<T: Ieee754> fmt::Debug for Iter<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut dbg = f.debug_struct("Iter");
        if self.done() {
            dbg.field("done", &true);
        } else {
            let mut iter = self.clone();
            let (from, to) = match (iter.next(), iter.next_back()) {
                (Some(f), Some(t)) => (f, t),
                (Some(f), None) => (f, f),
                _ => unreachable!()
            };
            dbg.field("from", &from)
                .field("to", &to);
        }
        dbg.finish()
    }
}

// Infrastructure for working with a side of the range that has a
// single sign, i.e. -x to -0.0 or +0.0 to +y. Loops using these types
// have no branches other than the loop condition, and compile done to
// a plain old C-style `for (x = ...; x != y; x++)` (or
// x--. statically determined).

trait Sign {
    fn to_pos_inf<B: Bits>(x: B) -> B;
    fn to_neg_inf<B: Bits>(x: B) -> B;

    fn dist<B: Bits>(from: B, to: B) -> u64;

}
#[derive(Clone, Eq, PartialEq, Debug)]
struct Positive;
#[derive(Clone, Eq, PartialEq, Debug)]
struct Negative;

impl Sign for Positive {
    fn to_pos_inf<B: Bits>(x: B) -> B { x.next() }
    fn to_neg_inf<B: Bits>(x: B) -> B { x.prev() }

    fn dist<B: Bits>(from: B, to: B) -> u64 {
        to.as_u64() - from.as_u64()
    }
}
impl Sign for Negative {
    fn to_pos_inf<B: Bits>(x: B) -> B { x.prev() }
    fn to_neg_inf<B: Bits>(x: B) -> B { x.next() }

    fn dist<B: Bits>(from: B, to: B) -> u64 {
        // sign-magnitude has the order reversed when negative
        from.as_u64() - to.as_u64()
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
struct SingleSignIter<T: Ieee754, S: Sign> {
    from: T::Bits,
    to: T::Bits,
    _sign: S,
}

impl<T: Ieee754, S: Sign> SingleSignIter<T, S> {
    fn len(&self) -> u64 {
        S::dist(self.from, self.to)
    }

    fn done(&self) -> bool {
        self.from == self.to
    }

    fn try_fold_result<B, F, E>(&mut self, mut value: B, mut f: F) -> Result<B, E>
    where F: FnMut(B, T) -> Result<B, E> {
        let SingleSignIter { mut from, to, .. } = *self;
        while from != to {
            let this = T::from_bits(from);
            from = S::to_pos_inf(from);
            value = f(value, this).map_err(|e| {
                // save the new state before leaving, since we might iterate again
                self.from = from;
                e
            })?;
        }
        self.from = from;
        Ok(value)
    }

    fn try_rfold_result<B, F, E>(&mut self, mut value: B, mut f: F) -> Result<B, E>
    where F: FnMut(B, T) -> Result<B, E> {
        let SingleSignIter { from, mut to, .. } = *self;
        while from != to {
            to = S::to_neg_inf(to);
            let this = T::from_bits(to);
            value = f(value, this).map_err(|e| {
                self.to = to;
                e
            })?;
        }
        self.to = to;
        Ok(value)
    }
}

impl<T: Ieee754, S: Sign> Iterator for SingleSignIter<T, S> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.from != self.to {
            let ret = self.from;
            self.from = S::to_pos_inf(ret);
            Some(T::from_bits(ret))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        u64_to_size_hint(self.len())
    }

    fn fold<B, F>(mut self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        enum Void {}

        match self.try_fold_result(init, |b, x| Ok::<_, Void>(f(b, x))) {
            Ok(result) => result,
            Err(_) => unreachable!()
        }
    }

    #[cfg(nightly)]
    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where F: FnMut(B, Self::Item) -> R,
          R: Try<Ok = B> {
        result_to_try(self.try_fold_result(init, |b, x| f(b, x).into_result()))
    }
}

impl<T: Ieee754, S: Sign> DoubleEndedIterator for SingleSignIter<T, S> {
    fn next_back(&mut self) -> Option<T> {
        if self.from != self.to {
            self.to = S::to_neg_inf(self.to);
            Some(T::from_bits(self.to))
        } else {
            None
        }
    }

    #[cfg(nightly)]
    fn try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where F: FnMut(B, Self::Item) -> R,
          R: Try<Ok = B> {
        result_to_try(self.try_fold_result(init, |b, x| f(b, x).into_result()))
    }
}
