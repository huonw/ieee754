use core::usize;
use core::fmt;
use {Bits, Ieee754};

/// An iterator over floating point numbers, created by `Ieee754::upto`.
#[derive(Clone, Eq, PartialEq)]
pub struct Iter<T: Ieee754> {
    from: T::Bits,
    to: T::Bits,
    done: bool
}
pub fn new_iter<T: Ieee754>(from: T, to: T) -> Iter<T> {
    Iter { from: from.bits(), to: to.bits(), done: false }
}

fn u64_to_size_hint(x: u64) -> (usize, Option<usize>) {
    if x <= usize::MAX as u64 {
        let d = x as usize;
        (d, Some(d))
    } else {
        (usize::MAX, None)
    }
}

impl<T: Ieee754> Iter<T> {
    fn len(&self) -> u64 {
        let (neg, pos) = self.split_by_sign();
        neg.len() + pos.len()
    }

    fn split_by_sign(&self) -> (SingleSignIter<T, Negative>, SingleSignIter<T, Positive>) {
        let negative = !self.done && self.from.high();
        let positive = !self.done && !self.to.high();

        let neg_start = self.from;
        let pos_end = self.to.next();

        let (neg_end, pos_start) = match (negative, positive) {
            (true, true) => (T::Bits::imin(), T::Bits::zero()),
            // self is a range with just one sign, so one side is
            // empty (has start == end)
            (false, true) => (neg_start, self.from),
            (true, false) => (self.to.prev(), pos_end),
            // self is done, so both sides are empty
            (false, false) => (neg_start, pos_end),
        };

        (
            SingleSignIter { from: neg_start, to: neg_end, _sign: Negative },
            SingleSignIter { from: pos_start, to: pos_end, _sign: Positive },
        )
    }
}

impl<T: Ieee754> Iterator for Iter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.done { return None }

        let ret = self.from;
        self.from = match ret.high() {
            // sign true => negative => the bit representation needs
            // to go down
            true => {
                let prev = ret.prev();
                if prev == T::Bits::imin() {
                    prev.flip_high()
                } else {
                    prev
                }
            }
            // sign false => positive => the bit representation needs
            // to go up
            false => ret.next()
        };

        if ret == self.to {
            self.done = true;
        }
        return Some(T::from_bits(ret))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        u64_to_size_hint(self.len())
    }

    // internal iteration optimisations:
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where F: FnMut(B, Self::Item) -> B
    {
        let (neg, pos) = self.split_by_sign();
        let next = neg.fold(init, &mut f);
        pos.fold(next, f)
    }
}

impl<T: Ieee754> DoubleEndedIterator for Iter<T> {
    fn next_back(&mut self) -> Option<T> {
        if self.done { return None }

        let ret = self.to;
        self.to = if ret.high() {
            ret.next()
        } else {
            if ret == T::Bits::zero() {
                ret.flip_high().next()
            } else {
                ret.prev()
            }
        };

        if ret == self.from {
            self.done = true
        }
        return Some(T::from_bits(ret))
    }
}

impl<T: Ieee754 + fmt::Debug> fmt::Debug for Iter<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut dbg = f.debug_struct("Iter");
        if self.done {
            dbg.field("done", &true);
        } else {
            dbg.field("from", &T::from_bits(self.from))
                .field("to", &T::from_bits(self.to));
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
struct Positive;
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

struct SingleSignIter<T: Ieee754, S: Sign> {
    from: T::Bits,
    to: T::Bits,
    _sign: S,
}

impl<T: Ieee754, S: Sign> SingleSignIter<T, S> {
    fn len(&self) -> u64 {
        S::dist(self.from, self.to)
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

    fn fold<B, F>(self, mut value: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let SingleSignIter { mut from, to, .. } = self;
        while from != to {
            value = f(value, T::from_bits(from));
            from = S::to_pos_inf(from);
        }
        value
    }
}
