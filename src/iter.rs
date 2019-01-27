use core::usize;
use {Bits, Ieee754};

/// An iterator over floating point numbers, created by `Ieee754::upto`.
pub struct Iter<T: Ieee754> {
    from: T::Bits,
    to: T::Bits,
    done: bool
}
pub fn new_iter<T: Ieee754>(from: T, to: T) -> Iter<T> {
    Iter { from: from.bits(), to: to.bits(), done: false }
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
                if prev.is_imin() {
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
        if self.done {
            return (0, Some(0))
        }

        fn position<B: Bits>(x: B) -> i64 {
            let sign = x.high();
            let abs = x.clear_high().as_u64() as i64;
            if sign {
                -abs
            } else {
                abs
            }
        }

        let from_key = position(self.from);
        let to_key = position(self.to);

        let distance = (to_key - from_key + 1) as u64;
        if distance <= usize::MAX as u64 {
            let d = distance as usize;
            (d, Some(d))
        } else {
            (usize::MAX, None)
        }
    }
}
impl<T: Ieee754> DoubleEndedIterator for Iter<T> {
    fn next_back(&mut self) -> Option<T> {
        if self.done { return None }

        let ret = self.to;
        self.to = if ret.high() {
            ret.next()
        } else {
            if ret.is_zero() {
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
