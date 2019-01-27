use core::ops::Try;
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

    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where F: FnMut(B, Self::Item) -> R, R: Try<Ok = B>
    {
        let mut value = init;
        if !self.done {
            let negative = self.from.high();
            let positive = !self.to.high();
            if negative {
                let end = if positive {
                    T::Bits::imin().next()
                } else {
                    self.to
                };

                while self.from != end {
                    let float = T::from_bits(self.from);
                    self.from = self.from.prev();
                    value = f(value, float)?;
                }

                if positive {
                    let float = T::from_bits(self.from);
                    self.from = self.from.prev().flip_high();
                    value = f(value, float)?;
                } else {
                    let float = T::from_bits(self.from);
                    self.from = self.from.prev();
                    self.done = true;
                    value = f(value, float)?;
                }
            }
            if positive {
                debug_assert!(!self.from.high());
                while self.from != self.to {
                    let float = T::from_bits(self.from);
                    self.from = self.from.next();
                    value = f(value, float)?;
                }

                let float = T::from_bits(self.from);
                self.from = self.from.next();
                self.done = true;
                value = f(value, float)?;
            }
        }
        R::from_ok(value)
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

    fn try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where F: FnMut(B, Self::Item) -> R, R: Try<Ok = B>
    {
        let mut value = init;
        if !self.done {
            let negative = self.from.high();
            let positive = !self.to.high();
            if positive {
                let end = if negative {
                    T::Bits::zero()
                } else {
                    self.from
                };

                while self.to != end {
                    let float = T::from_bits(self.to);
                    self.to = self.to.prev();
                    value = f(value, float)?;
                }

                if negative {
                    let float = T::from_bits(self.to);
                    self.to = self.to.flip_high().next();
                    value = f(value, float)?;
                } else {
                    let float = T::from_bits(self.to);
                    self.to = self.to.prev();
                    self.done = true;
                    value = f(value, float)?;
                }
            }
            if negative {
                debug_assert!(self.to.high());
                while self.from != self.to {
                    let float = T::from_bits(self.to);
                    self.to = self.to.next();
                    value = f(value, float)?;
                }

                let float = T::from_bits(self.to);
                self.to = self.to.prev();
                self.done = true;
                value = f(value, float)?;
            }
        }
        R::from_ok(value)
    }
}
