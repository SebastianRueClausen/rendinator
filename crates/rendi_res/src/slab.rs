use std::cell::Cell;
use std::rc::{Rc, Weak};
use std::{array, mem, ptr};

use crate::res::{Res, ResInner, ResStorage};

type MaskType = u64;

#[derive(Default)]
struct Mask {
    mask: Cell<MaskType>,
}

impl Mask {
    #[cfg(test)]
    fn space_left(&self) -> u32 {
        self.mask.get().count_zeros() as u32
    }

    #[inline]
    fn is_full(&self) -> bool {
        self.mask.get() == MaskType::MAX
    }

    #[inline]
    fn alloc(&self) -> Option<u32> {
        if self.is_full() {
            None
        } else {
            // Find the first bit set to `0`.
            let mask = self.mask.get();
            let index = mask.trailing_ones();

            // Clear the bit.
            self.mask.set(mask | (1 << index));

            Some(index)
        }
    }

    #[inline]
    fn dealloc(&self, index: u32) {
        let mask = self.mask.get();

        self.mask.set(mask & !(1 << index));
    }
}

pub(crate) struct SlabBlock<T> {
    memory: Box<[mem::MaybeUninit<T>; MaskType::BITS as usize]>,
    mask: Mask,
}

impl<T> SlabBlock<T> {
    #[must_use]
    pub(crate) fn new() -> Self {
        let memory = Box::new(array::from_fn(|_| mem::MaybeUninit::uninit()));

        let mask = Mask::default();

        Self { memory, mask }
    }

    /// The amount of space left in the block. i.e. the amount of allocations left before running
    /// out of space.
    #[cfg(test)]
    pub(crate) fn space_left(&self) -> u32 {
        self.mask.space_left()
    }

    /// Returns true if the block is full.
    #[inline]
    pub(crate) fn is_full(&self) -> bool {
        self.mask.is_full()
    }

    #[inline]
    fn get(&self, index: u32) -> &mem::MaybeUninit<T> {
        &self.memory[index as usize]
    }

    /// Allocate object of type `T`.
    ///
    /// # Safety
    ///
    /// The value of the returned pointer is uninitialized, and thus the value may be malformed.
    #[must_use]
    #[inline]
    pub(crate) fn alloc(&self) -> Option<ptr::NonNull<T>> {
        assert_ne!(mem::size_of::<T>(), 0, "can't alloc type with a size of 0");

        self.mask
            .alloc()
            .map(|index| unsafe { self.get(index).assume_init_ref().into() })
    }

    /// dealloc the memory at `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must point to an object currently allocated in the block. Reading the value of `ptr`
    /// after calling this function is undefined behaviour.
    ///
    pub(crate) unsafe fn dealloc_by_ptr(&self, ptr: ptr::NonNull<T>) {
        assert_ne!(
            mem::size_of::<T>(),
            0,
            "can't dealloc type with a size of 0"
        );

        let index = ptr
            .as_ptr()
            .cast::<mem::MaybeUninit<T>>()
            .offset_from(self.memory.as_ptr());

        debug_assert!(
            index >= 0 && index < self.memory.len() as isize,
            "`val` lies outside range of block, index {index}",
        );

        self.mask.dealloc(index as u32);
    }
}

impl<T> ResStorage for SlabBlock<T> {
    unsafe fn dealloc(&self, ptr: ptr::NonNull<()>) {
        self.dealloc_by_ptr(ptr.cast());
    }
}

impl<T> Default for SlabBlock<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) type ResBlock<T> = SlabBlock<ResInner<T>>;

/// Slab allocator for the [`Res`] pointer.
///
/// A slab allocator is best used for allocating many objects of the same type, with potentially
/// different lifetimes.
///
/// The memory of destroyed objects is immediately reclaimed and fragmentation is unlikely to
/// happen.
///
/// # Examples
///
/// ```
/// use rendi_res::slab::Slabs;
///
/// let mut slabs = Slabs::new();
/// let mut numbers: Vec<_> = (0..100u32)
///     .map(|n| slabs.alloc(n))
///     .collect();
///
/// numbers.retain(|n| **n % 2 == 0);
///
/// assert_eq!(numbers.len(), 50);
/// ```
pub struct Slabs<T> {
    blocks: Vec<Weak<ResBlock<T>>>,
    block: Rc<ResBlock<T>>,
}

impl<T: 'static> Default for Slabs<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: 'static> Slabs<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            block: Rc::new(ResBlock::default()),
        }
    }

    /// Allocate a new `Res<T>` and initialize it with `init`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_res::slab::Slabs;
    ///
    /// let mut slabs = Slabs::new();
    ///
    /// let val = slabs.alloc_with(|| (1, 2, 3));
    ///
    /// assert_eq!(*val, (1, 2, 3));
    /// ```
    #[must_use]
    pub fn alloc_with<F>(&mut self, init: F) -> Res<T>
    where
        F: FnOnce() -> T,
    {
        let ptr = loop {
            let Some(ptr) = self.block.alloc() else {
                let mut new_block = None;

                // Both find a new potential block and remove all empty blocks.
                self.blocks.retain(|block| block
                    .upgrade()
                    .map_or(false, |block| {
                        if !block.is_full() {
                            new_block = block.into();
                        }

                        true
                    })
                );

                if let Some(block) = new_block {
                    self.block = block;
                } else {
                    let block = mem::take(&mut self.block);

                    self.blocks.push(Rc::downgrade(&block));
                }

                continue;
            };

            break ptr;
        };

        let inner = ResInner::new(self.block.clone(), init());

        // SAFETY: `ptr` is valid.
        unsafe {
            ptr.as_ptr().write(inner);
        }

        Res::new(ptr)
    }

    /// Allocate a new `Res<T>` and move `val` into it.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_res::slab::Slabs;
    ///
    /// let mut slabs = Slabs::new();
    ///
    /// let a = slabs.alloc('a');
    /// let b = slabs.alloc('b');
    ///
    /// assert_eq!(*a, 'a');
    /// assert_eq!(*b, 'b');
    /// ```
    #[inline]
    #[must_use]
    pub fn alloc(&mut self, val: T) -> Res<T> {
        self.alloc_with(|| val)
    }
}

#[cfg(test)]
mod test_mask {
    use super::*;

    #[test]
    fn alloc() {
        let mask = Mask::default();

        assert_eq!(mask.alloc(), Some(0));
        assert_eq!(mask.alloc(), Some(1));

        assert_eq!(mask.space_left(), MaskType::BITS - 2);
    }

    #[test]
    fn dealloc() {
        let mask = Mask::default();

        let index = mask.alloc().unwrap();
        mask.dealloc(index);

        assert_eq!(mask.space_left(), MaskType::BITS);
        assert_eq!(mask.alloc(), Some(0));
    }
}

#[cfg(test)]
mod test_slab_block {
    use super::*;

    #[test]
    fn alloc() {
        let slabs = SlabBlock::<u32>::new();

        let _ = slabs.alloc().unwrap();
        let _ = slabs.alloc().unwrap();

        assert_eq!(slabs.space_left(), MaskType::BITS - 2);
    }

    #[test]
    fn dealloc() {
        let slabs = SlabBlock::<u32>::new();

        let value = slabs.alloc().unwrap();
        let space = slabs.space_left();

        unsafe {
            slabs.dealloc_by_ptr(value);
        }

        assert_eq!(slabs.space_left(), space + 1);
    }
}

#[cfg(test)]
mod test_slabs {
    use super::*;

    #[test]
    fn alloc() {
        let mut slabs = Slabs::new();

        let a = slabs.alloc('a');
        let b = slabs.alloc('b');

        assert_eq!(*a, 'a');
        assert_eq!(*b, 'b');
    }
}
