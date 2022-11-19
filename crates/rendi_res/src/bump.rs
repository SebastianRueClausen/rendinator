use std::cell::Cell;
use std::rc::Rc;
use std::{alloc, mem, ptr};

use crate::res::{Res, ResInner, ResStorage};

pub(crate) struct BumpBlock {
    base: ptr::NonNull<u8>,
    layout: alloc::Layout,
    offset: Cell<usize>,
}

impl BumpBlock {
    pub(crate) fn new(size: usize) -> Self {
        // SAFETY: The alignment guarenteed to be valid as we are allocating `u8`s.
        let layout = unsafe { alloc::Layout::from_size_align_unchecked(size, 1) };

        let offset = 0.into();

        // SAFETY: The allocation is checked and will panic if it fails.
        let base = unsafe {
            let Some(ptr) = ptr::NonNull::new(alloc::alloc(layout)) else {
                alloc::handle_alloc_error(layout);
            };

            ptr
        };

        Self {
            offset,
            layout,
            base,
        }
    }

    fn end_ptr(&self) -> *mut u8 {
        // SAFETY: `self.base` is valid which makes this safe.
        unsafe { self.base.as_ptr().add(self.layout.size()) }
    }

    fn offset(&self) -> usize {
        self.offset.get()
    }

    fn inc_offset(&self, amount: usize) {
        self.offset.set(self.offset() + amount);
    }

    pub(crate) fn alloc<T>(&self) -> Option<ptr::NonNull<T>> {
        let size = mem::size_of::<T>();
        let align = mem::align_of::<T>();

        let offset = self.offset.get();

        // SAFETY: This is some simple pointer arithmatics to find the base pointer of the
        // allocation and the new offset into the block. The output pointer is guarenteed to
        // aligned properly and valid.
        unsafe {
            let ptr = self.base.as_ptr().add(offset);
            let align_offset = ptr.align_offset(align);

            let start = ptr.add(align_offset);
            let end = start.add(size);

            (end <= self.end_ptr()).then(|| {
                self.inc_offset(align_offset + size);

                ptr::NonNull::new_unchecked(start.cast())
            })
        }
    }
}

impl ResStorage for BumpBlock {
    unsafe fn dealloc(&self, _ptr: ptr::NonNull<()>) {}
}

/// Bump allocator for the [`Res`] pointer.
///
/// The advantage of this allocator is that allocations are very fast and densly packed into blocks.
/// Allocating an object consists of simply offsetting a pointer and copying the data. This also
/// means that unlike [`Slabs`][crate::slab::Slabs], this allocator can allocate multiple different
/// types.
///
/// The caveat is that memory isn't freed until every object in a block is destroyed. The
/// bump allocator is therefore best for allocating objects with similar lifetimes.
///
/// # Examples
///
/// ```
/// use rendi_res::bump::Bump;
///
/// // Create new allocator with a block size of 1 kilobyte.
/// let mut bump = Bump::new(1024);
///
/// let twelve = bump.alloc(12);
/// let five = bump.alloc(5);
///
/// assert_eq!(*twelve + *five, 17);
/// ```
pub struct Bump {
    block: Rc<BumpBlock>,
    block_size: usize,
}

impl Default for Bump {
    /// Create a new allocator with a block size of 1 kilobyte.
    fn default() -> Self {
        Self::new(1024)
    }
}

impl Bump {
    /// Create a new bump allocator.
    ///
    /// `block_size` is the byte size of each block.
    #[must_use]
    pub fn new(block_size: usize) -> Self {
        Self {
            block: BumpBlock::new(block_size).into(),
            block_size,
        }
    }

    /// Returns the block size currently used when allocating new blocks.
    #[inline]
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Change the block size for future blocks.
    ///
    /// Note that this doesn't change the size of any previously allocated blocks.
    #[inline]
    pub fn change_block_size(&mut self, block_size: usize) {
        self.block_size = block_size;
    }

    /// Allocate a new `Res<T>` and initialize it with `init`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_res::bump::Bump;
    ///
    /// let mut bump = Bump::new(32);
    ///
    /// let val = bump.alloc_with(|| (1, 2, 3));
    ///
    /// assert_eq!(*val, (1, 2, 3));
    /// ```
    #[must_use]
    pub fn alloc_with<T, F>(&mut self, init: F) -> Res<T>
    where
        F: FnOnce() -> T,
    {
        let mut size = mem::size_of::<ResInner<T>>();

        let (block, ptr) = loop {
            if let Some(ptr) = self.block.alloc::<ResInner<T>>() {
                break (self.block.clone(), ptr);
            }

            // `self.block` is out of room.

            let mut block_size = self.block_size;
            while size > block_size {
                block_size *= 2;
            }

            // Rare (impossible?) case where the allocation fails even if the block size is
            // the the same size as the allocation because of aligment offsets.
            size = block_size + 1;

            self.block = BumpBlock::new(block_size).into();
        };

        let inner = ResInner::new(block, init());

        // SAFETY: `ptr` has just been allocated and is therefore valid.
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
    /// use rendi_res::bump::Bump;
    ///
    /// let mut bump = Bump::new(32);
    ///
    /// let a = bump.alloc('a');
    /// let b = bump.alloc('b');
    ///
    /// assert_eq!(*a, 'a');
    /// assert_eq!(*b, 'b');
    /// ```
    #[must_use]
    #[inline]
    pub fn alloc<T>(&mut self, val: T) -> Res<T> {
        self.alloc_with(|| val)
    }
}

#[cfg(test)]
mod test_bump {
    use super::*;

    #[test]
    fn stress() {
        let mut bump = Bump::new(1024);  
        let mut save = Vec::new();

        for i in 0..32 {
            let numbers: Vec<_> = (0..256)
                .map(|n| bump.alloc(n))
                .collect();

            if i % 2 == 0 {
                save.push(numbers);
            }
        }
    }

    #[test]
    fn big_alloc() {
        let mut bump = Bump::new(32);  
        let _ = bump.alloc([0x0; 1024]);
    }
}
