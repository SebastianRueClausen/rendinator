use std::cell::Cell;
use std::rc::Rc;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::{alloc, borrow, ops, ptr, fmt};

/// Storage interface for the [`Res`] type.
pub(crate) trait ResStorage {
    unsafe fn dealloc(&self, ptr: ptr::NonNull<()>);
}

/// Allocator for allocating [`Res`] in seperate heap allocations.
struct Heap {
    layout: alloc::Layout,
}

impl ResStorage for Heap {
    unsafe fn dealloc(&self, ptr: ptr::NonNull<()>) {
        unsafe {
            alloc::dealloc(ptr.as_ptr() as _, self.layout);
        }
    }
}

/// The shared data of [´Res´]'s.
pub(crate) struct ResInner<T: ?Sized> {
    /// The amount of [`Res`]'s pointing to this object.
    ///
    /// Should never be 0.
    ref_count: Cell<usize>,

    /// A ref-counted reference to the block where this is allocated.
    ///
    /// This is used to make a deallocation and keeps the block alive long enough.
    alloc: Rc<dyn ResStorage>,

    pub(crate) val: T,
}

impl<T> ResInner<T> {
    #[inline]
    pub(crate) fn new(alloc: Rc<dyn ResStorage>, val: T) -> Self {
        let ref_count = Cell::new(1);

        Self {
            ref_count,
            alloc,
            val,
        }
    }
}

impl<T: ?Sized> ResInner<T> {
    #[inline(always)]
    pub(crate) fn inc_ref_count(&self) {
        self.ref_count.set(self.ref_count() + 1);
    }

    #[inline(always)]
    pub(crate) fn dec_ref_count(&self) {
        self.ref_count.set(self.ref_count() - 1);
    }

    #[inline(always)]
    pub(crate) fn ref_count(&self) -> usize {
        self.ref_count.get()
    }

    #[inline(always)]
    pub(crate) fn alloc(&self) -> Rc<dyn ResStorage> {
        self.alloc.clone()
    }
}

/// A reference counted pointer. 'Res' is short for 'Resource'.
///
/// The purpose of [`Res`] is to provide an alternative to [`Rc<T>`][`std::rc::Rc`] for objects
/// with shared ownership.
///
/// The main difference to [`Rc<T>`][`std::rc::Rc`] is that [`Res`] can be backed by different
/// allocators, such as slab or bump allocators.
///
/// # Examples
///
/// ```
/// use rendi_res::slab::Slabs;
/// use rendi_res::res::Res;
///
/// // In a seperate allocation.
/// let a = Res::new_heap('a');
///
/// let mut slabs = Slabs::new();
///
/// // From a slab allocator.
/// let b = slabs.alloc('b');
///
/// assert_eq!(*a, 'a');
/// assert_eq!(*b, 'b');
/// ```
pub struct Res<T: ?Sized> {
    pub(crate) inner: ptr::NonNull<ResInner<T>>,
}

impl<T> Res<T> {
    /// Create a new [`Res`] in a seperate heap allocation. This is almost exactly like [`Rc`],
    /// except a bit slower.
    ///
    /// ```
    /// use rendi_res::res::Res;
    ///
    /// let number = Res::new_heap(123);
    ///
    /// assert_eq!(*number, 123);
    /// ```
    #[inline]
    pub fn new_heap(val: T) -> Self {
        let layout = alloc::Layout::new::<ResInner<T>>();

        let inner = unsafe {
            let ptr = alloc::alloc(layout);

            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }

            ptr::NonNull::new_unchecked(ptr).cast()
        };

        let alloc = Rc::new(Heap { layout });

        unsafe {
            ptr::write(inner.as_ptr(), ResInner::new(alloc, val));
        }

        Self { inner }
    }
}

impl<T: ?Sized> Res<T> {
    #[inline]
    pub(crate) const fn new(inner: ptr::NonNull<ResInner<T>>) -> Self {
        Self { inner }
    }

    #[inline(always)]
    fn inner(&self) -> &ResInner<T> {
        unsafe { self.inner.as_ref() }
    }

    /// Get the number of references to the same resource.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_res::slab::Slabs;
    /// use rendi_res::res::Res;
    ///
    /// let mut slabs = Slabs::new();
    ///
    /// let a = slabs.alloc(0);
    /// assert_eq!(Res::ref_count(&a), 1);
    ///
    /// let b = a.clone();
    /// assert_eq!(Res::ref_count(&b), 2);
    /// ```
    #[inline]
    pub fn ref_count(this: &Self) -> usize {
        this.inner().ref_count()
    }

    /// Returns `true` if the two `Rc`s point to the same allocation.
    ///
    /// ```
    /// use rendi_res::res::Res;
    ///
    /// let one = Res::new_heap(1);
    /// let other_one = one.clone();
    ///
    /// let two = Res::new_heap(2);
    ///
    /// assert!(Res::ptr_eq(&one, &other_one));
    /// assert!(!Res::ptr_eq(&one, &two));
    /// ```
    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.inner.as_ptr() == other.inner.as_ptr()
    }
}

impl<T> Clone for Res<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        self.inner().inc_ref_count();

        Self { inner: self.inner }
    }
}

impl<T: ?Sized> Drop for Res<T> {
    fn drop(&mut self) {
        self.inner().dec_ref_count();

        if Self::ref_count(self) == 0 {
            let alloc = self.inner().alloc();

            // SAFETY: There is no longer any references to `self.inner` and can therefore be
            // dropped safely.
            unsafe {
                ptr::drop_in_place(self.inner.as_ptr());
                alloc.dealloc(self.inner.cast());
            }
        }
    }
}

impl<T: ?Sized> ops::Deref for Res<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &self.inner.as_ref().val }
    }
}

impl<T: ?Sized> borrow::Borrow<T> for Res<T> {
    fn borrow(&self) -> &T {
        self
    }
}

impl<T: ?Sized> AsRef<T> for Res<T> {
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T: ?Sized + PartialEq> PartialEq for Res<T> {
    fn eq(&self, other: &Self) -> bool {
        (**self) == (**other)
    }
}

impl<T: ?Sized + Eq> Eq for Res<T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for Res<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl<T: ?Sized + Ord> Ord for Res<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: ?Sized + Hash> Hash for Res<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for Res<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Res<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: ?Sized> fmt::Pointer for Res<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&(&**self as *const T), f)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::slab::Slabs;

    #[test]
    fn heap() {
        for _ in 0..1000 {
            let _ = Res::new_heap(0);
        }
    }

    #[test]
    fn drop() {
        let mut slabs = Slabs::new();

        static mut DROP_COUNT: usize = 0;

        struct DropMark(usize);
        impl Drop for DropMark {
            fn drop(&mut self) {
                unsafe {
                    DROP_COUNT += 1;
                }
            }
        }

        {
            let _a = slabs.alloc(DropMark(0));
            let _b = slabs.alloc(DropMark(0));
        }

        unsafe {
            assert_eq!(DROP_COUNT, 2);
        }

        {
            let _a = Res::new_heap(DropMark(0));
            let _b = Res::new_heap(DropMark(0));
        }

        unsafe {
            assert_eq!(DROP_COUNT, 4);
        }
    }
}
