use std::{mem, ptr};

use crate::res::{Res, ResInner};

pub struct DummyRes {
    inner: ptr::NonNull<ResInner<()>>,

    val: *mut (),
    drop: Option<unsafe fn(*mut ())>,
}

impl DummyRes {
    pub fn new<T>(res: Res<T>) -> Self {
        let drop = if mem::needs_drop::<T>() {
            Some(unsafe {
                mem::transmute::<unsafe fn(*mut T), unsafe fn(*mut ())>(
                    ptr::drop_in_place::<T>
                )
            })
        } else {
            None 
        };

        let val = unsafe {
            mem::transmute::<*mut T, *mut ()>(&mut (*res.inner.as_ptr()).val as *mut T)
        };

        let inner = res.inner.cast();

        mem::forget(res);

        Self {
            drop,
            inner,
            val,
        }
    }
}

impl Drop for DummyRes {
    fn drop(&mut self) {
        let inner = unsafe { self.inner.as_ref() };

        inner.dec_ref_count();

        if inner.ref_count() == 0 {
            let alloc = inner.alloc();

            if let Some(drop_ptr) = self.drop {
                unsafe { (drop_ptr)(self.val); }
            }

            unsafe {
                alloc.dealloc(self.inner.cast());
            }
        }
    }
}

