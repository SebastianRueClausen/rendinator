use super::MemoryType;
use crate::{Device, RenderError};
use ash::vk::{self, DeviceSize};
use rendi_res::Res;
use std::cell::Cell;
use std::fmt;
use std::ptr::NonNull;

#[derive(Clone, Copy)]
struct RawMapped {
    count: usize,
    ptr: NonNull<u8>,
}

pub struct MemoryBlock {
    pub(crate) handle: vk::DeviceMemory,
    ty: MemoryType,
    size: DeviceSize,
    device: Res<Device>,
    mapped: Cell<Option<RawMapped>>,
}

impl MemoryBlock {
    pub(crate) fn new(
        device: Res<Device>,
        ty: MemoryType,
        size: DeviceSize,
    ) -> Result<Self, RenderError> {
        let info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(ty.index());
        let handle = unsafe { device.handle().allocate_memory(&info, None)? };
        Ok(Self {
            mapped: Cell::new(None),
            handle,
            device,
            size,
            ty,
        })
    }

    pub(crate) fn get_mapped(&self) -> Result<NonNull<u8>, RenderError> {
        let mapped = if let Some(mut mapped) = self.mapped.take() {
            mapped.count += 1;
            mapped
        } else {
            // SAFETY: `map_memory` should return an error if the pointer isn't valid.
            let ptr = unsafe {
                let flags = vk::MemoryMapFlags::empty();
                self.device
                    .handle()
                    .map_memory(self.handle, 0, self.size, flags)
                    .map(|ptr| NonNull::new_unchecked(ptr as *mut u8))?
            };
            RawMapped { ptr, count: 1 }
        };

        self.mapped.set(Some(mapped));

        Ok(mapped.ptr)
    }

    pub(crate) fn give_back_mapped(&self) {
        let Some(mut mapped) = self.mapped.take() else {
            return;
        };
        mapped.count -= 1;
        if mapped.count == 0 {
            unsafe {
                self.device.handle().unmap_memory(self.handle);
            }
        } else {
            self.mapped.set(mapped.into());
        }
    }

    /// Returns the size of the memory block.
    #[must_use]
    pub fn size(&self) -> DeviceSize {
        self.size
    }

    /// Returns the [`MemoryType`] of the memory block.
    #[must_use]
    pub fn memory_type(&self) -> MemoryType {
        self.ty
    }
}

impl PartialEq for MemoryBlock {
    fn eq(&self, other: &Self) -> bool {
        self.handle.eq(&other.handle)
    }
}

impl Eq for MemoryBlock {}

impl fmt::Debug for MemoryBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryBlock")
            .field("size", &self.size)
            .field("type", &self.ty)
            .field("is_mapped", &self.mapped.get().is_some())
            .finish()
    }
}
