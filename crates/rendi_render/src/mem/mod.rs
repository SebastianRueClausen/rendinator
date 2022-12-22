pub mod block;
pub mod bump;

use crate::RenderError;
use ash::vk::{self, DeviceSize};
pub use block::MemoryBlock;
use rendi_res::Res;
use std::fmt;

/// Interface for a GPU memory allocator.
pub trait Allocator {
    fn alloc(&mut self, layout: MemoryLayout, ty: MemoryType) -> Result<MemorySlice, RenderError>;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum MemoryLocation {
    /// The memory is located in VRAM. It's faster to access but can't be memory mapped.
    Gpu = 0,
    /// The memory is located in RAM. It's very slow to access but can be memory mapped.
    Cpu = 1,
}

impl From<MemoryLocation> for vk::MemoryPropertyFlags {
    fn from(location: MemoryLocation) -> Self {
        match location {
            MemoryLocation::Gpu => Self::DEVICE_LOCAL,
            MemoryLocation::Cpu => {
                // NOTE: `HOST_COHERENT` is not guaranteed to be supported, but should work
                // basically everywhere.
                Self::DEVICE_LOCAL | Self::HOST_COHERENT
            }
        }
    }
}

impl fmt::Display for MemoryLocation {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let name = match self {
            MemoryLocation::Gpu => "GPU",
            MemoryLocation::Cpu => "CPU",
        };

        write!(fmt, "{name}")
    }
}

/// A range of GPU memory.
#[derive(Debug, Clone)]
pub struct MemorySlice {
    block: Res<MemoryBlock>,
    start: DeviceSize,
    end: DeviceSize,
}

impl MemorySlice {
    /// Returns a new memory range.
    ///
    /// # Panics
    ///
    /// * If `start > end`. The range should always be start to end.
    /// * If `end > block.size()`. The range should always be start to end.
    #[must_use]
    pub fn new(block: Res<MemoryBlock>, start: DeviceSize, end: DeviceSize) -> Self {
        assert!(start <= end, "memory range should always be start to end");
        assert!(end <= block.size(), "memory range goes past end of block");
        unsafe { Self::new_unchecked(block, start, end) }
    }

    /// Returns a new memory range without checking that it's valid.
    ///
    /// # Safety
    ///
    /// `start > end` should always be true.
    /// `end > block.size()` should always be true.
    #[must_use]
    pub unsafe fn new_unchecked(
        block: Res<MemoryBlock>,
        start: DeviceSize,
        end: DeviceSize,
    ) -> Self {
        Self { start, end, block }
    }

    /// Returns the start of the range which is the first byte inside the range.
    #[must_use]
    pub fn start(&self) -> DeviceSize {
        self.start
    }

    /// Returns the end of the range which is the first byte outside the range.
    #[must_use]
    pub fn end(&self) -> DeviceSize {
        self.end
    }

    /// Returns the block that owns the memory.
    #[must_use]
    pub fn block(&self) -> &Res<MemoryBlock> {
        &self.block
    }

    /// Returns the length of the range.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_render::MemoryRange;
    /// assert_eq!(MemoryRange::new(0, 4).len(), 4);
    /// ```
    pub fn len(&self) -> DeviceSize {
        self.end - self.start
    }
}

#[inline]
fn align_up_to(a: DeviceSize, alignment: DeviceSize) -> DeviceSize {
    ((a + alignment - 1) / alignment) * alignment
}

/// The layout of a block of memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MemoryLayout {
    size: DeviceSize,
    alignment: DeviceSize,
}

impl MemoryLayout {
    /// Returns a new layout if the alingment is valid.
    /// The alignment must be a power of 2.
    #[must_use]
    pub fn new(size: DeviceSize, alignment: DeviceSize) -> Option<Self> {
        if alignment.is_power_of_two() {
            unsafe { Some(Self::new_unchecked(size, alignment)) }
        } else {
            None
        }
    }

    /// Returns a new layout without checking that the alingment is valid.
    ///
    /// # Safety
    ///
    /// The alignment must be a power of 2.
    pub unsafe fn new_unchecked(size: DeviceSize, alignment: DeviceSize) -> Self {
        Self { size, alignment }
    }

    #[must_use]
    pub fn size(&self) -> DeviceSize {
        self.size
    }

    #[must_use]
    pub fn alignment(&self) -> DeviceSize {
        self.alignment
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
pub struct MemoryType {
    index: u32,
    location: MemoryLocation,
}

impl MemoryType {
    #[must_use]
    pub(crate) fn new(index: u32, location: MemoryLocation) -> Self {
        Self { index, location }
    }

    #[must_use]
    pub fn index(&self) -> u32 {
        self.index
    }

    #[must_use]
    pub fn location(&self) -> MemoryLocation {
        self.location
    }
}
