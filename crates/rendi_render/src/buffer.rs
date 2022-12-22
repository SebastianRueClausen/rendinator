use crate::{mem, Access, BufferKind, Device, MemorySlice, Queue, RenderError};
use ash::vk::{self, DeviceSize};
use rendi_res::Res;
use smallvec::SmallVec;
use std::cell::Cell;
use std::hash::{Hash, Hasher};
use std::ptr::NonNull;
use std::slice;

type Queues = SmallVec<[Res<Queue>; 4]>;

#[derive(Debug)]
pub enum BufferUsage {
    /// The buffer can be used to in transfer commands.
    /// The `Access` indicates if the buffer the used as a source and/or destination.
    Transfer(Res<Queue>, Access),
    /// The buffer can be used as an indirect draw command buffer.
    IndirectDraw(Res<Queue>),
    /// The buffer can be used as a vertex buffer.
    Vertex(Res<Queue>),
    /// The buffer can be used as an index buffer.
    Index(Res<Queue>),
}

fn parse_usages(usages: &[BufferUsage]) -> (vk::BufferUsageFlags, Queues) {
    let mut usage_flags = vk::BufferUsageFlags::empty();
    let mut queues: Queues = usages
        .iter()
        .map(|usage| match usage {
            BufferUsage::Transfer(queue, access) => {
                if access.writes() {
                    usage_flags |= vk::BufferUsageFlags::TRANSFER_DST;
                }
                if access.reads() {
                    usage_flags |= vk::BufferUsageFlags::TRANSFER_SRC;
                }
                queue.clone()
            }
            BufferUsage::IndirectDraw(queue) => {
                usage_flags |= vk::BufferUsageFlags::INDIRECT_BUFFER;
                queue.clone()
            }
            BufferUsage::Vertex(queue) => {
                usage_flags |= vk::BufferUsageFlags::VERTEX_BUFFER;
                queue.clone()
            }
            BufferUsage::Index(queue) => {
                usage_flags |= vk::BufferUsageFlags::INDEX_BUFFER;
                queue.clone()
            }
        })
        .collect();

    queues.sort();
    queues.dedup();

    (usage_flags, queues)
}

#[derive(Debug, Clone)]
pub struct BufferInfo<'a> {
    pub device: Res<Device>,
    /// The usages of the buffer.
    pub usages: &'a [BufferUsage],
    /// The byte size of the buffer.
    pub size: DeviceSize,
    /// The kind of buffer.
    pub kind: BufferKind,
    /// The memory location where the buffer should be allocated.
    pub location: mem::MemoryLocation,
}

pub struct Buffer {
    pub(crate) handle: vk::Buffer,
    pub(crate) usage_flags: vk::BufferUsageFlags,
    queues: Queues,
    slice: MemorySlice,
    size: DeviceSize,
    kind: BufferKind,
    is_mapped: Cell<bool>,
    device: Res<Device>,
}

impl Buffer {
    pub fn new<A: mem::Allocator>(
        allocator: &mut A,
        info: BufferInfo,
    ) -> Result<Self, RenderError> {
        let device = info.device;
        let (mut usage_flags, queues) = parse_usages(info.usages);
        usage_flags |= info.kind.into();
        let handle = if queues.len() > 1 {
            let queue_indices: SmallVec<[_; 4]> =
                queues.iter().map(|queue| queue.family_index()).collect();
            let info = vk::BufferCreateInfo::builder()
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .usage(usage_flags)
                .size(info.size)
                .queue_family_indices(&queue_indices);
            unsafe { device.handle().create_buffer(&info, None)? }
        } else {
            let info = vk::BufferCreateInfo::builder()
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .usage(usage_flags)
                .size(info.size);
            unsafe { device.handle().create_buffer(&info, None)? }
        };
        let reqs = unsafe { device.handle().get_buffer_memory_requirements(handle) };
        let layout = unsafe { mem::MemoryLayout::new_unchecked(reqs.size, reqs.alignment) };
        let memory_type = device
            .physical()
            .get_memory_type(reqs.memory_type_bits, info.location)?;
        let slice = allocator.alloc(layout, memory_type)?;
        unsafe {
            device
                .handle()
                .bind_buffer_memory2(&[vk::BindBufferMemoryInfo::builder()
                    .memory(slice.block().handle)
                    .memory_offset(slice.start())
                    .buffer(handle)
                    .build()])?;
        }
        Ok(Self {
            is_mapped: Cell::new(true),
            size: info.size,
            kind: info.kind,
            handle,
            usage_flags,
            slice,
            queues,
            device,
        })
    }

    /// Returns the memory slice of the buffer.
    pub fn memory_slice(&self) -> &MemorySlice {
        &self.slice
    }

    /// Returns the kind of buffer.
    pub fn kind(&self) -> BufferKind {
        self.kind
    }

    /// Returns the byte size of the buffer.
    pub fn size(&self) -> DeviceSize {
        self.size
    }

    /// Returns the queues connected to the buffer.
    pub fn queues(&self) -> &[Res<Queue>] {
        &self.queues
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.handle().destroy_buffer(self.handle, None);
        }
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.handle.eq(&other.handle)
    }
}

impl Eq for Buffer {}

impl Hash for Buffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
    }
}

pub struct MappedMemory {
    ptr: NonNull<u8>,
    buffer: Res<Buffer>,
}

impl MappedMemory {
    /// Returns the mapped memory of `block` in `range`.
    /// Fails if the buffer is already mapped.
    pub fn new(buffer: Res<Buffer>) -> Result<Self, RenderError> {
        if buffer.is_mapped.get() {
            return Err(RenderError::BufferRemap);
        }

        buffer.is_mapped.set(true);

        let slice = buffer.memory_slice();
        let base = slice.block().get_mapped()?.as_ptr();

        // SAFETY: `get_mapped` should always return a valid pointer if it's `Ok`.
        let ptr = unsafe { NonNull::new_unchecked(base.offset(slice.start() as isize)) };

        Ok(Self { buffer, ptr })
    }

    /// Returns a mutable slice of the mapped memory.
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        // SAFETY: An invariant of this type is that the memory isn't aliased.
        unsafe {
            slice::from_raw_parts_mut(self.ptr.as_ptr(), self.buffer.memory_slice().len() as usize)
        }
    }

    /// Returns a slice of the mapped memory.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: An invariant of this type is that the memory isn't aliased, so no mutable access
        // to the same memory should exist.
        unsafe {
            slice::from_raw_parts(self.ptr.as_ptr(), self.buffer.memory_slice().len() as usize)
        }
    }

    /// Get the underlying memory slice.
    pub fn slice(&self) -> &MemorySlice {
        self.buffer.memory_slice()
    }
}

impl Drop for MappedMemory {
    fn drop(&mut self) {
        self.buffer.is_mapped.set(false);
        self.buffer.memory_slice().block().give_back_mapped();
    }
}
