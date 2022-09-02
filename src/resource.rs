use anyhow::Result;
use glam::Vec3;
use ash::vk;

use std::{ops, slice, alloc, mem};
use std::rc::Rc;
use std::cell::{UnsafeCell, Cell};
use std::ptr::{self, NonNull};
use std::hash::{Hash, Hasher};

use crate::core::*;

type MemoryRange = ops::Range<vk::DeviceSize>;

fn range_length(range: &MemoryRange) -> vk::DeviceSize {
    range.end - range.start
}

/// A wrapper around a raw ptr into a mapped memory block.
pub struct MappedMemory {
    ptr: NonNull<u8>,
    size: vk::DeviceSize,

    block: Rc<MemoryBlock>,
}

impl MappedMemory {
    pub fn new(block: Rc<MemoryBlock>, range: MemoryRange) -> Result<Self> {
        let ptr = unsafe {
            NonNull::new_unchecked(
                block.get_mapped()?.as_ptr().offset(range.start as isize)
            )
        };

        let size = range_length(&range);

        Ok(MappedMemory { size, block, ptr, })

    }

    unsafe fn as_slice_range_unchecked(&self, range: MemoryRange) -> &mut [u8] {
        let start = self.ptr.as_ptr().offset(range.start as isize);
        let len = range_length(&range);

        slice::from_raw_parts_mut(start, len as usize)
    }

    unsafe fn as_slice(&self) -> &mut [u8] {
        slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size as usize)
    }

    /// Fill memory with the memory of `bytes`.
    ///
    /// # Panics
    ///
    /// * If the length of `bytes` isn't the same as the size of mapped memory.
    ///
    pub fn fill(&self, bytes: &[u8]) {
        if self.size != bytes.len() as vk::DeviceSize {
            panic!("bytes is not the same size as the mapped memory range");
        }
    
        unsafe { self.as_slice().copy_from_slice(bytes) };
    }

    pub fn fill_from_start(&self, bytes: &[u8]) {
        if self.size < bytes.len() as vk::DeviceSize {
            panic!("bytes is longer than the mapped memory range");
        }

        let slice = unsafe { self.as_slice() };

        for (src, dst) in bytes.iter().zip(slice.iter_mut()) {
            *dst = *src;
        }
    }

    /// Fill memory in `range` with the memory of `bytes`.
    ///
    /// # Panics
    ///
    /// * If the range doesn't fit into the mapped block.
    /// * If the length of `bytes` isn't the same as the span of range.
    ///
    pub fn fill_range(&self, range: MemoryRange, bytes: &[u8]) {
        if range_length(&range) != bytes.len() as vk::DeviceSize {
            panic!("bytes is not the same size as the range");
        }
       
        // The end should always be greater than start, but just to be sure.
        if range.start.max(range.end) > self.block.size() {
            panic!("range goes past the end of the mapped memory");
        }

        let dst = unsafe { self.as_slice_range_unchecked(range) };

        dst.copy_from_slice(bytes);
    }
}

/// A block of raw device memory.
#[derive(Clone)]
pub struct MemoryBlock {
    handle: vk::DeviceMemory,
    size: vk::DeviceSize,

    /// Holds pointer to start of the block if mapped, and the number of [`MappedMemory`] mapped to
    /// this buffer.
    mapped: Cell<Option<NonNull<u8>>>,

    device: Res<Device>,
}

impl MemoryBlock {
    fn new(device: Res<Device>, info: &vk::MemoryAllocateInfo) -> Result<Self> {
        let handle = unsafe { device.handle.allocate_memory(info, None)? };
        let size = info.allocation_size;
        let mapped = Cell::new(None);

        Ok(Self { device, mapped, handle, size })
    }

    fn get_mapped(&self) -> Result<NonNull<u8>> {
        let ptr = if let Some(mapped) = self.mapped.replace(None) {
            mapped
        } else {
            unsafe {
                self.device.handle
                    .map_memory(self.handle, 0, self.size, vk::MemoryMapFlags::empty())
                    .map(|ptr| NonNull::new_unchecked(ptr as *mut u8))?
            }
        };

        self.mapped.set(Some(ptr));

        Ok(ptr)
    }

    #[allow(dead_code)]
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }
}

impl Drop for MemoryBlock {
    fn drop(&mut self) {
        trace!("deallocating GPU memory block of {} bytes", self.size());

        unsafe { self.device.handle.free_memory(self.handle, None); }
    }
}

impl PartialEq for MemoryBlock {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl Eq for MemoryBlock {}

#[derive(Clone, Copy)]
pub struct BufferReq {
    pub usage: vk::BufferUsageFlags,
    pub size: u64,
}

impl Into<vk::BufferCreateInfo> for BufferReq {
    fn into(self) -> vk::BufferCreateInfo {
        vk::BufferCreateInfo::builder()
            .usage(self.usage)
            .size(self.size)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build()
    }
}

#[derive(Clone)]
pub struct Buffer {
    pub handle: vk::Buffer,
    pub block: Rc<MemoryBlock>,
    pub range: MemoryRange,

    device: Res<Device>,
}

impl Buffer {
    pub fn new(
        renderer: &Renderer,
        pool: &ResourcePool,
        memory_flags: vk::MemoryPropertyFlags,
        info: &BufferReq,
    ) -> Result<Res<Self>> {
        Self::from_raw(renderer.device.clone(), pool,  memory_flags, info)
    }

    pub fn from_raw<T: Into<vk::BufferCreateInfo> + Clone + Copy>(
        device: Res<Device>,
        pool: &ResourcePool,
        memory_flags: vk::MemoryPropertyFlags,
        req: &T,
    ) -> Result<Res<Self>> {
        let info = (*req).into();

        let pool = unsafe {
            let handle = device.handle.create_buffer(&info, None)?;
            let req = device.handle.get_buffer_memory_requirements(handle);

            let memory_type = device.physical
                .get_memory_type_index(req.memory_type_bits, memory_flags)
                .ok_or_else(|| anyhow!("no compatible memory type"))?;

            let (block, range) = pool.gpu_alloc(
                device.clone(),
                memory_type,
                req.size,
                req.alignment,
            )?;

            let range = range.start..range.start + info.size;

            device.handle.bind_buffer_memory(handle, block.handle, range.start)?;

            pool.alloc(Self { handle, range, block, device })
        };

        Ok(pool)
    }

    pub fn size(&self) -> vk::DeviceSize {
        range_length(&self.range)
    }

    pub fn get_mapped(&self) -> Result<MappedMemory> {
        MappedMemory::new(self.block.clone(), self.range.clone())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_buffer(self.handle, None); }
    }
}

#[derive(Clone, Copy)]
pub enum ImageKind {
    Texture,
    CubeMap,
}

#[derive(Clone, Copy)]
pub struct ImageReq {
    pub kind: ImageKind,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
}

impl Into<vk::ImageCreateInfo> for ImageReq {
    fn into(self) -> vk::ImageCreateInfo {
        let (array_layers, flags) = match self.kind {
            ImageKind::Texture => (1, vk::ImageCreateFlags::empty()),
            ImageKind::CubeMap => (6, vk::ImageCreateFlags::CUBE_COMPATIBLE),
        };
        vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .format(self.format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .extent(self.extent)
            .mip_levels(self.mip_levels)
            .samples(vk::SampleCountFlags::TYPE_1)
            .array_layers(array_layers)
            .flags(flags)
            .build()
    }
}

impl Into<vk::ImageViewCreateInfo> for ImageReq {
    fn into(self) -> vk::ImageViewCreateInfo {
        let (view_type, layer_count) = match self.kind {
            ImageKind::Texture => (vk::ImageViewType::TYPE_2D, 1),
            ImageKind::CubeMap => (vk::ImageViewType::CUBE, 6),
        };
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(self.mip_levels)
            .base_array_layer(0)
            .layer_count(layer_count);
        vk::ImageViewCreateInfo::builder()
            .view_type(view_type)
            .subresource_range(*subresource_range)
            .format(self.format)
            .build()
    }
}

#[derive(Clone)]
pub struct Image {
    pub handle: vk::Image,
    pub view: vk::ImageView,
    pub extent: vk::Extent3D,
    pub format: vk::Format,

    pub array_layers: u32,
    pub mip_levels: u32,

    // Layout may change.
    pub layout: Cell<vk::ImageLayout>,

    pub range: MemoryRange,
    pub block: Rc<MemoryBlock>,
}

impl Image {
    #[allow(dead_code)]
    pub fn size(&self) -> vk::DeviceSize {
        self.range.end - self.range.start
    }

    pub fn new(
        renderer: &Renderer,
        pool: &ResourcePool,
        memory_flags: vk::MemoryPropertyFlags,
        req: &ImageReq,
    ) -> Result<Res<Self>> {
        Self::from_raw(renderer.device.clone(), pool, memory_flags, req, req)
    }

    /// Create image from raw [`vk::ImageCreateInfo`] and [`vk::ImageViewCreateInfo`].
    pub fn from_raw<I, V>(
        device: Res<Device>,
        pool: &ResourcePool,
        memory_flags: vk::MemoryPropertyFlags,
        image_info: &I,
        view_info: &V,
    ) -> Result<Res<Self>>
    where
        I: Into<vk::ImageCreateInfo> + Clone + Copy,
        V: Into<vk::ImageViewCreateInfo> + Clone + Copy,
    {
        let device = device.clone();

        let image_info: vk::ImageCreateInfo = (*image_info).into();

        let (handle, req) = unsafe {
            let handle = device.handle.create_image(&image_info, None)?;
            let req = device.handle.get_image_memory_requirements(handle);

            (handle, req)
        };

        let memory_type = device.physical.get_memory_type_index(req.memory_type_bits, memory_flags)
            .ok_or_else(|| anyhow!("no compatible memory type"))?;

        let (block, range) = pool.gpu_alloc(
            device.clone(),
            memory_type,
            req.size,
            req.alignment,
        )?;

        unsafe {
            device.handle.bind_image_memory(handle, block.handle, range.start)?;
        }
       
        let view = unsafe {
            let mut view_info: vk::ImageViewCreateInfo = (*view_info).into();
            view_info.image = handle;

            device.handle.create_image_view(&view_info, None)?
        };

        Ok(pool.alloc(Image {
            mip_levels: image_info.mip_levels,
            layout: Cell::new(image_info.initial_layout),
            extent: image_info.extent,
            format: image_info.format,
            range: 0..req.size,
            array_layers: image_info.array_layers,
            handle,
            view,
            block,
        }))
    }

    pub fn layout(&self) -> vk::ImageLayout {
        self.layout.get()
    }

    pub fn extent(&self, mip_level: u32) -> vk::Extent3D {
        vk::Extent3D {
            width: self.extent.width >> mip_level,
            height: self.extent.height >> mip_level,
            depth: self.extent.depth,
        }
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.block.device.handle.destroy_image(self.handle, None);
            self.block.device.handle.destroy_image_view(self.view, None);
        }
    }
}

#[derive(Clone)]
pub struct TextureSampler {
    pub handle: vk::Sampler,
    device: Res<Device>,
}

impl TextureSampler {
    pub fn new(renderer: &Renderer) -> Result<Self> {
        let device = renderer.device.clone(); 
        let create_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(device.physical.properties.limits.max_sampler_anisotropy)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE);
        let handle = unsafe { device.handle.create_sampler(&create_info, None)? };
        Ok(Self { handle, device })
    }
}

impl Drop for TextureSampler {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_sampler(self.handle, None); }
    }
}

const CUBE_VERTICES: [Vec3; 36] = [
    Vec3::new(-1.0, 1.0, 1.0),
    Vec3::new(-1.0, -1.0, 1.0),
    Vec3::new(1.0, -1.0, 1.0),
    Vec3::new(1.0, -1.0, 1.0),
    Vec3::new(1.0, 1.0, 1.0),
    Vec3::new(-1.0, 1.0, 1.0),

    Vec3::new(-1.0, -1.0, -1.0),
    Vec3::new(-1.0, -1.0, 1.0),
    Vec3::new(-1.0, 1.0, 1.0),
    Vec3::new(-1.0, 1.0, 1.0),
    Vec3::new(-1.0, 1.0, -1.0),
    Vec3::new(-1.0, -1.0, -1.0),

    Vec3::new(1.0, -1.0, 1.0),
    Vec3::new(1.0, -1.0, -1.0),
    Vec3::new(1.0, 1.0, -1.0),
    Vec3::new(1.0, 1.0, -1.0),
    Vec3::new(1.0, 1.0, 1.0),
    Vec3::new(1.0, -1.0, 1.0),

    Vec3::new(-1.0, -1.0, -1.0),
    Vec3::new(-1.0, 1.0, -1.0),
    Vec3::new(1.0, 1.0, -1.0),
    Vec3::new(1.0, 1.0, -1.0),
    Vec3::new(1.0, -1.0, -1.0),
    Vec3::new(-1.0, -1.0, -1.0),

    Vec3::new(-1.0, 1.0, 1.0),
    Vec3::new(1.0, 1.0, 1.0),
    Vec3::new(1.0, 1.0, -1.0),
    Vec3::new(1.0, 1.0, -1.0),
    Vec3::new(-1.0, 1.0, -1.0),
    Vec3::new(-1.0, 1.0, 1.0),

    Vec3::new(-1.0, -1.0, 1.0),
    Vec3::new(-1.0, -1.0, -1.0),
    Vec3::new(1.0, -1.0, 1.0),
    Vec3::new(1.0, -1.0, 1.0),
    Vec3::new(-1.0, -1.0, -1.0),
    Vec3::new(1.0, -1.0, -1.0),
];

pub struct CubeMap {
    pub image: Res<Image>,
    pub vertex_buffer: Res<Buffer>,
}

impl CubeMap {
    pub fn new(renderer: &Renderer, pool: &ResourcePool, image: Res<Image>) -> Result<Self> {
        let vertex_data: &[u8] = bytemuck::cast_slice(&CUBE_VERTICES);

        let staging = {
            let memory_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

            let buffer = Buffer::new(renderer, pool, memory_flags, &BufferReq {
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                size: vertex_data.len() as u64
            })?;

            buffer.get_mapped()?.fill(vertex_data);

            buffer
        };

        let vertex_buffer = {
            let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

            Buffer::new(renderer, pool, memory_flags, &BufferReq {
                usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                size: vertex_data.len() as u64
            })?
        };

        renderer.transfer_with(|recorder|
            recorder.copy_buffers(&staging, &vertex_buffer)
        )?;

        Ok(Self { image, vertex_buffer })
    }
}

#[repr(C)]
struct CpuBlock {
    /// The first byte of the block.
    base: NonNull<u8>,

    /// The offset into `base` thats points at the first free byte.
    offset: Cell<usize>,

    /// The layout of the block.
    layout: alloc::Layout,
}

impl CpuBlock {
    const DEFAULT_BLOCK_SIZE: usize = 1024;

    fn new(size: usize) -> Self {
        let (base, layout) = unsafe {
            let layout = alloc::Layout::from_size_align_unchecked(size, 1);
            let base = NonNull::new(alloc::alloc(layout)).expect("out of memory");

            (base, layout)
        };

        Self { base, layout, offset: Cell::new(0) }
    }

    fn alloc<T>(&self) -> Option<NonNull<T>> {
        let (size, align) = (mem::size_of::<T>(), mem::align_of::<T>());

        let offset = self.offset.get();

        unsafe {
            let ptr = self.base.as_ptr().add(offset);
            let align_offset = ptr.align_offset(align);

            let start = ptr.add(ptr.align_offset(align));
            let end = start.add(size);

            if end > self.base.as_ptr().add(self.layout.size()) {
                None
            } else {
                self.offset.set(offset + align_offset + size);
                Some(NonNull::new_unchecked(start.cast()))
            }
        }
    }
}

impl Drop for CpuBlock {
    fn drop(&mut self) {
        unsafe { alloc::dealloc(self.base.as_ptr(), self.layout) }
    }
}

struct CpuBlocks {
    block: Rc<CpuBlock>,
    block_size: usize,
}

impl CpuBlocks {
    fn new(block_size: usize) -> Self {
        Self { block: Rc::new(CpuBlock::new(block_size)), block_size }
    }

    fn alloc<T>(&mut self, val: T) -> NonNull<ResState<T>> {
        let mut size = mem::size_of::<ResState<T>>();

        let (block, ptr) = loop {
            if let Some(ptr) = self.block.alloc::<ResState<T>>() {
                break (self.block.clone(), ptr);
            };

            // At this point we know that the current block is too small.
            
            let mut block_size = self.block_size;
            while size > block_size {
                block_size *= 2;
            }

            // Rare (impossible?) case where the allocation fails even if the block size is
            // the the same size as the allocation because of aligment offsets.
            size = block_size + 1;

            self.block = Rc::new(CpuBlock::new(block_size));
        };

        // SAFETY: `ptr` has just been allocated and is therefore valid.
        unsafe {
            ptr.as_ptr().write(ResState {
                ref_count: Cell::new(1),
                block, 
                val
            });
        }
       
        ptr
    }
}

struct GpuBlock {
    block: Rc<MemoryBlock>,
    offset: vk::DeviceSize,
}

impl GpuBlock {
    const DEFAULT_BLOCK_SIZE: vk::DeviceSize = 6 * 1024 * 1024;

    fn new(block: MemoryBlock) -> Self {
        Self { block: Rc::new(block), offset: 0 }
    }

    fn alloc(&mut self, size: vk::DeviceSize, alignment: vk::DeviceSize) -> Option<MemoryRange> {
        let start = align_up_to(self.offset, alignment);
        let end = start + size;

        if end > self.block.size() {
            return None; 
        }

        self.offset = end;

        Some(start..end)
    }
}

struct GpuBlocks {
    block_size: vk::DeviceSize,
    blocks: Vec<(u32, GpuBlock)>, 
}

impl GpuBlocks {
    fn new(block_size: vk::DeviceSize) -> Self {
        Self { block_size, blocks: Vec::new() }
    }

    fn alloc(
        &mut self,
        device: Res<Device>,
        memory_type: u32,
        size: vk::DeviceSize,
        aligment: vk::DeviceSize,
    ) -> Result<(Rc<MemoryBlock>, MemoryRange)> {
        let index = self.blocks.iter().position(|(type_index, _)| *type_index == memory_type);

        let create_new_block = || -> Result<GpuBlock> {
            let mut block_size = self.block_size;

            while block_size < size {
                block_size += GpuBlock::DEFAULT_BLOCK_SIZE; 
            }

            trace!("allocating a new {block_size} byte GPU block");

            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(block_size)
                .memory_type_index(memory_type);

            Ok(GpuBlock::new(MemoryBlock::new(device.clone(), &alloc_info)?))
        };
       
        let (_, block) = if let Some(index) = index {
            &mut self.blocks[index]
        } else {
            self.blocks.push((memory_type, create_new_block()?));
            self.blocks.last_mut().unwrap()
        };

        let range = loop {
            if let Some(range) = block.alloc(size, aligment) {
                break range;
            }

            *block = create_new_block()?;
        };

        Ok((block.block.clone(), range))
    }

}

struct SharedResources {
    /// The memory blocks where all the CPU resources are allocated.
    cpu_blocks: UnsafeCell<CpuBlocks>,
    gpu_blocks: UnsafeCell<GpuBlocks>,
}

impl SharedResources {
    fn new(cpu_block_size: usize, gpu_block_size: vk::DeviceSize) -> Self {
        Self {
            cpu_blocks: UnsafeCell::new(CpuBlocks::new(cpu_block_size)),
            gpu_blocks: UnsafeCell::new(GpuBlocks::new(gpu_block_size)),
        }
    }
}

/// A pool of resources.
///
/// When allocating an item of type `T` via `alloc`, you receivce an reference object of type
/// `Res<T>`. This is guarenteed to be valid for it's whole lifetime.
///
/// This should be kept in mind when used. A single [`Res`] may keep the whole pool alive longer
/// than expected, which could be a waste of memory.
///
/// If `T` implements `Drop`, then drop will be called for each [`Res`] once the last copy goes out
/// of scope.
#[derive(Clone)]
pub struct ResourcePool {
    shared: Rc<SharedResources>,
}

impl PartialEq for ResourcePool {
    fn eq(&self, other: &Self) -> bool {
        Rc::as_ptr(&self.shared) == Rc::as_ptr(&other.shared)
    }
}

impl Eq for ResourcePool {}

impl Hash for ResourcePool {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.shared).hash(state);
    }
}

impl ResourcePool {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::with_block_size(CpuBlock::DEFAULT_BLOCK_SIZE, GpuBlock::DEFAULT_BLOCK_SIZE)
    }

    #[inline]
    #[must_use]
    pub fn with_block_size(cpu_block_size: usize, gpu_block_size: vk::DeviceSize) -> Self {
        Self { shared: Rc::new(SharedResources::new(cpu_block_size, gpu_block_size)) }
    }

    fn gpu_alloc(
        &self,
        device: Res<Device>,
        memory_type: u32,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
    ) -> Result<(Rc<MemoryBlock>, MemoryRange)> {
        trace!("allocating {size} bytes on the GPU");

        unsafe { (*self.shared.gpu_blocks.get()).alloc(device, memory_type, size, alignment) }
    }

    #[inline]
    #[must_use]
    pub fn alloc<T>(&self, val: T) -> Res<T> {
        // SAFETY: `cpu_blocks` is only used here, and is therefore never borrowed.
        let ptr = unsafe { (*self.shared.cpu_blocks.get()).alloc::<T>(val) };
        Res { ptr }
    }
}

struct ResState<T> {
    /// The number of references to the item.
    ref_count: Cell<u32>,
    block: Rc<CpuBlock>,  

    /// The resource value.
    val: T,
}

/// A reference to a resource allocated from a [`ResourcePool`].
///
/// The resource is guarenteed to be alive as long as a reference exists. Drop will be called for
/// the resource as soon as the last reference goes out of scope. However, the memory may not be
/// immediately reclaimed.
pub struct Res<T> {
    ptr: NonNull<ResState<T>>,
}

impl<T> Res<T> {
    unsafe fn drop_in_place(&mut self) {
        if mem::needs_drop::<T>() {
            ptr::drop_in_place::<T>((&mut self.ptr.as_mut().val) as *mut T);
        }
    }
}

impl<T> Clone for Res<T> {
    fn clone(&self) -> Self {
        if mem::needs_drop::<T>() {
            unsafe {
                self.ptr.as_ref().ref_count.set(self.ptr.as_ref().ref_count.get() + 1);
            }
        }

        Self { ptr: self.ptr }
    }
}

impl<T> Drop for Res<T> {
    fn drop(&mut self) {
        unsafe {
            if self.ptr.as_ref().ref_count.get() == 1 {
                if mem::needs_drop::<T>() {
                    self.drop_in_place();
                }

                let rc = self.ptr.as_mut().block.clone();
                Rc::decrement_strong_count(Rc::into_raw(rc));
            } else {
                self.ptr.as_ref().ref_count.set(self.ptr.as_ref().ref_count.get() - 1);
            }
        }
    }
}

impl<T> PartialEq for Res<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<T> Eq for Res<T> {}

impl<T> ops::Deref for Res<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &self.ptr.as_ref().val }
    }
}

/// Round `a` up to next integer with aligned to `aligment`.
#[inline]
pub fn align_up_to(a: u64, alignment: u64) -> u64 {
    ((a + alignment - 1) / alignment) * alignment
}

#[test]
fn res_simple_alloc() {
    let p1 = ResourcePool::new();

    let nums: Vec<_> = (0..2048)
        .map(|i| p1.alloc(i as usize))
        .collect();

    for (i, j) in nums.into_iter().enumerate() {
        assert_eq!(i, *j);
    }
}

#[test]
fn res_big_alloc() {
    let p1 = ResourcePool::new();

    let _ = p1.alloc([0_u32; 1024]);
    let _ = p1.alloc([0_u32; 2048]);
}

#[test]
fn res_drop() {
    static mut COUNT: usize = 0;

    struct Test(usize);
    impl Drop for Test {
        fn drop(&mut self) {
            unsafe { COUNT += 1; }
        }
    }
    
    {
        let p1 = ResourcePool::new();

        let _ = p1.alloc(Test(0));
        let _ = p1.alloc(Test(0));
    }

    assert_eq!(unsafe { COUNT }, 2);
}

#[test]
fn res_with_inter_block_ref() {
    static mut COUNT: usize = 0;

    struct Test {
        #[allow(dead_code)]
        val: Res<usize>,
    }

    impl Drop for Test {
        fn drop(&mut self) {
            unsafe { COUNT += 1; }
        }
    }
    
    {
        let p1 = ResourcePool::new();

        let val = p1.alloc(0_usize);

        let _ = p1.alloc(Test { val: val.clone() });
        let _ = p1.alloc(Test { val: val.clone() });
    }

    assert_eq!(unsafe { COUNT }, 2);
}
