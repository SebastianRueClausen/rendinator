use anyhow::Result;
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

    unsafe fn as_slice_mut(&self) -> &mut [u8] {
        slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size as usize)
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(self.ptr.as_ptr(), self.size as usize)
        }
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
    
        unsafe { self.as_slice_mut().copy_from_slice(bytes) };
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
        req: &BufferReq,
    ) -> Result<Res<Self>> {
        let pool = unsafe {
            let info = vk::BufferCreateInfo::builder()
                .usage(req.usage)
                .size(req.size)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .build();

            let handle = renderer.device.handle.create_buffer(&info, None)?;
            let req = renderer.device.handle.get_buffer_memory_requirements(handle);

            let memory_type = renderer.device.physical
                .get_memory_type_index(req.memory_type_bits, memory_flags)
                .ok_or_else(|| anyhow!("no compatible memory type"))?;

            let (block, range) = pool.gpu_alloc(
                renderer.device.clone(),
                memory_type,
                req.size,
                req.alignment,
            )?;

            let range = range.start..range.start + info.size;

            renderer.device.handle.bind_buffer_memory(handle, block.handle, range.start)?;

            pool.alloc(Self { handle, range, block, device: renderer.device.clone() })
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

#[derive(Clone)]
pub enum ImageKind {
    Texture,
    CubeMap,
    RenderTarget {
        samples: vk::SampleCountFlags,
        queue: Res<Queue>,
    },
    Swapchain { handle: vk::Image },
}

pub struct ImageReq {
    pub kind: ImageKind,
    pub usage: vk::ImageUsageFlags,
    pub aspect_flags: vk::ImageAspectFlags,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
}

enum ImageStorage {
    Swapchain,
    Block {
        range: MemoryRange,

        #[allow(dead_code)]
        block: Rc<MemoryBlock>,
    },
}

// FIXME: Swapchain images may be invalid if the swapchain is recreated or destroyed.
pub struct Image {
    pub handle: vk::Image,
    pub layout: Cell<vk::ImageLayout>,

    kind: ImageKind,

    aspect_flags: vk::ImageAspectFlags,
    extent: vk::Extent3D,
    format: vk::Format,
    mip_levels: u32,

    storage: ImageStorage,
    device: Res<Device>,
}

impl Image {
    pub fn new(
        renderer: &Renderer,
        pool: &ResourcePool,
        memory_flags: vk::MemoryPropertyFlags,
        req: &ImageReq,
    ) -> Result<Res<Self>> {
        Self::from_device(renderer.device.clone(), pool, memory_flags, req)
    }

    pub fn from_device(
        device: Res<Device>,
        pool: &ResourcePool,
        memory_flags: vk::MemoryPropertyFlags,
        req: &ImageReq,
    ) -> Result<Res<Self>> {
        let (array_layers, flags) = if let ImageKind::CubeMap = req.kind {
            (6, vk::ImageCreateFlags::CUBE_COMPATIBLE)
        } else {
            (1, vk::ImageCreateFlags::empty())
        };

        let layout = Cell::new(vk::ImageLayout::UNDEFINED);

        if let ImageKind::Swapchain { handle } = req.kind {
            return Ok(pool.alloc(Self {
                storage: ImageStorage::Swapchain,
                mip_levels: req.mip_levels,
                aspect_flags: req.aspect_flags,
                device: device.clone(),
                extent: req.extent,
                format: req.format,
                kind: req.kind.clone(),
                layout,
                handle,
            }));
        }

        let handle = {
            let info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .usage(req.usage)
                .format(req.format)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .extent(req.extent)
                .mip_levels(req.mip_levels)
                .samples(vk::SampleCountFlags::TYPE_1)
                .array_layers(array_layers)
                .flags(flags);

            match &req.kind {
                ImageKind::RenderTarget { queue, samples } => {
                    let queue_families = [queue.family_index()];
                    let info = info
                        .queue_family_indices(&queue_families)
                        .samples(*samples)
                        .clone();
                   
                    unsafe {
                        device.handle.create_image(&info, None)?
                    }
                }
                _ => unsafe {
                    device.handle.create_image(&info, None)?
                }
            }
        };

        let memory_req = unsafe {
            device.handle.get_image_memory_requirements(handle)
        };
         
        let memory_type = device.physical
            .get_memory_type_index(memory_req.memory_type_bits, memory_flags)
            .ok_or_else(|| anyhow!("no compatible memory type"))?;

        let (block, range) = pool.gpu_alloc(
            device.clone(),
            memory_type,
            memory_req.size,
            memory_req.alignment,
        )?;

        unsafe {
            device.handle.bind_image_memory(handle, block.handle, range.start)?;
        }

        let kind = req.kind.clone();

        let storage = ImageStorage::Block { range: 0..memory_req.size, block };

        Ok(pool.alloc(Self {
            mip_levels: req.mip_levels,
            aspect_flags: req.aspect_flags,
            device: device.clone(),
            extent: req.extent,
            format: req.format,
            storage,
            layout,
            handle,
            kind,
        }))
    }

    pub fn sample_count(&self) -> vk::SampleCountFlags {
        if let ImageKind::RenderTarget { samples, .. } = self.kind {
            samples
        } else {
            vk::SampleCountFlags::TYPE_1
        }
    }

    #[allow(dead_code)]
    pub fn size(&self) -> vk::DeviceSize {
        match &self.storage {
            ImageStorage::Block { range, .. } => range_length(range),
            ImageStorage::Swapchain => todo!(),
        }
    }

    pub fn layout(&self) -> vk::ImageLayout {
        self.layout.get()
    }

    pub fn layer_count(&self) -> u32 {
        if let ImageKind::CubeMap = &self.kind {
            6
        } else {
            1
        }
    }
    
    pub fn aspect_flags(&self) -> vk::ImageAspectFlags {
        self.aspect_flags
    }

    pub fn format(&self) -> vk::Format {
        self.format
    }

    pub fn mip_level_count(&self) -> u32 {
        self.mip_levels
    }

    pub fn mip_levels(&self) -> ops::Range<u32> {
        0..self.mip_level_count()
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
        // Swapchain images should not be manually destroyd.
        if let ImageStorage::Block { .. } = self.storage {
            unsafe {
                self.device.handle.destroy_image(self.handle, None);
            }
        }
    }
}

pub struct ImageViewReq {
    pub image: Res<Image>,
    pub view_type: vk::ImageViewType,
    pub mips: ops::Range<u32>,
}

pub struct ImageView {
    pub handle: vk::ImageView,
    mips: ops::Range<u32>,
    image: Res<Image>,
}

impl ImageView {
    pub fn new(renderer: &Renderer, pool: &ResourcePool, req: &ImageViewReq) -> Result<Res<Self>> {
        Self::from_device(renderer.device.clone(), pool, req)
    }

    pub fn from_device(
        device: Res<Device>,
        pool: &ResourcePool,
        req: &ImageViewReq,
    ) -> Result<Res<Self>> {
        assert!(
            req.image.mip_level_count() >= req.mips.end,
            "mip levels outside range of image mips, is {} max is {}",
            req.mips.end,
            req.image.mip_level_count(),
        ); 

        let mip_count = req.mips.end - req.mips.start;

        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(req.image.aspect_flags())
            .base_mip_level(req.mips.start)
            .layer_count(req.image.layer_count())
            .level_count(mip_count)
            .base_array_layer(0);

        let view_info = vk::ImageViewCreateInfo::builder()
            .view_type(req.view_type)
            .subresource_range(*subresource_range)
            .format(req.image.format())
            .image(req.image.handle)
            .build();

        let handle = unsafe {
            device.handle.create_image_view(&view_info, None)?
        };
    
        Ok(pool.alloc(Self { handle, image: req.image.clone(), mips: req.mips.clone() }))
    }

    /// Get the extent of a given mip level.
    ///
    /// This is diffferent from [`Image::extent`] in that this gives the extent of the mip level
    /// relative to the first level of the image view.
    #[allow(dead_code)]
    pub fn extent(&self, mip_level: u32) -> vk::Extent3D {
        self.image.extent(self.mips.start + mip_level)
    }

    #[allow(dead_code)]
    pub fn layout(&self) -> vk::ImageLayout {
        self.image.layout()
    }

    pub fn image(&self) -> &Res<Image> {
        &self.image
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe { self.image.device.handle.destroy_image_view(self.handle, None) }
    }
}

pub struct TextureSampler {
    pub handle: vk::Sampler,
    device: Res<Device>,
}

impl TextureSampler {
    pub fn new(renderer: &Renderer, reduction: vk::SamplerReductionMode) -> Result<Self> {
        let device = renderer.device.clone(); 
        let mut create_info = vk::SamplerCreateInfo::builder()
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

        let mut reduction_info = vk::SamplerReductionModeCreateInfo::default();

        if reduction != vk::SamplerReductionMode::WEIGHTED_AVERAGE {
            reduction_info.reduction_mode = reduction;
            create_info = create_info.push_next(&mut reduction_info);
        }

        let handle = unsafe {
            device.handle.create_sampler(&create_info, None)?
        };

        Ok(Self { handle, device })
    }
}

impl Drop for TextureSampler {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_sampler(self.handle, None); }
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

#[derive(Default)]
struct DescriptorPools {
    current_pool: Option<Rc<DescriptorPool>>, 
}

impl DescriptorPools {
    fn new_pool(renderer: &Renderer, size_factor: f32) -> Result<Rc<DescriptorPool>> {
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: (200.0 * size_factor) as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: (100.0 * size_factor) as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: (100.0 * size_factor) as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: (200.0 * size_factor) as u32,
            },
        ];

        let max_sets = (50.0 * size_factor) as u32;

        DescriptorPool::new(renderer, max_sets, &sizes).map(|pool| Rc::new(pool))
    }

    pub fn alloc(
        &mut self,
        renderer: &Renderer,
        layout: &DescriptorSetLayout,
    ) -> Result<(vk::DescriptorSet, Rc<DescriptorPool>)> {
        let layouts = [layout.handle];
        let set_counts = [layout.bindings.variable_set_count];

        let mut size_factor = 1.0;

        let (handles, pool) = loop {
            let Some(current_pool) = &self.current_pool else {
                self.current_pool = Some(Self::new_pool(renderer, size_factor)?);

                continue;
            };

            let handles = unsafe {
                let mut variable_count_info =
                    vk::DescriptorSetVariableDescriptorCountAllocateInfo::builder()
                        .descriptor_counts(&set_counts)
                        .build();
                let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(current_pool.handle)
                    .set_layouts(&layouts)
                    .push_next(&mut variable_count_info);

                renderer.device.handle.allocate_descriptor_sets(&alloc_info)
            };

            match handles {
                Ok(handles) => break (handles, current_pool.clone()),
                Err(vk::Result::ERROR_FRAGMENTED_POOL | vk::Result::ERROR_OUT_OF_POOL_MEMORY) => {
                    self.current_pool = None;
                    size_factor *= 2.0;
                }
                Err(err) => {
                    return Err(err.into());
                }
            }
        };

        Ok((handles[0], pool))
    }
}

struct SharedResources {
    /// The memory blocks where all the CPU resources are allocated.
    cpu_blocks: UnsafeCell<CpuBlocks>,
    gpu_blocks: UnsafeCell<GpuBlocks>,
    descriptor_pools: UnsafeCell<DescriptorPools>,

}

impl SharedResources {
    fn new(cpu_block_size: usize, gpu_block_size: vk::DeviceSize) -> Self {
        Self {
            cpu_blocks: UnsafeCell::new(CpuBlocks::new(cpu_block_size)),
            gpu_blocks: UnsafeCell::new(GpuBlocks::new(gpu_block_size)),
            descriptor_pools: UnsafeCell::new(DescriptorPools::default()),
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

    #[inline]
    #[must_use]
    fn gpu_alloc(
        &self,
        device: Res<Device>,
        memory_type: u32,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
    ) -> Result<(Rc<MemoryBlock>, MemoryRange)> {
        trace!("allocating {size} bytes on the GPU");

        unsafe {
            (*self.shared.gpu_blocks.get()).alloc(device, memory_type, size, alignment)
        }
    }

    #[inline]
    #[must_use]
    pub fn descriptor_alloc(
        &self,
        renderer: &Renderer,
        layout: &DescriptorSetLayout,
    ) -> Result<(vk::DescriptorSet, Rc<DescriptorPool>)> {
        unsafe {
            (*self.shared.descriptor_pools.get()).alloc(renderer, layout)
        }
    }

    #[inline]
    #[must_use]
    pub fn alloc<T>(&self, val: T) -> Res<T> {
        // SAFETY: `cpu_blocks` is only used here, and is therefore never borrowed.
        let ptr = unsafe {
            (*self.shared.cpu_blocks.get()).alloc::<T>(val)
        };

        Res { ptr }
    }
}

#[repr(C)]
struct ResState<T> {
    /// The number of references to the item.
    ref_count: Cell<u32>,
    block: Rc<CpuBlock>,  

    /// The resource value.
    val: T,
}

/// A dummy res exists to hold resources alive without holding a typed [`Res`] struct.
///
/// This is handy if you for instance want to have a list of resources with different types that
/// you want to ensure lives for a certain time.
pub struct DummyRes {
    ptr: NonNull<ResState<()>>,

    val: *mut (),
    drop: Option<unsafe fn(*mut ())>,
}

impl Drop for DummyRes {
    fn drop(&mut self) {
        unsafe {
            if self.ptr.as_ref().ref_count.get() == 1 {
                if let Some(drop_ptr) = self.drop {
                    (drop_ptr)(self.val);
                }

                let rc = self.ptr.as_mut().block.clone();
                Rc::decrement_strong_count(Rc::into_raw(rc));
            } else {
                let count = self.ptr.as_ref().ref_count.get() - 1;
                self.ptr.as_ref().ref_count.set(count);
            }
        }
    }
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

    fn increase_ref_count(&self) {
        if mem::needs_drop::<T>() {
            unsafe {
                self.ptr.as_ref().ref_count.set(self.ptr.as_ref().ref_count.get() + 1);
            }
        }
    }

    pub fn to_dummy(self) -> DummyRes {
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
            mem::transmute::<*mut T, *mut ()>(&mut (*self.ptr.as_ptr()).val as *mut T)
        };

        let ptr = self.ptr.cast();

        mem::forget(self);

        DummyRes { drop, val, ptr }
    }

    pub fn create_dummy(&self) -> DummyRes {
        self.clone().to_dummy()
    }
}

impl<T> Clone for Res<T> {
    fn clone(&self) -> Self {
        self.increase_ref_count();
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

pub struct PerFrame<T> {
    items: [T; FRAMES_IN_FLIGHT],
}

impl<T> PerFrame<T> {
    pub fn from_fn<F>(func: F) -> Self
    where F: Fn(FrameIndex) -> T
    {
        Self { items: FrameIndex::ALL.map(|idx| func(idx)) }
    }

    pub fn try_from_fn<F>(func: F) -> Result<Self>
    where F: Fn(FrameIndex) -> Result<T>
    {
        FrameIndex::ALL
            .try_map(|idx| func(idx))
            .map(|items| Self { items })
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    pub fn any(&self) -> &T {
        &self[FrameIndex::Uno]
    }
}

impl<T> ops::Index<FrameIndex> for PerFrame<T> {
    type Output = T;

    fn index(&self, idx: FrameIndex) -> &Self::Output {
        &self.items[idx as usize]
    }
}

impl<T: Clone> Clone for PerFrame<T> {
    fn clone(&self) -> Self {
        Self { items: self.items.clone() }
    }
}

impl<T> Into<[T; FRAMES_IN_FLIGHT]> for PerFrame<T> {
    fn into(self) -> [T; FRAMES_IN_FLIGHT] {
        self.items
    }
}

impl<'a, T> IntoIterator for &'a PerFrame<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.as_slice().iter()
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

/// Just test it doesn't crash.
#[test]
fn dummy_res() {
    let p1 = ResourcePool::new();

    let res = p1.alloc(0);

    for _ in 0..100 {
        res.create_dummy();
    }
}

#[test]
fn dummy_res_drop() {
    static mut COUNT: usize = 0;

    struct Test(usize);
    impl Drop for Test {
        fn drop(&mut self) {
            unsafe { COUNT += 1; }
        }
    }
    
    let _dummy = {
        let p1 = ResourcePool::new();

        let res = p1.alloc(Test(0));
        res.create_dummy()
    };

    assert_eq!(unsafe { COUNT }, 0);
}
