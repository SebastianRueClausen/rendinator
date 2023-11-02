use std::cell::Cell;
use std::hash;
use std::ops::{self, Deref};

use ash::vk;
use eyre::{Context, Result};

use crate::command::CommandBuffer;
use crate::device::Device;

#[derive(Debug, Clone, Copy)]
pub(crate) enum BufferKind {
    Index,
    Storage,
    Scratch,
    Descriptor { sampler: bool },
}

impl BufferKind {
    fn usage_flags(self) -> vk::BufferUsageFlags {
        let base_flags = vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let specific_flags = match self {
            BufferKind::Storage => vk::BufferUsageFlags::STORAGE_BUFFER,
            BufferKind::Descriptor { sampler: true } => {
                vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT
            }
            BufferKind::Descriptor { sampler: false } => {
                vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
            }
            BufferKind::Index => {
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
            }
            BufferKind::Scratch => vk::BufferUsageFlags::empty(),
        };
        specific_flags | base_flags
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BufferRequest {
    pub size: u64,
    pub kind: BufferKind,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Buffer {
    pub buffer: vk::Buffer,
    #[allow(dead_code)]
    pub size: vk::DeviceSize,
}

impl ops::Deref for Buffer {
    type Target = vk::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl Buffer {
    pub fn new(device: &Device, request: &BufferRequest) -> Result<Self> {
        let size = request.size.max(4);
        let usage = request.kind.usage_flags();
        let buffer_info =
            vk::BufferCreateInfo::builder().size(size).usage(usage);
        let buffer = unsafe {
            device
                .create_buffer(&buffer_info, None)
                .wrap_err("failed creating buffer")?
        };
        Ok(Self { buffer, size })
    }

    pub fn device_address(&self, device: &Device) -> vk::DeviceAddress {
        let address_info =
            vk::BufferDeviceAddressInfo::builder().buffer(self.buffer);
        unsafe { device.get_buffer_device_address(&address_info) }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
    }
}

fn format_aspect(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ImageViewRequest {
    pub mip_level_count: u32,
    pub base_mip_level: u32,
}

impl ImageViewRequest {
    pub const BASE: Self = Self { mip_level_count: 1, base_mip_level: 0 };
}

#[derive(Debug, Clone)]
pub(crate) struct ImageView {
    view: vk::ImageView,
    request: ImageViewRequest,
}

impl ops::Deref for ImageView {
    type Target = vk::ImageView;

    fn deref(&self) -> &Self::Target {
        &self.view
    }
}

impl ImageView {
    pub fn new(
        device: &Device,
        image: vk::Image,
        format: vk::Format,
        request: ImageViewRequest,
    ) -> Result<Self> {
        let ImageViewRequest { base_mip_level, mip_level_count } = request;
        let aspect_mask = format_aspect(format);
        let image_view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                base_mip_level,
                level_count: mip_level_count,
                base_array_layer: 0,
                aspect_mask,
                layer_count: 1,
            });
        let view = unsafe {
            device
                .create_image_view(&image_view_info, None)
                .wrap_err("failed creating image view")?
        };
        Ok(ImageView { view, request })
    }

    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_image_view(self.view, None);
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ImageRequest<'a> {
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub mip_level_count: u32,
    pub usage: vk::ImageUsageFlags,
    pub views: &'a [ImageViewRequest],
}

#[derive(Debug, Clone)]
pub(crate) struct Image {
    pub image: vk::Image,
    pub extent: vk::Extent3D,
    pub aspect: vk::ImageAspectFlags,
    pub views: Vec<ImageView>,
    pub swapchain: bool,
    pub layout: Cell<vk::ImageLayout>,
}

impl ops::Deref for Image {
    type Target = vk::Image;

    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        self.image == other.image
    }
}

impl Eq for Image {}

impl hash::Hash for Image {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.image.hash(state);
    }
}

impl Image {
    pub fn new(device: &Device, request: &ImageRequest) -> Result<Self> {
        let image_info = vk::ImageCreateInfo::builder()
            .format(request.format)
            .extent(request.extent)
            .mip_levels(request.mip_level_count)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(request.usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe {
            device
                .create_image(&image_info, None)
                .wrap_err("failed creating image")?
        };
        let views = request
            .views
            .iter()
            .copied()
            .map(|view_request| {
                ImageView::new(device, image, request.format, view_request)
            })
            .collect::<Result<_>>()?;
        let aspect = format_aspect(request.format);
        Ok(Self {
            layout: vk::ImageLayout::UNDEFINED.into(),
            extent: request.extent,
            swapchain: false,
            aspect,
            image,
            views,
        })
    }

    pub fn view(&self, request: &ImageViewRequest) -> &ImageView {
        self.views.iter().find(|view| view.request == *request).unwrap_or_else(
            || panic!("image wasn't created with view {request:?}"),
        )
    }

    pub fn layout(&self) -> vk::ImageLayout {
        self.layout.get()
    }

    pub fn set_layout(&self, layout: vk::ImageLayout) {
        self.layout.set(layout);
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            for view in &self.views {
                view.destroy(device);
            }
            if !self.swapchain {
                device.destroy_image(self.image, None);
            }
        }
    }
}

pub(crate) fn mip_level_extent(
    extent: vk::Extent3D,
    level: u32,
) -> vk::Extent3D {
    vk::Extent3D {
        width: extent.width >> level,
        height: extent.height >> level,
        depth: extent.depth,
    }
}

fn memory_type_index(
    device: &Device,
    memory_type_bits: u32,
    memory_flags: vk::MemoryPropertyFlags,
) -> Result<u32> {
    for (index, memory_type) in
        device.memory_properties.memory_types.iter().enumerate()
    {
        let has_bits = memory_type_bits & (1 << index) != 0;
        let has_flags = memory_type.property_flags.contains(memory_flags);
        if has_bits && has_flags {
            return Ok(index as u32);
        }
    }
    Err(eyre::eyre!("invalid memory type"))
}

pub(crate) struct Memory {
    memory: vk::DeviceMemory,
}

impl Deref for Memory {
    type Target = vk::DeviceMemory;

    fn deref(&self) -> &Self::Target {
        &self.memory
    }
}

impl Memory {
    pub fn allocate(
        device: &Device,
        size: vk::DeviceSize,
        memory_flags: vk::MemoryPropertyFlags,
        memory_type_bits: u32,
    ) -> Result<Self> {
        let memory_type_index =
            memory_type_index(device, memory_type_bits, memory_flags)?;
        let mut allocate_flags_info = vk::MemoryAllocateFlagsInfo::builder()
            .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let allocation_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(memory_type_index)
            .push_next(&mut allocate_flags_info);
        let memory = unsafe {
            device
                .allocate_memory(&allocation_info, None)
                .wrap_err("failed to allocate memory")?
        };
        Ok(Self { memory })
    }

    pub fn free(&self, device: &Device) {
        unsafe {
            device.free_memory(self.memory, None);
        }
    }

    pub fn map(&self, device: &Device) -> Result<*mut u8> {
        let flags = vk::MemoryMapFlags::empty();
        Ok(unsafe {
            device
                .map_memory(self.memory, 0, vk::WHOLE_SIZE, flags)
                .wrap_err("failed to map memory")? as *mut u8
        })
    }

    pub fn unmap(&self, device: &Device) {
        unsafe { device.unmap_memory(self.memory) };
    }
}

#[derive(Debug)]
pub(crate) struct BufferRange<'a> {
    pub buffer: &'a Buffer,
    pub offset: u64,
}

pub(crate) fn bind_buffer_memory(
    device: &Device,
    memory: &Memory,
    buffer_ranges: &[BufferRange],
) -> Result<()> {
    let bind_infos: Vec<_> = buffer_ranges
        .iter()
        .map(|range| {
            vk::BindBufferMemoryInfo::builder()
                .buffer(*range.buffer.deref())
                .memory_offset(range.offset)
                .memory(*memory.deref())
                .build()
        })
        .collect();
    unsafe {
        device
            .bind_buffer_memory2(&bind_infos)
            .wrap_err("failed to bind buffer memory")
    }
}

#[derive(Debug)]
pub(crate) struct ImageRange<'a> {
    pub image: &'a Image,
    pub offset: u64,
}

pub(crate) fn bind_image_memory(
    device: &Device,
    memory: &Memory,
    image_ranges: &[ImageRange],
) -> Result<()> {
    let bind_infos: Vec<_> = image_ranges
        .iter()
        .map(|range| {
            vk::BindImageMemoryInfo::builder()
                .image(*range.image.deref())
                .memory_offset(range.offset)
                .memory(*memory.deref())
                .build()
        })
        .collect();
    if bind_infos.is_empty() {
        return Ok(());
    }
    unsafe {
        device
            .bind_image_memory2(&bind_infos)
            .wrap_err("failed to bind image memory")
    }
}

pub(crate) fn buffer_memory(
    device: &Device,
    buffer: &Buffer,
    memory_flags: vk::MemoryPropertyFlags,
) -> Result<Memory> {
    let memory_reqs =
        unsafe { device.get_buffer_memory_requirements(**buffer) };
    let buffer_memory = Memory::allocate(
        device,
        memory_reqs.size,
        memory_flags,
        memory_reqs.memory_type_bits,
    )?;
    bind_buffer_memory(
        device,
        &buffer_memory,
        &[BufferRange { offset: 0, buffer }],
    )?;
    Ok(buffer_memory)
}

pub(crate) struct Scratch {
    buffer: Buffer,
    memory: Memory,
}

impl Scratch {
    fn new(device: &Device, size: vk::DeviceSize) -> Result<Self> {
        let buffer = Buffer::new(
            device,
            &BufferRequest { size, kind: BufferKind::Scratch },
        )?;
        let memory_flags = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory = buffer_memory(device, &buffer, memory_flags)?;
        Ok(Self { buffer, memory })
    }

    pub(crate) fn destroy(&self, device: &Device) {
        self.buffer.destroy(device);
        self.memory.free(device);
    }
}

pub(crate) struct BufferWrite<'a> {
    pub buffer: &'a Buffer,
    pub data: &'a [u8],
}

pub(crate) fn upload_buffer_data(
    device: &Device,
    command_buffer: &CommandBuffer,
    buffer_writes: &[BufferWrite],
) -> Result<Scratch> {
    let scratch_size =
        buffer_writes.iter().map(|write| write.data.len() as u64).sum();
    let scratch = Scratch::new(device, scratch_size)?;
    buffer_writes.iter().fold(
        scratch.memory.map(device)?,
        |ptr, write| unsafe {
            ptr.copy_from_nonoverlapping(write.data.as_ptr(), write.data.len());
            ptr.add(write.data.len())
        },
    );
    scratch.memory.unmap(device);
    buffer_writes.iter().fold(0, |offset, write| unsafe {
        let byte_count = write.data.len() as u64;
        if byte_count != 0 {
            let buffer_copy = vk::BufferCopy::builder()
                .src_offset(offset)
                .dst_offset(0)
                .size(byte_count);
            device.cmd_copy_buffer(
                *command_buffer.deref(),
                *scratch.buffer,
                *write.buffer.deref(),
                &[*buffer_copy],
            );
        }
        offset + byte_count
    });
    Ok(scratch)
}

pub(crate) struct ImageWrite<'a> {
    pub image: &'a Image,
    pub extent: vk::Extent3D,
    pub mips: &'a [Box<[u8]>],
}

pub(crate) fn upload_image_data(
    device: &Device,
    command_buffer: &CommandBuffer,
    image_writes: &[ImageWrite],
) -> Result<Scratch> {
    let scratch_size = image_writes
        .iter()
        .flat_map(|write| write.mips.iter().map(|mip| mip.len() as u64))
        .sum();
    let scratch = Scratch::new(device, scratch_size)?;
    image_writes.iter().flat_map(|write| write.mips.iter()).fold(
        scratch.memory.map(device)?,
        |ptr, mip| unsafe {
            ptr.copy_from_nonoverlapping(mip.as_ptr(), mip.len());
            ptr.add(mip.len())
        },
    );
    scratch.memory.unmap(device);
    image_writes
        .iter()
        .flat_map(|write| {
            write.mips.iter().enumerate().map(|(level, data)| {
                let extent = mip_level_extent(write.extent, level as u32);
                (write.image, extent, data, level as u32)
            })
        })
        .fold(0, |offset, (image, extent, data, level)| unsafe {
            let subresource = vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .mip_level(level)
                .build();
            let image_copy = vk::BufferImageCopy::builder()
                .buffer_offset(offset)
                .image_extent(extent)
                .buffer_image_height(extent.height)
                .buffer_row_length(extent.width)
                .image_subresource(subresource);
            device.cmd_copy_buffer_to_image(
                *command_buffer.deref(),
                *scratch.buffer,
                *image.deref(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[*image_copy],
            );
            offset + data.len() as u64
        });
    Ok(scratch)
}

pub(crate) struct Allocator<'a> {
    device: &'a Device,
    offset: vk::DeviceSize,
    memory_type_bits: u32,
    buffer_ranges: Vec<BufferRange<'a>>,
    image_ranges: Vec<ImageRange<'a>>,
}

impl<'a> Allocator<'a> {
    pub fn new(device: &'a Device) -> Self {
        Self {
            device,
            memory_type_bits: u32::MAX,
            buffer_ranges: Vec::default(),
            image_ranges: Vec::default(),
            offset: 0,
        }
    }

    fn allocate_range(
        &mut self,
        alignment: vk::DeviceSize,
        size: vk::DeviceSize,
    ) -> u64 {
        let offset = self.offset.next_multiple_of(alignment);
        self.offset = offset + size;
        offset
    }

    pub fn alloc_buffer(&mut self, buffer: &'a Buffer) {
        let reqs = unsafe {
            self.device.get_buffer_memory_requirements(*buffer.deref())
        };
        let offset = self.allocate_range(reqs.alignment, reqs.size);
        self.memory_type_bits &= reqs.memory_type_bits;
        self.buffer_ranges.push(BufferRange { buffer, offset });
    }

    pub fn alloc_image(&mut self, image: &'a Image) {
        let reqs = unsafe {
            self.device.get_image_memory_requirements(*image.deref())
        };
        let offset = self.allocate_range(reqs.alignment, reqs.size);
        self.memory_type_bits &= reqs.memory_type_bits;
        self.image_ranges.push(ImageRange { image, offset });
    }

    pub fn finish(
        self,
        memory_flags: vk::MemoryPropertyFlags,
    ) -> Result<Memory> {
        let memory = Memory::allocate(
            self.device,
            self.offset,
            memory_flags,
            self.memory_type_bits,
        )?;
        bind_buffer_memory(self.device, &memory, &self.buffer_ranges)?;
        bind_image_memory(self.device, &memory, &self.image_ranges)?;
        Ok(memory)
    }
}

/*
*/
