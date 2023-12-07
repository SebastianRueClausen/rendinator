use std::cell::Cell;
use std::ops::{self, Deref};
use std::{array, hash, mem, slice};

use ash::vk;
use eyre::{Context, Result};
use glam::Mat4;

use crate::command::{self, CommandBuffer};
use crate::device::Device;

#[derive(Debug, Clone, Copy)]
pub(crate) enum BufferKind {
    Index,
    Storage,
    Uniform,
    Scratch,
    Descriptor,
    Indirect,
    AccStructStorage,
    AccStructInput,
}

impl BufferKind {
    fn usage_flags(self) -> vk::BufferUsageFlags {
        let base_flags = vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let specific_flags = match self {
            BufferKind::Storage => vk::BufferUsageFlags::STORAGE_BUFFER,
            BufferKind::Uniform => vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferKind::Descriptor => {
                vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT
                    | vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
            }
            BufferKind::Index => {
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            }
            BufferKind::Scratch => vk::BufferUsageFlags::empty(),
            BufferKind::Indirect => {
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
            }
            BufferKind::AccStructStorage => {
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            }
            BufferKind::AccStructInput => {
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            }
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
pub(crate) struct ImageRequest {
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub mip_level_count: u32,
    pub usage: vk::ImageUsageFlags,
}

#[derive(Debug)]
pub(crate) struct Image {
    pub image: vk::Image,
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub aspect: vk::ImageAspectFlags,
    pub mip_level_count: u32,
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
        let layout = vk::ImageLayout::UNDEFINED;
        let image_info = vk::ImageCreateInfo::builder()
            .format(request.format)
            .extent(request.extent)
            .mip_levels(request.mip_level_count)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(request.usage)
            .image_type(vk::ImageType::TYPE_2D)
            .initial_layout(layout);
        let image = unsafe {
            device
                .create_image(&image_info, None)
                .wrap_err("failed creating image")?
        };
        let aspect = format_aspect(request.format);
        Ok(Self {
            layout: Cell::new(layout),
            extent: request.extent,
            format: request.format,
            mip_level_count: request.mip_level_count,
            swapchain: false,
            views: Vec::default(),
            aspect,
            image,
        })
    }

    pub fn spanning_offsets(&self, mip_level: u32) -> [vk::Offset3D; 2] {
        let extent = mip_level_extent(self.extent, mip_level);
        let max = vk::Offset3D {
            x: extent.width as i32,
            y: extent.height as i32,
            z: 1,
        };
        [vk::Offset3D::default(), max]
    }

    pub fn add_view(
        &mut self,
        device: &Device,
        request: ImageViewRequest,
    ) -> Result<()> {
        let view = ImageView::new(device, self.image, self.format, request)?;
        self.views.push(view);
        Ok(())
    }

    pub fn view(&self, request: &ImageViewRequest) -> &ImageView {
        self.views.iter().find(|view| view.request == *request).unwrap_or_else(
            || panic!("image wasn't created with view {request:?}"),
        )
    }

    pub fn full_view(&self) -> &ImageView {
        self.view(&ImageViewRequest {
            mip_level_count: self.mip_level_count,
            base_mip_level: 0,
        })
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

pub(crate) fn mip_level_offset(
    offset: vk::Offset3D,
    level: u32,
) -> vk::Offset3D {
    vk::Offset3D { x: offset.x >> level, y: offset.y >> level, z: offset.z }
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
        assert!(size > 0, "trying to allocate 0 bytes");
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
    pub offset: vk::DeviceSize,
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
    if !bind_infos.is_empty() {
        unsafe {
            device
                .bind_buffer_memory2(&bind_infos)
                .wrap_err("failed to bind buffer memory")
        }
    } else {
        Ok(())
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
    if !bind_infos.is_empty() {
        unsafe {
            device
                .bind_image_memory2(&bind_infos)
                .wrap_err("failed to bind image memory")
        }
    } else {
        Ok(())
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

pub(crate) fn image_memory(
    device: &Device,
    image: &Image,
    memory_flags: vk::MemoryPropertyFlags,
) -> Result<Memory> {
    let memory_reqs = unsafe { device.get_image_memory_requirements(**image) };
    let image_memory = Memory::allocate(
        device,
        memory_reqs.size,
        memory_flags,
        memory_reqs.memory_type_bits,
    )?;
    bind_image_memory(
        device,
        &image_memory,
        &[ImageRange { image, offset: 0 }],
    )?;
    Ok(image_memory)
}

pub(crate) struct Scratch {
    pub buffer: Buffer,
    pub memory: Memory,
}

impl Scratch {
    pub(crate) fn new(device: &Device, size: vk::DeviceSize) -> Result<Self> {
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
                **command_buffer,
                *scratch.buffer,
                **write.buffer,
                &[*buffer_copy],
            );
        }
        offset + byte_count
    });
    Ok(scratch)
}

pub(crate) struct ImageWrite<'a> {
    pub image: &'a Image,
    pub offset: vk::Offset3D,
    pub extent: vk::Extent3D,
    pub mips: &'a [Box<[u8]>],
}

pub(crate) fn upload_image_data(
    device: &Device,
    command_buffer: &CommandBuffer,
    image_writes: &[ImageWrite],
) -> Result<Scratch> {
    const CHUNK_ALIGNMENT: usize = 16;
    let scratch_size = image_writes
        .iter()
        .flat_map(|write| {
            write
                .mips
                .iter()
                .map(|mip| mip.len().next_multiple_of(CHUNK_ALIGNMENT) as u64)
        })
        .sum();
    let scratch = Scratch::new(device, scratch_size)?;
    image_writes.iter().flat_map(|write| write.mips.iter()).fold(
        scratch.memory.map(device)?,
        |ptr, mip| unsafe {
            ptr.copy_from_nonoverlapping(mip.as_ptr(), mip.len());
            ptr.add(mip.len().next_multiple_of(CHUNK_ALIGNMENT))
        },
    );
    scratch.memory.unmap(device);
    image_writes
        .iter()
        .flat_map(|write| {
            write.mips.iter().enumerate().map(move |(level, data)| {
                let level = level as u32;
                let offset = mip_level_offset(write.offset, level);
                let extent = mip_level_extent(write.extent, level);
                (write.image, extent, offset, data, level)
            })
        })
        .fold(
            0,
            |buffer_offset, (image, extent, offset, data, level)| unsafe {
                let subresource = vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .mip_level(level)
                    .build();
                let image_copy = vk::BufferImageCopy::builder()
                    .buffer_offset(buffer_offset)
                    .image_extent(extent)
                    .image_offset(offset)
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
                buffer_offset
                    + data.len().next_multiple_of(CHUNK_ALIGNMENT) as u64
            },
        );
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

    pub fn alloc_buffer(&mut self, buffer: &'a Buffer) -> &mut Self {
        let reqs = unsafe {
            self.device.get_buffer_memory_requirements(*buffer.deref())
        };
        let offset = self.allocate_range(reqs.alignment, reqs.size);
        self.memory_type_bits &= reqs.memory_type_bits;
        self.buffer_ranges.push(BufferRange { buffer, offset });
        self
    }

    pub fn alloc_image(&mut self, image: &'a Image) -> &mut Self {
        let reqs = unsafe {
            self.device.get_image_memory_requirements(*image.deref())
        };
        let offset = self.allocate_range(reqs.alignment, reqs.size);
        self.memory_type_bits &= reqs.memory_type_bits;
        self.image_ranges.push(ImageRange { image, offset });
        self
    }

    pub fn alloc_blas(&mut self, blas: &'a Blas) -> &mut Self {
        self.alloc_buffer(&blas.buffer)
    }

    pub fn alloc_tlas(&mut self, tlas: &'a Tlas) -> &mut Self {
        self.alloc_buffer(&tlas.buffer)
            .alloc_buffer(&tlas.instances)
            .alloc_buffer(&tlas.scratch)
    }

    pub fn finish(
        &mut self,
        memory_flags: vk::MemoryPropertyFlags,
    ) -> Result<Memory> {
        let size = self.offset.max(4);
        let memory = Memory::allocate(
            self.device,
            size,
            memory_flags,
            self.memory_type_bits,
        )?;
        bind_buffer_memory(self.device, &memory, &self.buffer_ranges)?;
        bind_image_memory(self.device, &memory, &self.image_ranges)?;
        self.buffer_ranges.clear();
        self.image_ranges.clear();
        Ok(memory)
    }
}

#[derive(Default)]
pub(crate) struct SamplerRequest {
    pub filter: vk::Filter,
    pub max_anisotropy: Option<f32>,
    pub address_mode: vk::SamplerAddressMode,
    pub reduction_mode: Option<vk::SamplerReductionMode>,
}

pub(crate) struct Sampler {
    sampler: vk::Sampler,
}

impl Deref for Sampler {
    type Target = vk::Sampler;

    fn deref(&self) -> &Self::Target {
        &self.sampler
    }
}

impl Sampler {
    pub fn new(device: &Device, request: &SamplerRequest) -> Result<Self> {
        let mut create_info = vk::SamplerCreateInfo::builder()
            .mag_filter(request.filter)
            .min_filter(request.filter)
            .address_mode_u(request.address_mode)
            .address_mode_v(request.address_mode)
            .address_mode_w(request.address_mode)
            .max_anisotropy(request.max_anisotropy.unwrap_or_default())
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .unnormalized_coordinates(false)
            .anisotropy_enable(request.max_anisotropy.is_some())
            .compare_enable(false)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE);
        let mut reduction_info = vk::SamplerReductionModeCreateInfo::default();
        if let Some(reduction_mode) = request.reduction_mode {
            reduction_info.reduction_mode = reduction_mode;
            create_info = create_info.push_next(&mut reduction_info);
        }
        let sampler = unsafe {
            device
                .create_sampler(&create_info, None)
                .wrap_err("failed to create sampler")?
        };
        Ok(Self { sampler })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BlasRequest {
    pub vertex_format: vk::Format,
    pub vertex_stride: u64,
    pub triangle_count: u32,
    pub vertex_count: u32,
    pub first_vertex: u32,
}

impl BlasRequest {
    fn geometry(&self) -> vk::AccelerationStructureGeometryKHR {
        let geometry_data = vk::AccelerationStructureGeometryDataKHR {
            triangles:
                vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                    .vertex_stride(self.vertex_stride)
                    .vertex_format(self.vertex_format)
                    .max_vertex(self.first_vertex + self.vertex_count - 1)
                    .index_type(vk::IndexType::UINT32)
                    .build(),
        };
        vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(geometry_data)
            .build()
    }
}

#[derive(Debug)]
pub(crate) struct Blas {
    acceleration_structure: vk::AccelerationStructureKHR,
    build_scratch_size: vk::DeviceSize,
    request: BlasRequest,
    buffer: Buffer,
}

impl Blas {
    pub fn new(device: &Device, request: &BlasRequest) -> Result<Self> {
        let geometry = request.geometry();
        let build_info =
            vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(
                    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
                )
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(slice::from_ref(&geometry))
                .build();
        let sizes = unsafe {
            device.acc_struct_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[request.triangle_count],
            )
        };
        let buffer = Buffer::new(
            device,
            &BufferRequest {
                size: sizes.acceleration_structure_size + 128,
                kind: BufferKind::AccStructStorage,
            },
        )?;
        let as_info = vk::AccelerationStructureCreateInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .size(sizes.acceleration_structure_size)
            .buffer(*buffer)
            .offset(0);
        let acceleration_structure = unsafe {
            device
                .acc_struct_loader
                .create_acceleration_structure(&as_info, None)
                .wrap_err("failed to create blas")?
        };
        Ok(Self {
            build_scratch_size: sizes.build_scratch_size * 2,
            acceleration_structure,
            request: *request,
            buffer,
        })
    }

    pub fn destroy(&self, device: &Device) {
        self.buffer.destroy(device);
        unsafe {
            device.acc_struct_loader.destroy_acceleration_structure(
                self.acceleration_structure,
                None,
            );
        }
    }

    pub fn device_address(&self, device: &Device) -> vk::DeviceAddress {
        acceleration_structure_device_address(
            device,
            self.acceleration_structure,
        )
    }
}

pub(crate) struct BlasBuild<'a> {
    pub blas: &'a Blas,
    pub vertices: BufferRange<'a>,
    pub indices: BufferRange<'a>,
}

pub(crate) fn build_blases<'a>(
    device: &Device,
    builds: &[BlasBuild<'a>],
) -> Result<()> {
    if builds.is_empty() {
        return Ok(());
    }

    let scratch_buffers: Vec<_> = builds
        .iter()
        .map(|build| {
            Buffer::new(
                device,
                &BufferRequest {
                    kind: BufferKind::Storage,
                    size: build.blas.build_scratch_size,
                },
            )
        })
        .collect::<Result<_>>()?;

    let mut allocator = Allocator::new(device);
    for buffer in &scratch_buffers {
        allocator.alloc_buffer(buffer);
    }

    let scratch_memory =
        allocator.finish(vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

    let geometries: Vec<_> = builds
        .into_iter()
        .map(|build| {
            let request = &build.blas.request;

            assert!({
                let last_access = build.vertices.offset
                    + request.vertex_stride * request.vertex_count as u64;
                let size = build.vertices.buffer.size - build.vertices.offset;
                last_access < size
            });

            assert!({
                let last_access = request.triangle_count as u64
                    * mem::size_of::<u32>() as u64
                    * 3;
                let size = build.indices.buffer.size - build.indices.offset;
                last_access < size
            });

            let geometry_data = vk::AccelerationStructureGeometryDataKHR {
                triangles:
                    vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                        .vertex_stride(request.vertex_stride)
                        .vertex_format(request.vertex_format)
                        .index_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: build
                                .indices
                                .buffer
                                .device_address(device),
                        })
                        .vertex_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: build
                                .vertices
                                .buffer
                                .device_address(device)
                                + build.vertices.offset,
                        })
                        .max_vertex(
                            request.first_vertex + request.vertex_count - 1,
                        )
                        .index_type(vk::IndexType::UINT32)
                        .build(),
            };
            vk::AccelerationStructureGeometryKHR::builder()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(geometry_data)
                .build()
        })
        .collect();

    let build_infos: Vec<_> = builds
        .iter()
        .zip(geometries.iter())
        .zip(scratch_buffers.iter())
        .map(|((build, geometry), scratch_buffer)| {
            vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                .dst_acceleration_structure(build.blas.acceleration_structure)
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(
                    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
                )
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(slice::from_ref(geometry))
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_buffer.device_address(device),
                })
                .build()
        })
        .collect();

    let range_infos: Vec<_> = builds
        .iter()
        .map(|build| {
            vk::AccelerationStructureBuildRangeInfoKHR::builder()
                .primitive_count(build.blas.request.triangle_count)
                .primitive_offset(build.indices.offset as u32)
                .first_vertex(build.blas.request.first_vertex)
                .build()
        })
        .collect();

    let range_infos_refs: Vec<_> =
        range_infos.iter().map(|range| slice::from_ref(range)).collect();

    command::quickie(device, |command_buffer| unsafe {
        device.acc_struct_loader.cmd_build_acceleration_structures(
            **command_buffer,
            &build_infos,
            &range_infos_refs,
        );
        Ok(())
    })?;

    for buffer in &scratch_buffers {
        buffer.destroy(device);
    }
    scratch_memory.free(device);

    Ok(())
}

pub(crate) struct Tlas {
    pub acceleration_structure: vk::AccelerationStructureKHR,
    pub buffer: Buffer,
    pub scratch: Buffer,
    pub instances: Buffer,
}

impl Tlas {
    pub fn new(device: &Device, instance_count: u32) -> Result<Self> {
        let modes = [
            vk::BuildAccelerationStructureModeKHR::BUILD,
            vk::BuildAccelerationStructureModeKHR::UPDATE,
        ];

        let [build_size, update_size] = modes.map(|mode| {
            let geometry_data = vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                    .build(),
            };
            let geometry = vk::AccelerationStructureGeometryKHR::builder()
                .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                .flags(vk::GeometryFlagsKHR::OPAQUE)
                .geometry(geometry_data)
                .build();
            let build_info =
                vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                    .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                    .geometries(slice::from_ref(&geometry))
                    .flags(
                        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    )
                    .mode(mode)
                    .build();
            unsafe {
                device.acc_struct_loader.get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    slice::from_ref(&instance_count),
                )
            }
        });

        let buffer = Buffer::new(
            device,
            &BufferRequest {
                size: build_size.acceleration_structure_size,
                kind: BufferKind::AccStructStorage,
            },
        )?;

        let scratch = Buffer::new(
            device,
            &BufferRequest {
                size: build_size
                    .build_scratch_size
                    .max(update_size.update_scratch_size),
                kind: BufferKind::Storage,
            },
        )?;

        let instances = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of::<TlasInstanceData>() as u64
                    * instance_count as u64,
                kind: BufferKind::AccStructInput,
            },
        )?;

        let as_info = vk::AccelerationStructureCreateInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .buffer(*buffer)
            .size(buffer.size)
            .offset(0);
        let acceleration_structure = unsafe {
            device
                .acc_struct_loader
                .create_acceleration_structure(&as_info, None)
                .wrap_err("failed to create tlas")?
        };

        Ok(Self { acceleration_structure, buffer, scratch, instances })
    }

    pub fn update(
        &self,
        device: &Device,
        mode: vk::BuildAccelerationStructureModeKHR,
        instances: &[TlasInstance],
    ) -> TlasUpdate {
        let instances = instances
            .iter()
            .map(|instance| {
                let blas_address = instance.blas.device_address(device);
                let flags = vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE |
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE;
                TlasInstanceData {
                    transform: acceleration_structure_transform(
                        instance.transform,
                    ),
                    blas_address,
                    flags: flags.as_raw() as u8,
                    mask: 0xff,
                    binding_table_offset: [0x0; 3],
                    index: [0x0; 3],
                }
            })
            .collect();
        TlasUpdate { instances, mode }
    }

    pub fn device_address(&self, device: &Device) -> vk::DeviceAddress {
        acceleration_structure_device_address(
            device,
            self.acceleration_structure,
        )
    }

    pub fn destroy(&self, device: &Device) {
        self.buffer.destroy(device);
        self.instances.destroy(device);
        self.scratch.destroy(device);
        unsafe {
            device.acc_struct_loader.destroy_acceleration_structure(
                self.acceleration_structure,
                None,
            );
        }
    }
}

pub(crate) struct TlasInstance<'a> {
    pub blas: &'a Blas,
    pub transform: Mat4,
}

pub(crate) struct TlasUpdate {
    instances: Vec<TlasInstanceData>,
    mode: vk::BuildAccelerationStructureModeKHR,
}

impl TlasUpdate {
    pub fn update(
        &self,
        device: &Device,
        command_buffer: &mut CommandBuffer,
        tlas: &Tlas,
    ) {
        let geometry = &tlas_geometry_info(device, &tlas.instances);
        let build_info =
            vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                .dst_acceleration_structure(tlas.acceleration_structure)
                .src_acceleration_structure(tlas.acceleration_structure)
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE)
                .mode(self.mode)
                .flags(
                    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                        | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
                )
                .geometries(slice::from_ref(geometry))
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: tlas.scratch.device_address(device),
                })
                .build();
        let build_ranges =
            vk::AccelerationStructureBuildRangeInfoKHR::builder()
                .primitive_count(self.instances.len() as u32)
                .build();
        unsafe {
            device.acc_struct_loader.cmd_build_acceleration_structures(
                **command_buffer,
                slice::from_ref(&build_info),
                slice::from_ref(&slice::from_ref(&build_ranges)),
            );
        }
    }

    pub fn buffer_write<'a>(&'a self, tlas: &'a Tlas) -> BufferWrite<'a> {
        BufferWrite {
            data: bytemuck::cast_slice(&self.instances),
            buffer: &tlas.instances,
        }
    }
}

// vk::AccelerationStructureInstanceKHR.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct TlasInstanceData {
    transform: [f32; 12],
    index: [u8; 3],
    mask: u8,
    binding_table_offset: [u8; 3],
    flags: u8,
    blas_address: vk::DeviceAddress,
}

fn acceleration_structure_device_address(
    device: &Device,
    acc_struct: vk::AccelerationStructureKHR,
) -> vk::DeviceAddress {
    let address_info = vk::AccelerationStructureDeviceAddressInfoKHR::builder()
        .acceleration_structure(acc_struct)
        .build();
    unsafe {
        device
            .acc_struct_loader
            .get_acceleration_structure_device_address(&address_info)
    }
}

fn tlas_geometry_info(
    device: &Device,
    instance_buffer: &Buffer,
) -> vk::AccelerationStructureGeometryKHR {
    let geometry_data = vk::AccelerationStructureGeometryDataKHR {
        instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: instance_buffer.device_address(device),
            })
            .build(),
    };
    vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .geometry(geometry_data)
        .build()
}

fn acceleration_structure_transform(transform: Mat4) -> [f32; 12] {
    let src = transform.transpose().to_cols_array();
    array::from_fn(|index| src[index])
}

#[test]
fn tlas_instance_size() {
    assert_eq!(
        mem::size_of::<vk::AccelerationStructureInstanceKHR>(),
        mem::size_of::<TlasInstanceData>()
    );
    assert_eq!(
        mem::align_of::<vk::AccelerationStructureInstanceKHR>(),
        mem::align_of::<TlasInstanceData>()
    );
}
