use anyhow::Result;
use smallvec::{smallvec, SmallVec};
use ash::vk;

use std::collections::HashMap;
use std::ffi::CString;
use std::{array, ops, slice, alloc, mem};
use std::rc::Rc;
use std::cell::{UnsafeCell, Cell};
use std::ptr::{self, NonNull};
use std::hash::{Hash, Hasher};

use crate::core::*;

type MemoryRange = ops::Range<vk::DeviceSize>;

fn range_length(range: &MemoryRange) -> vk::DeviceSize {
    range.end - range.start
}

#[derive(Clone, Copy)]
pub enum MemoryLocation {
    Gpu,
    Cpu,
}

impl Into<vk::MemoryPropertyFlags> for MemoryLocation {
    fn into(self) -> vk::MemoryPropertyFlags {
        match self {
            MemoryLocation::Gpu => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryLocation::Cpu => {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            }
        }
    }
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

    device: Rc<Device>,
}

impl MemoryBlock {
    fn new(device: Rc<Device>, info: &vk::MemoryAllocateInfo) -> Result<Self> {
        let handle = unsafe {
            device.handle.allocate_memory(info, None)?
        };

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
pub struct BufferInfo {
    pub usage: vk::BufferUsageFlags,
    pub size: u64,
}

#[derive(Clone)]
pub struct Buffer {
    pub handle: vk::Buffer,
    pub block: Rc<MemoryBlock>,
    pub range: MemoryRange,

    device: Rc<Device>,
}

impl ResourcePool {
    pub fn create_buffer(&self, loc: MemoryLocation, info: &BufferInfo) -> Result<Res<Buffer>> {
        let pool = unsafe {
            let info = vk::BufferCreateInfo::builder()
                .usage(info.usage)
                .size(info.size)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .build();

            let handle = self.device.handle.create_buffer(&info, None)?;
            let req = self.device.handle.get_buffer_memory_requirements(handle);

            let memory_flags: vk::MemoryPropertyFlags = loc.into();

            let memory_type = self.device.physical
                .get_memory_type_index(req.memory_type_bits, memory_flags)
                .ok_or_else(|| anyhow!("no compatible memory type"))?;

            let (block, range) = self.gpu_alloc(memory_type, req.size, req.alignment)?;
            let range = range.start..range.start + info.size;

            self.device.handle.bind_buffer_memory(handle, block.handle, range.start)?;
            self.alloc(Buffer { handle, range, block, device: self.device.clone() })
        };

        Ok(pool)
    }
}

impl Buffer {
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

/// The kind of image.
#[derive(Clone)]
pub enum ImageKind {
    /// Standard 2D texture.
    Texture,

    /// 6-layer cubemap.
    CubeMap,

    /// Render target. Either color or depth.
    RenderTarget {
        samples: vk::SampleCountFlags,
        queue: Res<Queue>,
    },

    Swapchain { handle: vk::Image },
}

pub struct ImageInfo {
    pub kind: ImageKind,
    pub usage: vk::ImageUsageFlags,
    pub aspect_flags: vk::ImageAspectFlags,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
}

/// Enum of different image owners.
enum ImageStorage {
    /// Images created from the swapchain is owned by the swapchain.
    ///
    /// FIXME: This is not optimal as the image will be invalid after destroying the swapchain.
    Swapchain,

    /// The image is owned by an allocated memory block.
    Block {
        /// The range of `block` owned by the image.
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
    device: Rc<Device>,
}

impl ResourcePool {
    pub fn create_image(&self, loc: MemoryLocation, info: &ImageInfo) -> Result<Res<Image>> {
        let (array_layers, flags) = if let ImageKind::CubeMap = info.kind {
            (6, vk::ImageCreateFlags::CUBE_COMPATIBLE)
        } else {
            (1, vk::ImageCreateFlags::empty())
        };

        let layout = Cell::new(vk::ImageLayout::UNDEFINED);

        if let ImageKind::Swapchain { handle } = info.kind {
            return Ok(self.alloc(Image {
                storage: ImageStorage::Swapchain,
                mip_levels: info.mip_levels,
                aspect_flags: info.aspect_flags,
                device: self.device.clone(),
                extent: info.extent,
                format: info.format,
                kind: info.kind.clone(),
                layout,
                handle,
            }));
        }

        let handle = {
            let create_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .usage(info.usage)
                .format(info.format)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .extent(info.extent)
                .mip_levels(info.mip_levels)
                .samples(vk::SampleCountFlags::TYPE_1)
                .array_layers(array_layers)
                .flags(flags);

            match &info.kind {
                ImageKind::RenderTarget { queue, samples } => {
                    let queue_families = [queue.family_index()];
                    let create_info = create_info
                        .queue_family_indices(&queue_families)
                        .samples(*samples)
                        .clone();
                   
                    unsafe {
                        self.device.handle.create_image(&create_info, None)?
                    }
                }
                _ => unsafe {
                    self.device.handle.create_image(&create_info, None)?
                }
            }
        };

        let memory_req = unsafe {
            self.device.handle.get_image_memory_requirements(handle)
        };

        let memory_flags: vk::MemoryPropertyFlags = loc.into();
         
        let memory_type = self.device.physical
            .get_memory_type_index(memory_req.memory_type_bits, memory_flags)
            .ok_or_else(|| anyhow!("no compatible memory type"))?;

        let (block, range) = self.gpu_alloc(memory_type, memory_req.size, memory_req.alignment)?;

        unsafe {
            self.device.handle.bind_image_memory(handle, block.handle, range.start)?;
        }

        let kind = info.kind.clone();

        let storage = ImageStorage::Block { range: 0..memory_req.size, block };

        Ok(self.alloc(Image {
            mip_levels: info.mip_levels,
            aspect_flags: info.aspect_flags,
            device: self.device.clone(),
            extent: info.extent,
            format: info.format,
            storage,
            layout,
            handle,
            kind,
        }))
    }
}

impl Image {
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
        // Swapchain images should not be manually destroyed.
        if let ImageStorage::Block { .. } = self.storage {
            unsafe {
                self.device.handle.destroy_image(self.handle, None);
            }
        }
    }
}

pub struct ImageViewInfo {
    pub image: Res<Image>,
    pub view_type: vk::ImageViewType,
    pub mips: ops::Range<u32>,
}

pub struct ImageView {
    pub handle: vk::ImageView,
    mips: ops::Range<u32>,
    image: Res<Image>,
}

impl ResourcePool {
    pub fn create_image_view(&self, info: &ImageViewInfo) -> Result<Res<ImageView>> {
        assert!(
            info.image.mip_level_count() >= info.mips.end,
            "mip levels outside range of image mips, is {} max is {}",
            info.mips.end,
            info.image.mip_level_count(),
        ); 

        let mip_count = info.mips.end - info.mips.start;

        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(info.image.aspect_flags())
            .base_mip_level(info.mips.start)
            .layer_count(info.image.layer_count())
            .level_count(mip_count)
            .base_array_layer(0);

        let view_info = vk::ImageViewCreateInfo::builder()
            .view_type(info.view_type)
            .subresource_range(*subresource_range)
            .format(info.image.format())
            .image(info.image.handle)
            .build();

        let handle = unsafe {
            self.device.handle.create_image_view(&view_info, None)?
        };
    
        Ok(self.alloc(ImageView { handle, image: info.image.clone(), mips: info.mips.clone() }))
    }
}

impl ImageView {
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

pub struct Sampler {
    pub handle: vk::Sampler,
    device: Rc<Device>,
}

impl ResourcePool {
    pub fn create_sampler(&self) -> Result<Res<Sampler>> {
        let device = self.device.clone(); 

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

        let handle = unsafe {
            device.handle.create_sampler(&create_info, None)?
        };

        Ok(self.alloc(Sampler { handle, device }))
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_sampler(self.handle, None); }
    }
}

pub struct ShaderModule {
    pub handle: vk::ShaderModule,
    entry: CString,

    device: Rc<Device>,
}

impl ResourcePool {
    pub fn create_shader_module(&self, entry: &str, code: &[u8]) -> Result<Res<ShaderModule>> {
        let device = self.device.clone();

        if code.len() % mem::size_of::<u32>() != 0 {
            return Err(anyhow!("shader code size must be a multiple of 4"));
        }

        if code.as_ptr().align_offset(mem::align_of::<u32>()) != 0 {
            return Err(anyhow!("shader code must be aligned to `u32`"));
        }

        let code = unsafe {
            slice::from_raw_parts(code.as_ptr() as *const u32, code.len() / 4)
        };

        let info = vk::ShaderModuleCreateInfo::builder().code(code);

        let handle = unsafe {
            device.handle.create_shader_module(&info, None)?
        };

        let Ok(entry) = CString::new(entry) else {
            return Err(anyhow!("invalid entry name `entry`"));
        };

        Ok(self.alloc(ShaderModule { device, handle, entry }))
    }
}

impl ShaderModule {
    fn stage_create_info(
        &self,
        stage: vk::ShaderStageFlags,
    ) -> impl ops::Deref<Target = vk::PipelineShaderStageCreateInfo> + '_ {
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(stage)
            .module(self.handle)
            .name(&self.entry)
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_shader_module(self.handle, None);
        }
    }
}

#[derive(Clone)]
pub struct PipelineLayout {
    pub handle: vk::PipelineLayout,

    #[allow(unused)]
    pub push_const_ranges: SmallVec<[PushConstRange; 4]>,

    #[allow(unused)]
    desc_layouts: SmallVec<[Res<DescLayout>; 4]>,

    device: Rc<Device>,
}

impl ResourcePool {
    pub fn create_pipeline_layout(
        &self,
        consts: &[PushConstRange],
        layouts: &[Res<DescLayout>],
    ) -> Result<Res<PipelineLayout>> {
        let device = self.device.clone();

        let handle = unsafe {
            let layouts: SmallVec<[_; 12]> = layouts
                .iter()
                .map(|layout| layout.handle)
                .collect();
        
            let mut offset = 0;
            let consts: SmallVec<[_; 6]> = consts
                .iter()
                .map(|range| {
                    let range = vk::PushConstantRange::builder()
                        .stage_flags(range.stage)
                        .size(range.size as u32)
                        .offset(offset)
                        .build();

                    offset += range.size as u32;

                    range
                })
                .collect();

            let info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&layouts)
                .push_constant_ranges(&consts);

            device.handle.create_pipeline_layout(&info, None)?
        };

        let desc_layouts = layouts
            .iter()
            .map(|layout| layout.clone())
            .collect();

        let push_const_ranges = consts
            .iter()
            .cloned()
            .collect();

        Ok(self.alloc(PipelineLayout { handle, desc_layouts, push_const_ranges, device }))
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_pipeline_layout(self.handle, None);
        }
    }
}

pub struct ComputePipeline {
    pub handle: vk::Pipeline,
    layout: Res<PipelineLayout>,
    device: Rc<Device>,
}

impl ResourcePool {
    pub fn create_compute_pipeline( &self,
        layout: Res<PipelineLayout>,
        shader: Res<ShaderModule>,
    ) -> Result<Res<ComputePipeline>> {
        let device = self.device.clone();
        let stage = shader.stage_create_info(vk::ShaderStageFlags::COMPUTE);
        let create_infos = [vk::ComputePipelineCreateInfo::builder()
            .layout(layout.handle)
            .stage(*stage)
            .build()];

        let handle = unsafe {
            device.handle
                .create_compute_pipelines(vk::PipelineCache::null(), &create_infos, None)
                .map_err(|(_, err)| err)?
                .first()
                .unwrap()
                .clone()
        };

        Ok(self.alloc(ComputePipeline { device, handle, layout }))
    }
}

impl ComputePipeline {
    pub fn layout(&self) -> Res<PipelineLayout> {
        self.layout.clone()
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_pipeline(self.handle, None); }
    }
}

#[derive(Clone, Copy)]
pub struct RenderTargetInfo {
    /// The format of depth image used for rendering.
    pub depth_format: vk::Format,

    /// The format of color image used for rendering.
    ///
    /// `None` if no color target should be used.
    pub color_format: Option<vk::Format>,

    /// The amount of samples used when rendering.
    ///
    /// All render targets should have same sample count.
    pub sample_count: vk::SampleCountFlags,
}

pub struct GraphicsPipelineInfo<'a> {
    pub layout: Res<PipelineLayout>,

    pub vertex_attributes: &'a [VertexAttribute],
    pub depth_stencil_info: &'a vk::PipelineDepthStencilStateCreateInfo,

    pub render_target_info: RenderTargetInfo,

    pub vertex_shader: Res<ShaderModule>,
    pub fragment_shader: Res<ShaderModule>,

    pub cull_mode: vk::CullModeFlags,
}

pub struct GraphicsPipeline {
    pub handle: vk::Pipeline,
    layout: Res<PipelineLayout>,
    device: Rc<Device>,
}

impl ResourcePool {
    pub fn create_graphics_pipeline(
        &self,
        info: GraphicsPipelineInfo,
    ) -> Result<Res<GraphicsPipeline>> {
        let device = self.device.clone();

        let shader_stages = [
            *info.vertex_shader.stage_create_info(vk::ShaderStageFlags::VERTEX),
            *info.fragment_shader.stage_create_info(vk::ShaderStageFlags::FRAGMENT),
        ];

        let mut offset = 0;

        let vert_attribs: SmallVec<[_; 8]> = info.vertex_attributes
            .iter()
            .enumerate()
            .map(|(i, attrib)| {
                let desc = vk::VertexInputAttributeDescription {
                    format: attrib.format,
                    location: i as u32,
                    binding: 0,
                    offset,
                };

                offset += attrib.size as u32;

                desc
            })
            .collect();

        let vertex_bindings: SmallVec<[_; 1]> = if info.vertex_attributes.len() != 0 {
            SmallVec::from([vk::VertexInputBindingDescription {
                input_rate: vk::VertexInputRate::VERTEX,
                stride: offset,
                binding: 0,
            }])
        } else {
            SmallVec::new()
        };

        let vert_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vert_attribs)
            .vertex_binding_descriptions(&vertex_bindings);

        let vert_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_info = vk::PipelineViewportStateCreateInfo::default();

        let rasterize_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .front_face(vk::FrontFace::CLOCKWISE)
            .cull_mode(info.cull_mode)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0);

        let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(info.render_target_info.sample_count);

        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_blend_op(vk::BlendOp::ADD)
            .blend_enable(true)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build()];

        let color_blend_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(&color_blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        let color_formats: SmallVec<[_; 1]> =
            if let Some(format) = info.render_target_info.color_format {
                SmallVec::from([format])
            } else {
                SmallVec::new()
            };

        let mut rendering_info = vk::PipelineRenderingCreateInfo::builder()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(info.render_target_info.depth_format);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .dynamic_state(&dynamic_state)
            .stages(&shader_stages)
            .vertex_input_state(&vert_input_info)
            .input_assembly_state(&vert_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterize_info)
            .multisample_state(&multisample_info)
            .depth_stencil_state(&info.depth_stencil_info)
            .color_blend_state(&color_blend_info)
            .layout(info.layout.handle)
            .push_next(&mut rendering_info);

        let handle = unsafe {
            *device.handle
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_info.build()],
                    None,
                )
                .map_err(|(_, error)| anyhow!("failed to create pipeline: {error}"))?
                .first()
                .unwrap()
        };

        Ok(self.alloc(GraphicsPipeline { device, handle, layout: info.layout }))
    }
}

impl GraphicsPipeline {
    pub fn layout(&self) -> Res<PipelineLayout> {
        self.layout.clone()
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_pipeline(self.handle, None);
        }
    }
}

#[derive(Clone, Copy)]
pub struct PushConstRange {
    pub stage: vk::ShaderStageFlags, 
    pub size: vk::DeviceSize,
}

#[derive(Clone, Copy)]
pub struct VertexAttribute {
    pub format: vk::Format,

    // TODO: Remove this once we implement our own formats.
    pub size: vk::DeviceSize,
}

struct DescPool {
    pub handle: vk::DescriptorPool,
    device: Rc<Device>,
}

impl DescPool {
    fn new(
        device: Rc<Device>,
        max_sets: u32,
        sizes: &[vk::DescriptorPoolSize],
    ) -> Result<Self> {
        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&sizes)
            .max_sets(max_sets);

        let handle = unsafe {
            device.handle.create_descriptor_pool(&info, None)?
        };

        Ok(Self { handle, device })
    }
}

impl Drop for DescPool {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_descriptor_pool(self.handle, None); }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct DescLayoutSlot {
    pub stage: vk::ShaderStageFlags,
    pub ty: vk::DescriptorType,
    pub array_count: Option<u32>,
}

struct DescLayoutSlots {
    bindings: SmallVec<[vk::DescriptorSetLayoutBinding; 6]>,
    flags: SmallVec<[vk::DescriptorBindingFlags; 6]>,
    variable_set_count: u32,
}

impl DescLayoutSlots {
    fn new(bindings: &[DescLayoutSlot]) -> Self  {
        let flags = bindings
            .iter()
            .map(|binding| {
                let mut flags = vk::DescriptorBindingFlags::empty(); 

                if binding.array_count.is_some() {
                    flags |= vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
                }

                flags
            })
            .collect();
        let variable_set_count = bindings
            .last()
            .map(|b| b.array_count)
            .flatten()
            .unwrap_or(1);
        let bindings = bindings
            .iter()
            .enumerate()
            .map(|(i, binding)| {
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(i as u32)
                    .descriptor_type(binding.ty)
                    .descriptor_count(binding.array_count.unwrap_or(1))
                    .stage_flags(binding.stage)
                    .build()
            })
            .collect();
        Self { bindings, flags, variable_set_count }
    }

    fn iter(&self) -> impl Iterator<Item = &vk::DescriptorSetLayoutBinding> {
        self.bindings.iter()
    }
}

pub struct DescLayout {
    pub handle: vk::DescriptorSetLayout,
    slots: DescLayoutSlots,
    device: Rc<Device>, 
}

impl DescLayout {
    pub fn new(device: Rc<Device>, slots: &[DescLayoutSlot]) -> Result<Self> {
        let slots = DescLayoutSlots::new(slots);

        let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
            .binding_flags(&slots.flags)
            .build();

        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&slots.bindings)
            .push_next(&mut binding_flags)
            .build();

        let handle = unsafe {
            device.handle.create_descriptor_set_layout(&layout_info, None)?
        };

        Ok(Self { handle, slots, device }) 
    }
}

impl ResourcePool {
    pub fn create_desc_layout(&self, slots: &[DescLayoutSlot]) -> Result<Res<DescLayout>> {
        let shared = unsafe {
            self.get_shared()
        };

        if let Some(layout) = shared.desc_layouts.get(slots) {
            return Ok(layout);
        }

        let layout = self.alloc(
            DescLayout::new(self.device.clone(), slots)?
        );

        shared.desc_layouts.insert(slots, layout.clone());

        Ok(layout)
    }
}

impl Drop for DescLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_descriptor_set_layout(self.handle, None);
        }
    }
}

pub enum DescBinding<'a> {
    Buffer(Res<Buffer>),
    Image(Res<Sampler>, vk::ImageLayout, Res<ImageView>),
    ImageArray(Res<Sampler>, vk::ImageLayout, &'a [Res<ImageView>]),
}

pub struct DescSet {
    pub handle: vk::DescriptorSet,

    pub layout: Res<DescLayout>,

    #[allow(unused)]
    pool: Rc<DescPool>,

    #[allow(unused)]
    resources: SmallVec<[DummyRes; 32]>,
}

impl ResourcePool {
    pub fn create_desc_set(&self, layout: Res<DescLayout>, bindings: &[DescBinding]) -> Result<Res<DescSet>> {
        let device = self.device.clone();
        let (handle, pool) = self.desc_alloc(&layout)?;

        struct Info {
            ty: vk::DescriptorType,
            buffers: SmallVec<[vk::DescriptorBufferInfo; 1]>,
            images: SmallVec<[vk::DescriptorImageInfo; 1]>,
        }

        let infos: SmallVec<[Info; 12]> = bindings.iter()
            .zip(layout.slots.iter())
            .map(|(binding, layout_binding)| match &binding {
                DescBinding::Buffer(buffer) => Info {
                    ty: layout_binding.descriptor_type,
                    images: smallvec![vk::DescriptorImageInfo::default()],
                    buffers: smallvec![vk::DescriptorBufferInfo {
                        buffer: buffer.handle,
                        range: buffer.size(),
                        offset: 0,
                    }],
                },
                DescBinding::Image(sampler, layout, image) => Info {
                    ty: layout_binding.descriptor_type,
                    buffers: smallvec![vk::DescriptorBufferInfo::default()],
                    images: smallvec![vk::DescriptorImageInfo {
                        image_view: image.handle,
                        sampler: sampler.handle,
                        image_layout: *layout,
                    }],
                },
                DescBinding::ImageArray(sampler, layout, array) => Info {
                    ty: layout_binding.descriptor_type,
                    buffers: smallvec![vk::DescriptorBufferInfo::default()],
                    images: array
                        .iter()
                        .map(|image| vk::DescriptorImageInfo {
                            image_layout: *layout,
                            image_view: image.handle,
                            sampler: sampler.handle,
                        })
                        .collect(),
                }
            })
            .collect();

        let writes: SmallVec<[vk::WriteDescriptorSet; 12]> = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::builder()
                    .dst_binding(binding as u32)
                    .descriptor_type(info.ty)
                    .buffer_info(&info.buffers)
                    .image_info(&info.images)
                    .dst_set(handle)
                    .build()
            })
            .collect();

        unsafe {
            device.handle.update_descriptor_sets(&writes, &[])
        }

        let mut resources: SmallVec<[_; 32]> = SmallVec::default();
        for binding in bindings {
            match &binding {
                DescBinding::Buffer(buffer) => {
                    resources.push(buffer.create_dummy());
                }
                DescBinding::Image(sampler, _, image) => {
                    resources.push(sampler.create_dummy());
                    resources.push(image.create_dummy());
                }
                DescBinding::ImageArray(sampler, _, array) => {
                    resources.push(sampler.create_dummy());

                    for image in *array {
                        resources.push(image.create_dummy()); 
                    }
                }
            }
        }

        Ok(self.alloc(DescSet { layout, pool, handle, resources }))
    }
}

impl DescSet {
    pub fn layout(&self) -> Res<DescLayout> {
        self.layout.clone()
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
        device: Rc<Device>,
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
struct DescPools {
    current_pool: Option<Rc<DescPool>>, 
}

impl DescPools {
    fn new_pool(device: Rc<Device>, size_factor: f32) -> Result<Rc<DescPool>> {
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

        DescPool::new(device, max_sets, &sizes).map(|pool| Rc::new(pool))
    }

    pub fn alloc(
        &mut self,
        device: Rc<Device>,
        layout: &DescLayout,
    ) -> Result<(vk::DescriptorSet, Rc<DescPool>)> {
        let layouts = [layout.handle];
        let set_counts = [layout.slots.variable_set_count];

        let mut size_factor = 1.0;

        let (handles, pool) = loop {
            let Some(current_pool) = &self.current_pool else {
                self.current_pool = Some(Self::new_pool(device.clone(), size_factor)?);

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

                device.handle.allocate_descriptor_sets(&alloc_info)
            };

            match handles {
                Ok(handles) => {
                    break (handles, current_pool.clone())
                }

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

#[derive(PartialEq, Eq, Hash)]
struct DescLayoutKey {
    slots: SmallVec<[DescLayoutSlot; 8]>,
}

#[derive(Default)]
struct DescLayouts {
    layouts: HashMap<DescLayoutKey, Res<DescLayout>>, 
}

impl DescLayouts {
    fn insert(&mut self, bindings: &[DescLayoutSlot], layout: Res<DescLayout>) {
        let key = DescLayoutKey {
            slots: SmallVec::from_slice(bindings),
        };

        self.layouts.insert(key, layout);
    }

    fn get(&self, bindings: &[DescLayoutSlot]) -> Option<Res<DescLayout>> {
        let key = DescLayoutKey {
            slots: SmallVec::from_slice(bindings),
        };
      
        self.layouts.get(&key).cloned()
    }
}

struct SharedResources {
    /// The memory blocks where all the CPU resources are allocated.
    cpu_blocks: CpuBlocks,
    gpu_blocks: GpuBlocks,
    desc_pools: DescPools,
    desc_layouts: DescLayouts,
}

impl SharedResources {
    fn new(cpu_block_size: usize, gpu_block_size: vk::DeviceSize) -> Self {
        Self {
            cpu_blocks: CpuBlocks::new(cpu_block_size),
            gpu_blocks: GpuBlocks::new(gpu_block_size),
            desc_pools: DescPools::default(),
            desc_layouts: DescLayouts::default(),
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
    shared: Rc<UnsafeCell<SharedResources>>,
    device: Rc<Device>,
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
    /// Create new resource pool with default pool sizes.
    #[inline]
    #[must_use]
    pub fn new(device: Rc<Device>) -> Self {
        Self::with_block_size(device, CpuBlock::DEFAULT_BLOCK_SIZE, GpuBlock::DEFAULT_BLOCK_SIZE)
    }

    /// Create new resource pool with custom block sizes.
    #[inline]
    #[must_use]
    pub fn with_block_size(
        device: Rc<Device>,
        cpu_block_size: usize,
        gpu_block_size: vk::DeviceSize,
    ) -> Self {
        let shared = Rc::new(
            UnsafeCell::new(SharedResources::new(cpu_block_size, gpu_block_size))
        );

        Self { device, shared }
    }

    unsafe fn get_shared(&self) -> &mut SharedResources {
        self.shared.get().as_mut().unwrap_unchecked()
    }

    /// Allocate block of GPU memory.
    #[inline]
    #[must_use]
    fn gpu_alloc(
        &self,
        memory_type: u32,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
    ) -> Result<(Rc<MemoryBlock>, MemoryRange)> {
        trace!("allocating {size} bytes on the GPU");

        unsafe {
            self.get_shared().gpu_blocks.alloc(self.device.clone(), memory_type, size, alignment)
        }
    }

    /// Allocate descriptor set.
    #[inline]
    #[must_use]
    fn desc_alloc(
        &self,
        layout: &DescLayout,
    ) -> Result<(vk::DescriptorSet, Rc<DescPool>)> {
        unsafe {
            self.get_shared().desc_pools.alloc(self.device.clone(), layout)
        }
    }

    /// Allocate item from CPU memory pool.
    #[inline]
    #[must_use]
    pub fn alloc<T>(&self, val: T) -> Res<T> {
        // SAFETY: `cpu_blocks` is only used here, and is therefore never borrowed.
        let ptr = unsafe {
            self.get_shared().cpu_blocks.alloc::<T>(val)
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


pub const FRAMES_IN_FLIGHT: usize = 2;

#[derive(Default, Clone, Copy)]
pub struct FrameIndex {
    index: u8,
}

impl FrameIndex {
    pub fn enumerate() -> impl Iterator<Item = Self> {
        (0..FRAMES_IN_FLIGHT as u8).map(|index| Self { index })
    }

    pub fn last(self) -> Self {
        Self {
            index: self.index.wrapping_sub(1) % FRAMES_IN_FLIGHT as u8
        }
    }

    pub fn next(self) -> Self {
        Self {
            index: self.index.wrapping_add(1) % FRAMES_IN_FLIGHT as u8
        }
    }
}
pub struct PerFrame<T> {
    items: [T; FRAMES_IN_FLIGHT],
}

impl<T> PerFrame<T> {
    pub fn from_fn<F>(func: F) -> Self
    where F: Fn(FrameIndex) -> T
    {
        let items = array::from_fn(|index| {
            let frame_index = FrameIndex { index: index as u8 };
            func(frame_index)
        });

        Self { items }
    }

    pub fn try_from_fn<F>(func: F) -> Result<Self>
    where F: Fn(FrameIndex) -> Result<T>
    {
        let items = array::try_from_fn(|index| {
            let frame_index = FrameIndex { index: index as u8 };
            func(frame_index)
        })?;

        Ok(Self { items })
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    pub fn any(&self) -> &T {
        &self[FrameIndex { index: 0 }]
    }
}

impl<T> ops::Index<FrameIndex> for PerFrame<T> {
    type Output = T;

    fn index(&self, index: FrameIndex) -> &Self::Output {
        &self.items[index.index as usize]
    }
}

impl<T> ops::IndexMut<FrameIndex> for PerFrame<T> {
    fn index_mut(&mut self, index: FrameIndex) -> &mut Self::Output {
        &mut self.items[index.index as usize]
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
    let instance = Rc::new(Instance::new(false).unwrap());
    let physical = PhysicalDevice::select(&instance).unwrap();

    let device = Rc::new(Device::new(instance, physical, &[]).unwrap());
    let p1 = ResourcePool::new(device);

    let nums: Vec<_> = (0..2048)
        .map(|i| p1.alloc(i as usize))
        .collect();

    for (i, j) in nums.into_iter().enumerate() {
        assert_eq!(i, *j);
    }
}

#[test]
fn res_big_alloc() {
    let instance = Rc::new(Instance::new(false).unwrap());
    let physical = PhysicalDevice::select(&instance).unwrap();

    let device = Rc::new(Device::new(instance, physical, &[]).unwrap());
    let p1 = ResourcePool::new(device);

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
        let instance = Rc::new(Instance::new(false).unwrap());
        let physical = PhysicalDevice::select(&instance).unwrap();

        let device = Rc::new(Device::new(instance, physical, &[]).unwrap());
        let p1 = ResourcePool::new(device);

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
        let instance = Rc::new(Instance::new(false).unwrap());
        let physical = PhysicalDevice::select(&instance).unwrap();

        let device = Rc::new(Device::new(instance, physical, &[]).unwrap());
        let p1 = ResourcePool::new(device);

        let val = p1.alloc(0_usize);

        let _ = p1.alloc(Test { val: val.clone() });
        let _ = p1.alloc(Test { val: val.clone() });
    }

    assert_eq!(unsafe { COUNT }, 2);
}

/// Just test it doesn't crash.
#[test]
fn dummy_res() {
    let instance = Rc::new(Instance::new(false).unwrap());
    let physical = PhysicalDevice::select(&instance).unwrap();

    let device = Rc::new(Device::new(instance, physical, &[]).unwrap());
    let p1 = ResourcePool::new(device);

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
        let instance = Rc::new(Instance::new(false).unwrap());
        let physical = PhysicalDevice::select(&instance).unwrap();

        let device = Rc::new(Device::new(instance, physical, &[]).unwrap());
        let p1 = ResourcePool::new(device);

        let res = p1.alloc(Test(0));
        res.create_dummy()
    };

    assert_eq!(unsafe { COUNT }, 0);
}
