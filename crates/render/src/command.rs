use std::collections::HashMap;
use std::ops::Deref;
use std::slice;

use ash::vk;
use eyre::{Context, Result};

use crate::descriptor::{Descriptor, DescriptorBuffer};
use crate::device::Device;
use crate::resources::{Buffer, Image, ImageView};
use crate::shader::Pipeline;
use crate::sync::Sync;

pub(crate) struct CommandBuffer<'a> {
    buffer: vk::CommandBuffer,
    image_layouts: HashMap<&'a Image, vk::ImageLayout>,
}

impl<'a> Deref for CommandBuffer<'a> {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<'a> CommandBuffer<'a> {
    pub fn new(device: &Device) -> Result<Self> {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(device.command_pool)
            .command_buffer_count(1);
        let buffers =
            unsafe { device.allocate_command_buffers(&allocate_info)? };
        Ok(Self { buffer: buffers[0], image_layouts: HashMap::default() })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.free_command_buffers(device.command_pool, &[self.buffer]);
        }
    }

    pub fn begin(&mut self, device: &Device) -> Result<&mut Self> {
        let begin_info = vk::CommandBufferBeginInfo::builder();
        unsafe {
            device
                .begin_command_buffer(self.buffer, &begin_info)
                .wrap_err("failed to begin command buffer")?;
        }
        Ok(self)
    }

    pub fn end(&mut self, device: &Device) -> Result<&mut Self> {
        for (image, layout) in self.image_layouts.drain() {
            image.set_layout(layout);
        }
        unsafe {
            device
                .end_command_buffer(self.buffer)
                .wrap_err("failed to end command buffer")?;
        }
        Ok(self)
    }

    pub fn bind_pipeline(
        &mut self,
        device: &Device,
        pipeline: &Pipeline,
    ) -> &mut Self {
        unsafe {
            device.cmd_bind_pipeline(**self, pipeline.bind_point, **pipeline);
        }
        self
    }

    pub fn bind_descriptor(
        &mut self,
        device: &Device,
        pipeline: &Pipeline,
        descriptor: &Descriptor,
    ) -> &mut Self {
        unsafe {
            let Pipeline { bind_point, layout, .. } = pipeline;
            let offsets = [descriptor.buffer_offset];
            device.descriptor_buffer_loader.cmd_set_descriptor_buffer_offsets(
                **self,
                *bind_point,
                *layout,
                0,
                &[0],
                &offsets,
            );
        }
        self
    }

    pub fn bind_descriptor_buffer<'b>(
        &mut self,
        device: &Device,
        buffer: &DescriptorBuffer,
    ) -> &mut Self {
        let binding_info = vk::DescriptorBufferBindingInfoEXT::builder()
            .address(buffer.buffer.device_address(device))
            .usage(
                vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
                    | vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT,
            )
            .build();
        unsafe {
            device.descriptor_buffer_loader.cmd_bind_descriptor_buffers(
                **self,
                slice::from_ref(&binding_info),
            );
        }
        self
    }

    pub fn clear_color_image(
        &mut self,
        device: &Device,
        image: &Image,
    ) -> &mut Self {
        let range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_array_layer(0)
            .layer_count(1)
            .base_mip_level(0)
            .level_count(image.mip_level_count);
        unsafe {
            device.cmd_clear_color_image(
                **self,
                **image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: [0.0; 4] },
                slice::from_ref(&range),
            )
        }
        self
    }

    pub fn dispatch(
        &mut self,
        device: &Device,
        x: u32,
        y: u32,
        z: u32,
    ) -> &mut Self {
        unsafe { device.cmd_dispatch(self.buffer, x, y, z) }
        self
    }

    pub fn image_layout(&self, image: &Image) -> vk::ImageLayout {
        self.image_layouts.get(image).copied().unwrap_or_else(|| image.layout())
    }

    pub fn ensure_image_layouts(
        &mut self,
        device: &Device,
        layout: ImageLayouts,
        images: impl IntoIterator<Item = &'a Image>,
    ) -> &mut Self {
        let barriers: Vec<_> = images
            .into_iter()
            .filter_map(|image| {
                (self.image_layout(image) != layout.layout).then(|| {
                    ImageBarrier {
                        mip_levels: MipLevels::All,
                        image,
                        new_layout: layout.layout,
                        src: layout.src,
                        dst: layout.dst,
                    }
                })
            })
            .collect();
        self.pipeline_barriers(device, &barriers, &[]);
        self
    }

    pub fn pipeline_barriers(
        &mut self,
        device: &Device,
        image_barriers: &[ImageBarrier<'a>],
        buffer_barriers: &[BufferBarrier],
    ) -> &mut Self {
        let image_barriers: Vec<_> = image_barriers
            .iter()
            .map(|barrier| {
                let old_layout = self
                    .image_layouts
                    .get(barrier.image)
                    .copied()
                    .unwrap_or(barrier.image.layout());
                self.image_layouts.insert(barrier.image, barrier.new_layout);
                let (base_mip, mip_count) = match barrier.mip_levels {
                    MipLevels::All => (0, barrier.image.mip_level_count),
                    MipLevels::Levels { base, count } => (base, count),
                };
                let subresource_range = vk::ImageSubresourceRange::builder()
                    .aspect_mask(barrier.image.aspect)
                    .base_mip_level(base_mip)
                    .level_count(mip_count)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build();
                vk::ImageMemoryBarrier2::builder()
                    .src_access_mask(barrier.src.access)
                    .dst_access_mask(barrier.dst.access)
                    .src_stage_mask(barrier.src.stage)
                    .dst_stage_mask(barrier.dst.stage)
                    .old_layout(old_layout)
                    .new_layout(barrier.new_layout)
                    .image(**barrier.image)
                    .subresource_range(subresource_range)
                    .build()
            })
            .collect();
        let buffer_barriers: Vec<_> = buffer_barriers
            .iter()
            .map(|barrier| {
                vk::BufferMemoryBarrier2::builder()
                    .src_access_mask(barrier.src.access)
                    .dst_access_mask(barrier.dst.access)
                    .src_stage_mask(barrier.src.stage)
                    .dst_stage_mask(barrier.dst.stage)
                    .buffer(**barrier.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
                    .build()
            })
            .collect();
        let dependency_info = vk::DependencyInfo::builder()
            .image_memory_barriers(&image_barriers)
            .buffer_memory_barriers(&buffer_barriers);
        unsafe {
            device.cmd_pipeline_barrier2(self.buffer, &dependency_info);
        }
        self
    }

    pub fn fill_buffer(
        &mut self,
        device: &Device,
        buffer: &Buffer,
        value: u32,
    ) -> &mut Self {
        unsafe {
            device.cmd_fill_buffer(**self, **buffer, 0, buffer.size, value);
        }
        self
    }

    pub fn push_constants(
        &mut self,
        device: &Device,
        pipeline: &Pipeline,
        bytes: &[u8],
    ) -> &mut Self {
        unsafe {
            device.cmd_push_constants(
                self.buffer,
                pipeline.layout,
                pipeline.push_constant_stages,
                0,
                bytes,
            )
        }
        self
    }

    pub fn bind_index_buffer(
        &mut self,
        device: &Device,
        buffer: &Buffer,
    ) -> &mut Self {
        let format = vk::IndexType::UINT32;
        unsafe {
            device.cmd_bind_index_buffer(self.buffer, **buffer, 0, format);
        }
        self
    }

    pub fn draw_indexed(
        &mut self,
        device: &Device,
        draw: &DrawIndexed,
    ) -> &mut Self {
        unsafe {
            device.cmd_draw_indexed(
                self.buffer,
                draw.index_count,
                draw.instance_count,
                draw.first_index,
                draw.vertex_offset,
                draw.first_instance,
            );
        }
        self
    }

    pub fn draw_indexed_indirect(
        &mut self,
        device: &Device,
        buffer: &Buffer,
        count: u32,
        stride: u32,
        offset: u64,
    ) -> &mut Self {
        unsafe {
            device.cmd_draw_indexed_indirect(
                **self, **buffer, offset, count, stride,
            )
        }
        self
    }

    pub fn draw_indexed_indirect_count(
        &mut self,
        device: &Device,
        buffer: &Buffer,
        count_buffer: &Buffer,
        max_count: u32,
        stride: u32,
        offset: u64,
    ) -> &mut Self {
        unsafe {
            let count_offset = 0;
            device.cmd_draw_indexed_indirect_count(
                **self,
                **buffer,
                offset,
                **count_buffer,
                count_offset,
                max_count,
                stride,
            )
        }
        self
    }

    pub fn begin_rendering(
        &mut self,
        device: &Device,
        begin: &BeginRendering,
    ) -> &mut Self {
        let color_attachments: Vec<_> = begin
            .color_attachments
            .iter()
            .map(|attachment| {
                let mut attachment_info =
                    vk::RenderingAttachmentInfo::builder()
                        .image_view(**attachment.view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .load_op(attachment.load.load_op())
                        .build();
                if let Load::Clear(clear_value) = attachment.load {
                    attachment_info.clear_value = clear_value;
                }
                attachment_info
            })
            .collect();
        let mut rendering_info = vk::RenderingInfo::builder()
            .color_attachments(&color_attachments)
            .layer_count(1)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: begin.extent,
            });
        #[allow(unused_assignments)]
        let mut depth_attachment = vk::RenderingAttachmentInfo::default();
        if let Some(attachment) = &begin.depth_attachment {
            depth_attachment = vk::RenderingAttachmentInfo::builder()
                .image_view(**attachment.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .store_op(vk::AttachmentStoreOp::STORE)
                .load_op(attachment.load.load_op())
                .build();
            if let Load::Clear(clear_value) = attachment.load {
                depth_attachment.clear_value = clear_value;
            }
            rendering_info = rendering_info.depth_attachment(&depth_attachment)
        }
        unsafe {
            device.cmd_begin_rendering(self.buffer, &rendering_info);
        }
        self
    }

    pub fn end_rendering(&mut self, device: &Device) -> &mut Self {
        unsafe {
            device.cmd_end_rendering(self.buffer);
        }
        self
    }

    pub fn set_viewport(
        &mut self,
        device: &Device,
        viewports: &[vk::Viewport],
    ) -> &mut Self {
        unsafe {
            device.cmd_set_viewport(self.buffer, 0, viewports);
        }
        self
    }

    pub fn set_scissor(
        &mut self,
        device: &Device,
        scissors: &[vk::Rect2D],
    ) -> &mut Self {
        unsafe {
            device.cmd_set_scissor(self.buffer, 0, scissors);
        }
        self
    }

    pub fn blit_image(
        &mut self,
        device: &Device,
        blit: &ImageBlit,
    ) -> &mut Self {
        let region = vk::ImageBlit2::builder()
            .src_offsets(blit.src_offsets)
            .dst_offsets(blit.dst_offsets)
            .src_subresource(
                vk::ImageSubresourceLayers::builder()
                    .base_array_layer(0)
                    .layer_count(1)
                    .mip_level(blit.src_mip_level)
                    .aspect_mask(blit.src.aspect)
                    .build(),
            )
            .dst_subresource(
                vk::ImageSubresourceLayers::builder()
                    .base_array_layer(0)
                    .layer_count(1)
                    .mip_level(blit.dst_mip_level)
                    .aspect_mask(blit.dst.aspect)
                    .build(),
            );
        let blit_info = vk::BlitImageInfo2::builder()
            .src_image(**blit.src)
            .dst_image(**blit.dst)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .filter(blit.filter)
            .regions(slice::from_ref(&region));
        unsafe {
            device.cmd_blit_image2(self.buffer, &blit_info);
        }
        self
    }
}

pub(crate) enum Load {
    Clear(vk::ClearValue),
    Load,
}

impl Load {
    fn load_op(&self) -> vk::AttachmentLoadOp {
        match self {
            Load::Clear(_) => vk::AttachmentLoadOp::CLEAR,
            Load::Load => vk::AttachmentLoadOp::LOAD,
        }
    }
}

pub(crate) struct ImageBlit<'a> {
    pub src: &'a Image,
    pub dst: &'a Image,
    pub src_offsets: [vk::Offset3D; 2],
    pub dst_offsets: [vk::Offset3D; 2],
    pub src_mip_level: u32,
    pub dst_mip_level: u32,
    pub filter: vk::Filter,
}

pub(crate) struct Attachment<'a> {
    pub view: &'a ImageView,
    pub load: Load,
}

pub(crate) struct BeginRendering<'a> {
    pub color_attachments: &'a [Attachment<'a>],
    pub depth_attachment: Option<Attachment<'a>>,
    pub extent: vk::Extent2D,
}

pub(crate) struct DrawIndexed {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub first_instance: u32,
    pub vertex_offset: i32,
}

#[derive(Default)]
pub(crate) enum MipLevels {
    #[default]
    All,
    Levels {
        base: u32,
        count: u32,
    },
}

pub(crate) struct ImageBarrier<'a> {
    pub image: &'a Image,
    pub new_layout: vk::ImageLayout,
    pub mip_levels: MipLevels,
    pub src: Access,
    pub dst: Access,
}

pub(crate) struct BufferBarrier<'a> {
    pub buffer: &'a Buffer,
    pub src: Access,
    pub dst: Access,
}

pub(crate) struct ImageLayouts {
    pub layout: vk::ImageLayout,
    pub src: Access,
    pub dst: Access,
}

#[derive(Clone, Copy)]
pub(crate) struct Access {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

impl std::ops::BitOr for Access {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self { stage: self.stage | rhs.stage, access: self.access | rhs.access }
    }
}

impl Access {
    pub const TRANSFER_DST: Self = Self {
        stage: vk::PipelineStageFlags2::TRANSFER,
        access: vk::AccessFlags2::TRANSFER_WRITE,
    };

    pub const COMPUTE_WRITE: Self = Self {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_STORAGE_WRITE,
    };

    pub const COMPUTE_READ: Self = Self {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_STORAGE_READ,
    };

    pub const INDIRECT_READ: Self = Self {
        stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
        access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
    };

    pub const DEPTH_BUFFER_RENDER: Access = Access {
        stage: vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
    };

    pub const COLOR_BUFFER_RENDER: Access = Access {
        stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
    };

    pub const DEPTH_BUFFER_READ: Access = Access {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_SAMPLED_READ,
    };

    pub const NONE: Self = Self {
        stage: vk::PipelineStageFlags2::NONE,
        access: vk::AccessFlags2::NONE,
    };

    pub const ALL: Self = Self {
        stage: vk::PipelineStageFlags2::ALL_COMMANDS,
        access: vk::AccessFlags2::from_raw(
            vk::AccessFlags2::MEMORY_READ.as_raw()
                | vk::AccessFlags2::MEMORY_WRITE.as_raw(),
        ),
    };
}

pub(crate) fn quickie<'a, F, R>(device: &Device, f: F) -> Result<R>
where
    F: FnOnce(&mut CommandBuffer<'a>) -> Result<R>,
{
    let mut buffer = CommandBuffer::new(device)?;
    buffer.begin(device)?;
    let result = f(&mut buffer)?;
    buffer.end(device)?;
    submit(device, &[], &[], [&buffer])?;
    device.wait_until_idle()?;
    buffer.destroy(device);
    Ok(result)
}

pub(crate) fn frame<'a, F, R>(
    device: &Device,
    sync: &Sync,
    f: F,
) -> Result<(CommandBuffer<'a>, R)>
where
    F: FnOnce(&mut CommandBuffer<'a>) -> Result<R>,
{
    let mut buffer = CommandBuffer::new(device)?;
    buffer.begin(device)?;
    let ret = f(&mut buffer)?;
    buffer.end(device)?;
    submit(device, &[sync.acquire], &[sync.release], [&buffer])?;
    Ok((buffer, ret))
}

fn submit<'a>(
    device: &Device,
    wait: &[vk::Semaphore],
    signal: &[vk::Semaphore],
    buffers: impl IntoIterator<Item = &'a CommandBuffer<'a>>,
) -> Result<()> {
    let buffers: Vec<_> =
        buffers.into_iter().map(|buffer| buffer.buffer).collect();
    let submit_info = vk::SubmitInfo::builder()
        .wait_dst_stage_mask(slice::from_ref(
            &vk::PipelineStageFlags::ALL_COMMANDS,
        ))
        .wait_semaphores(wait)
        .signal_semaphores(signal)
        .command_buffers(&buffers);
    unsafe {
        device
            .queue_submit(
                device.queue,
                slice::from_ref(&submit_info),
                vk::Fence::null(),
            )
            .wrap_err("failed to submit command buffers")
    }
}
