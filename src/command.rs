use ash::vk;
use anyhow::Result;
use smallvec::SmallVec;

use std::{mem, ops};
use std::cell::UnsafeCell;
use std::rc::Rc;

use crate::core::*;
use crate::resource::*;
use rendi_res::{Res, DummyRes};

pub struct CommandBuffer {
    pub handle: vk::CommandBuffer,

    /// This keeps track of all the items used by the command buffer.
    ///
    /// It's not free since clearing this means jumping through a pointer for each item, but it
    /// makes sure the items live as long as they are used by the buffer.
    bound_resources: UnsafeCell<Vec<DummyRes>>,

    pub queue: Res<Queue>,
    device: Rc<Device>,
}

pub enum SubmitCount {
    OneTime,
    Multiple,
}

impl CommandBuffer {
    pub fn new(device: Rc<Device>, queue: Res<Queue>) -> Result<Self> {
        let info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(queue.pool)
            .command_buffer_count(1);

        let bound_resources = UnsafeCell::new(Vec::new());
        let handles = unsafe {
            device.handle.allocate_command_buffers(&info)?
        };

        Ok(Self { handle: *handles.first().unwrap(), queue, device, bound_resources })
    }

    pub fn reset(&self) -> Result<()> {
        unsafe {
            let flags = vk::CommandBufferResetFlags::empty();
            self.device.handle.reset_command_buffer(self.handle, flags)?;

            // We can clear all the bound items when the buffer is reset.
            (*self.bound_resources.get()).clear();
        }

        Ok(())
    }

    fn bind_resource<T>(&self, res: Res<T>) {
        unsafe { (*self.bound_resources.get()).push(DummyRes::new(res)); }
    }

    pub fn record<F, R>(&self, submit_count: SubmitCount, func: F) -> Result<R>
    where
        F: FnOnce(&CommandRecorder) -> R
    {
        let flags = match submit_count {
            SubmitCount::OneTime => vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            SubmitCount::Multiple => vk::CommandBufferUsageFlags::empty(),
        };

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::builder().flags(flags);
            self.device.handle.begin_command_buffer(self.handle, &begin_info)?;
        }

        let recorder = CommandRecorder { buffer: &self };
        let ret = func(&recorder);

        unsafe {
            self.device.handle.end_command_buffer(self.handle)?;
        }

        Ok(ret)
    }

    pub fn submit_wait_idle(&self) -> Result<()> {
        self.queue.submit_wait_idle(self)
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        self.queue.wait_idle().expect("failed to wait for idle queue");

        unsafe {
            self.device.handle.free_command_buffers(self.queue.pool, &[self.handle]);
        }
    }
}

pub struct IndexedDrawInfo {
    pub instance: u32,
    pub index_count: u32,
    pub index_start: u32,
    pub vertex_offset: i32,
}

pub struct IndexedIndirectDrawInfo {
    pub draw_buffer: Res<Buffer>,
    pub count_buffer: Res<Buffer>,

    pub draw_command_size: vk::DeviceSize,
    pub draw_offset: vk::DeviceSize,
    pub count_offset: vk::DeviceSize,

    pub max_draw_count: u32,
}

pub struct BufferBarrierInfo {
    pub buffer: Res<Buffer>,

    pub src_mask: vk::AccessFlags2,
    pub dst_mask: vk::AccessFlags2,
    pub src_stage: vk::PipelineStageFlags2,
    pub dst_stage: vk::PipelineStageFlags2,
}

pub struct ImageBarrierInfo {
    pub image: Res<Image>,

    pub flags: vk::DependencyFlags,

    pub src_mask: vk::AccessFlags2,
    pub dst_mask: vk::AccessFlags2,
    pub src_stage: vk::PipelineStageFlags2,
    pub dst_stage: vk::PipelineStageFlags2,

    pub new_layout: vk::ImageLayout,

    pub mips: ops::Range<u32>,
}

pub struct ImageResolveInfo {
    pub src: Res<Image>,
    pub dst: Res<Image>,
    pub src_mip: u32,
    pub dst_mip: u32,
}

pub struct ImageBlitInfo {
    pub src: Res<Image>,
    pub dst: Res<Image>,
    pub filter: vk::Filter,
    pub src_mip: u32,
    pub dst_mip: u32,
}

pub struct DescBindInfo<'a> {
    pub bind_point: vk::PipelineBindPoint,
    pub layout: Res<PipelineLayout>,
    pub descs: &'a [Res<DescSet>],
}

pub struct RenderInfo {
    pub color_target: Option<Res<ImageView>>,
    pub depth_target: Res<ImageView>,
    
    pub color_load_op: vk::AttachmentLoadOp,
    pub depth_load_op: vk::AttachmentLoadOp,

    pub swapchain: Res<Swapchain>,
}

pub struct PushConst<'a> {
    pub stage: vk::ShaderStageFlags,
    pub bytes: &'a [u8],
}

macro_rules! impl_shared_commands {
    () => {
        #[allow(unused)]
        pub fn push_consts(&self, layout: Res<PipelineLayout>, consts: &[PushConst]) {
            let mut offset = 0;

            for (val, range) in consts.iter().zip(layout.push_const_ranges.iter()) {
                assert!(
                    (val.bytes.len() as vk::DeviceSize) <= range.size,
                    "byte size is greater than specified in pipeline layout, {} vs {}",
                    val.bytes.len(),
                    range.size,
                );

                unsafe {
                    self.device().handle.cmd_push_constants(
                        self.buffer.handle,
                        layout.handle,
                        val.stage,
                        offset,
                        val.bytes,
                    );

                    offset += range.size as u32;
                }
            }

            self.buffer.bind_resource(layout);
        }

        #[allow(unused)]
        pub fn bind_descs(&self, info: &DescBindInfo) {
            let descs: SmallVec<[_; 12]> = info.descs
                .iter()
                .map(|desc| desc.handle)
                .collect();

            unsafe {
                self.device().handle.cmd_bind_descriptor_sets(
                    self.buffer.handle,
                    info.bind_point,
                    info.layout.handle,
                    0,
                    &descs,
                    &[],
                );
            }

            self.buffer.bind_resource(info.layout.clone());

            for desc in info.descs {
                self.buffer.bind_resource(desc.clone());
            }
        }

        #[allow(unused)]
        pub fn image_barrier(&self, info: &ImageBarrierInfo) {
            let subresource = vk::ImageSubresourceRange::builder()
                .aspect_mask(info.image.aspect_flags())
                .base_mip_level(info.mips.start)
                .level_count(info.mips.end - info.mips.start)
                .base_array_layer(0)
                .layer_count(info.image.layer_count())
                .build();
            let barriers = [vk::ImageMemoryBarrier2::builder()
                .image(info.image.handle)
                .old_layout(info.image.layout())
                .new_layout(info.new_layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(info.src_mask)
                .dst_access_mask(info.dst_mask)
                .src_stage_mask(info.src_stage)
                .dst_stage_mask(info.dst_stage)
                .subresource_range(subresource)
                .build()];

            info.image.layout.set(info.new_layout);

            let dependency_info = vk::DependencyInfo::builder()
                .dependency_flags(info.flags)
                .image_memory_barriers(&barriers);

            unsafe {
                self.device().handle.cmd_pipeline_barrier2(self.buffer.handle, &dependency_info);
            }

            self.buffer.bind_resource(info.image.clone());
        }

        #[allow(unused)]
        pub fn buffer_barrier(&self, info: &BufferBarrierInfo) {
            let barriers = [vk::BufferMemoryBarrier2::builder()
                .src_access_mask(info.src_mask)
                .dst_access_mask(info.dst_mask)
                .src_stage_mask(info.src_stage)
                .dst_stage_mask(info.dst_stage)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(info.buffer.handle)
                .offset(0)
                .size(info.buffer.size())
                .build()];

            let dependency_info = vk::DependencyInfo::builder()
                .dependency_flags(vk::DependencyFlags::empty())
                .buffer_memory_barriers(&barriers);

            unsafe {
                self.device().handle.cmd_pipeline_barrier2(self.buffer.handle, &dependency_info);
            }

            self.buffer.bind_resource(info.buffer.clone());
        }
    }
}

pub struct DrawRecorder<'a> {
    buffer: &'a CommandBuffer,
}

impl<'a> DrawRecorder<'a> {
    impl_shared_commands!();

    fn device(&self) -> &Device {
        &self.buffer.device
    }

    pub fn bind_vertex_buffer(&self, buffer: Res<Buffer>) {
        unsafe {
            self.device().handle.cmd_bind_vertex_buffers(
                self.buffer.handle,
                0,
                &[buffer.handle],
                &[0],
            );
        }

        self.buffer.bind_resource(buffer);
    }

    pub fn bind_index_buffer(&self, buffer: Res<Buffer>, index_type: vk::IndexType) {
        unsafe {
            self.device().handle.cmd_bind_index_buffer(
                self.buffer.handle,
                buffer.handle,
                0,
                index_type,
            );
        }

        self.buffer.bind_resource(buffer);
    }

    pub fn bind_raster_pipeline(&self, pipeline: Res<RasterPipeline>) {
        let bind_point = vk::PipelineBindPoint::GRAPHICS;
        unsafe {
            self.device().handle.cmd_bind_pipeline(
                self.buffer.handle,
                bind_point,
                pipeline.handle,
            );
        }

        self.buffer.bind_resource(pipeline);
    }

    pub fn draw(&self, vertex_count: u32, vertex_start: u32) {
        unsafe {
            self.device().handle.cmd_draw(
                self.buffer.handle,
                vertex_count,
                1,
                vertex_start,
                0,
            );
        }
    }

    pub fn draw_indexed(&self, info: IndexedDrawInfo) {
        unsafe {
            self.device().handle.cmd_draw_indexed(
                self.buffer.handle,
                info.index_count,
                1,
                info.index_start,
                info.vertex_offset,
                info.instance,
            );
        }
    }

    pub fn draw_indexed_indirect_count(&self, info: &IndexedIndirectDrawInfo) {
        unsafe {
            self.device().handle.cmd_draw_indexed_indirect_count(
                self.buffer.handle,
                info.draw_buffer.handle,
                info.draw_offset,
                info.count_buffer.handle,
                info.count_offset,
                info.max_draw_count,
                info.draw_command_size as u32,
            );
        }

        self.buffer.bind_resource(info.draw_buffer.clone());
        self.buffer.bind_resource(info.count_buffer.clone());
    }
}

pub struct CommandRecorder<'a> {
    buffer: &'a CommandBuffer,
}

impl<'a> CommandRecorder<'a> {
    impl_shared_commands!();

    fn device(&self) -> &Device {
        &self.buffer.device
    }

    pub fn copy_buffers(&self, src: Res<Buffer>, dst: Res<Buffer>) {
        let size = src.size().min(dst.size());
        let regions = [vk::BufferCopy2::builder()
            .src_offset(0)
            .dst_offset(0)
            .size(size)
            .build()];

        let info = vk::CopyBufferInfo2::builder()
            .src_buffer(src.handle)
            .dst_buffer(dst.handle)
            .regions(&regions);

        unsafe { 
            self.device().handle.cmd_copy_buffer2(self.buffer.handle, &info);
        }

        self.buffer.bind_resource(src);
        self.buffer.bind_resource(dst);
    }

    pub fn copy_buffer_to_image(&self, src: Res<Buffer>, dst: Res<Image>, mip_level: u32) {
        let subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(dst.aspect_flags())
            .mip_level(mip_level)
            .base_array_layer(0)
            .layer_count(dst.layer_count())
            .build();
        
        let extent = dst.extent(mip_level);

        let regions = [vk::BufferImageCopy2::builder()
            .buffer_offset(0)
            .buffer_row_length(extent.width)
            .buffer_image_height(0)
            .image_extent(extent)
            .image_subresource(subresource)
            .build()];

        let info = vk::CopyBufferToImageInfo2::builder()
            .src_buffer(src.handle)
            .dst_image(dst.handle)
            .dst_image_layout(dst.layout())
            .regions(&regions);

        unsafe {
            self.device().handle.cmd_copy_buffer_to_image2(self.buffer.handle, &info);
        }

        self.buffer.bind_resource(src);
        self.buffer.bind_resource(dst);
    }

    pub fn resolve_image(&self, info: &ImageResolveInfo) {
        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(info.src.aspect_flags())
            .base_array_layer(0)
            .layer_count(1)
            .mip_level(info.src_mip)
            .build();
        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(info.dst.aspect_flags())
            .base_array_layer(0)
            .layer_count(1)
            .mip_level(info.dst_mip)
            .build();
        let regions = [vk::ImageResolve2::builder()
            .src_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .dst_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .src_subresource(src_subresource)
            .dst_subresource(dst_subresource)
            .extent(info.src.extent(info.src_mip))
            .build()];
        let resolve_info = vk::ResolveImageInfo2::builder()
           .src_image(info.src.handle)
           .dst_image(info.dst.handle)
           .src_image_layout(info.src.layout())
           .dst_image_layout(info.dst.layout())
           .regions(&regions);

        unsafe {
            self.device().handle.cmd_resolve_image2(self.buffer.handle, &resolve_info);
        }

        self.buffer.bind_resource(info.src.clone());
        self.buffer.bind_resource(info.dst.clone());
    }

    pub fn blit_image(&self, info: &ImageBlitInfo) {
        let src_extent = info.src.extent(info.src_mip);
        let dst_extent = info.dst.extent(info.dst_mip);

        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(info.src.aspect_flags())
            .base_array_layer(0)
            .layer_count(1)
            .mip_level(info.src_mip)
            .build();
        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(info.dst.aspect_flags())
            .base_array_layer(0)
            .layer_count(1)
            .mip_level(info.dst_mip)
            .build();
        let regions = [vk::ImageBlit2::builder()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: src_extent.width as i32,
                    y: src_extent.height as i32,
                    z: 1,
                },
            ])
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: dst_extent.width as i32,
                    y: dst_extent.height as i32,
                    z: 1,
                },
            ])
            .src_subresource(src_subresource)
            .dst_subresource(dst_subresource)
            .build()];
        let blit_info = vk::BlitImageInfo2::builder()
           .src_image(info.src.handle)
           .dst_image(info.dst.handle)
           .src_image_layout(info.src.layout())
           .dst_image_layout(info.dst.layout())
           .filter(info.filter)
           .regions(&regions);

        unsafe {
            self.device().handle.cmd_blit_image2(self.buffer.handle, &blit_info);
        }

        self.buffer.bind_resource(info.src.clone());
        self.buffer.bind_resource(info.dst.clone());
    }

    pub fn render<F: FnOnce(&DrawRecorder)>(&self, info: &RenderInfo, f: F) {
        let color_attachments: SmallVec<[_; 1]> = if let Some(target) = &info.color_target {
            SmallVec::from([
                vk::RenderingAttachmentInfo::builder()
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image_view(target.handle)
                    .load_op(info.color_load_op)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    })
                    .build()
            ])
        } else {
            SmallVec::new()
        };

        let depth_resolve_attachment = vk::RenderingAttachmentInfo::builder()
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .image_view(info.depth_target.handle)
            .load_op(info.depth_load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            });

        let rendering_info = vk::RenderingInfo::builder()
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_resolve_attachment)
            .layer_count(1)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: info.swapchain.extent(),
            });

        let viewports = info.swapchain.viewports();
        let scissors = info.swapchain.scissors();

        unsafe {
            self.device().handle.cmd_begin_rendering(self.buffer.handle, &rendering_info);
            self.device().handle.cmd_set_viewport(self.buffer.handle, 0, &viewports);
            self.device().handle.cmd_set_scissor(self.buffer.handle, 0, &scissors);
        }

        let draw_recorder = DrawRecorder {
            buffer: self.buffer,
        };
        
        f(&draw_recorder);

        unsafe {
            self.device().handle.cmd_end_rendering(self.buffer.handle);
        };
    }

    pub fn dispatch(&self, pipeline: Res<ComputePipeline>, group_count: [u32; 3]) {
        let bind_point = vk::PipelineBindPoint::COMPUTE;
        unsafe {
            self.device().handle.cmd_bind_pipeline(
                self.buffer.handle,
                bind_point,
                pipeline.handle,
            );
            self.device().handle.cmd_dispatch(
                self.buffer.handle,
                group_count[0],
                group_count[1],
                group_count[2],
            );
        }

        self.buffer.bind_resource(pipeline);
    }

    pub fn update_buffer<T: bytemuck::NoUninit>(&self, buffer: Res<Buffer>, val: &T) {
        assert_eq!(
            buffer.size(),
            mem::size_of::<T>() as vk::DeviceSize,
            "size of buffer doesn't match length of data"
        );

        unsafe {
            self.device().handle.cmd_update_buffer(
                self.buffer.handle,
                buffer.handle,
                0,
                bytemuck::bytes_of(val),
            );
        }

        self.buffer.bind_resource(buffer);
    }
}

