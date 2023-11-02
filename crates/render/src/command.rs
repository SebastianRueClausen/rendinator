use std::collections::HashMap;
use std::ops::Deref;
use std::slice;

use ash::vk;
use eyre::{Context, Result};

use crate::device::Device;
use crate::resources::Image;
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

    pub fn begin(&self, device: &Device) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::builder();
        unsafe {
            device
                .begin_command_buffer(self.buffer, &begin_info)
                .wrap_err("failed to begin command buffer")
        }
    }

    pub fn end(&mut self, device: &Device) -> Result<()> {
        for (image, layout) in self.image_layouts.drain() {
            image.set_layout(layout);
        }

        unsafe {
            device
                .end_command_buffer(self.buffer)
                .wrap_err("failed to end command buffer")
        }
    }

    pub fn dispatch(&self, device: &Device, x: u32, y: u32, z: u32) {
        unsafe { device.cmd_dispatch(self.buffer, x, y, z) }
    }

    pub fn pipeline_barriers(
        &mut self,
        device: &Device,
        image_barriers: &[ImageBarrier<'a>],
    ) {
        let image_barriers: Vec<_> = image_barriers
            .iter()
            .map(|barrier| {
                let old_layout = self
                    .image_layouts
                    .get(barrier.image)
                    .copied()
                    .unwrap_or(barrier.image.layout());
                self.image_layouts.insert(barrier.image, barrier.new_layout);
                let subresource_range = vk::ImageSubresourceRange::builder()
                    .aspect_mask(barrier.image.aspect)
                    .base_mip_level(0)
                    .base_array_layer(0)
                    .level_count(1)
                    .layer_count(1)
                    .build();
                vk::ImageMemoryBarrier2::builder()
                    .src_access_mask(barrier.src_access)
                    .dst_access_mask(barrier.dst_access)
                    .src_stage_mask(barrier.src_stage)
                    .dst_stage_mask(barrier.dst_stage)
                    .old_layout(old_layout)
                    .new_layout(barrier.new_layout)
                    .image(**barrier.image)
                    .subresource_range(subresource_range)
                    .build()
            })
            .collect();
        let dependency_info = vk::DependencyInfo::builder()
            .image_memory_barriers(&image_barriers);
        unsafe { device.cmd_pipeline_barrier2(self.buffer, &dependency_info) }
    }
}

pub(crate) struct ImageBarrier<'a> {
    pub image: &'a Image,
    pub new_layout: vk::ImageLayout,
    pub src_stage: vk::PipelineStageFlags2,
    pub dst_stage: vk::PipelineStageFlags2,
    pub src_access: vk::AccessFlags2,
    pub dst_access: vk::AccessFlags2,
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

pub(crate) fn frame<'a, F>(
    device: &Device,
    sync: &Sync,
    f: F,
) -> Result<CommandBuffer<'a>>
where
    F: FnOnce(&mut CommandBuffer<'a>) -> Result<()>,
{
    let mut buffer = CommandBuffer::new(device)?;
    buffer.begin(device)?;
    f(&mut buffer)?;
    buffer.end(device)?;
    submit(device, &[sync.acquire], &[sync.release], [&buffer])?;
    Ok(buffer)
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
