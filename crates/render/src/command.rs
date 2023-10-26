use std::ops::Deref;
use std::slice;

use ash::vk;
use eyre::{Context, Result};

use crate::device::Device;

pub(crate) struct CommandBuffer {
    buffer: vk::CommandBuffer,
}

impl Deref for CommandBuffer {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl CommandBuffer {
    pub fn new(device: &Device) -> Result<Self> {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(device.command_pool)
            .command_buffer_count(1);
        let buffers =
            unsafe { device.allocate_command_buffers(&allocate_info)? };
        Ok(Self { buffer: buffers[0] })
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

    pub fn end(&self, device: &Device) -> Result<()> {
        unsafe {
            device
                .end_command_buffer(self.buffer)
                .wrap_err("failed to end command buffer")
        }
    }
}

pub(crate) fn quickie<F, R>(device: &Device, f: F) -> Result<R>
where
    F: FnOnce(&CommandBuffer) -> Result<R>,
{
    let buffer = CommandBuffer::new(device)?;

    buffer.begin(device)?;
    let result = f(&buffer)?;
    buffer.end(device)?;

    submit_and_wait(device, [&buffer])?;
    buffer.destroy(device);

    Ok(result)
}

pub(crate) fn submit_and_wait<'a>(
    device: &Device,
    buffers: impl IntoIterator<Item = &'a CommandBuffer>,
) -> Result<()> {
    let buffers: Vec<_> =
        buffers.into_iter().map(|buffer| buffer.buffer).collect();
    let submit_info = vk::SubmitInfo::builder().command_buffers(&buffers);
    let result = unsafe {
        device.queue_submit(
            device.queue,
            slice::from_ref(&submit_info),
            vk::Fence::null(),
        )
    };
    result.wrap_err("failed to submit command buffers")?;
    device.wait_until_idle();
    Ok(())
}
