use std::mem;

use ash::vk;
use eyre::Result;
use glam::UVec2;

use crate::device::Device;
use crate::resources::{
    self, Buffer, BufferKind, BufferRequest, BufferWrite, Memory,
};
use crate::swapchain::Swapchain;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
pub(crate) struct ConstantData {
    pub screen_size: UVec2,
}

impl ConstantData {
    pub fn new(swapchain: &Swapchain, _prev: Option<Self>) -> Self {
        Self {
            screen_size: UVec2 {
                x: swapchain.extent.width,
                y: swapchain.extent.height,
            },
        }
    }
}

pub(crate) struct Constants {
    pub data: ConstantData,
    pub buffer: Buffer,
    pub memory: Memory,
}

impl Constants {
    pub fn new(device: &Device, swapchain: &Swapchain) -> Result<Self> {
        let buffer = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of::<ConstantData>() as vk::DeviceSize,
                kind: BufferKind::Uniform,
            },
        )?;
        let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let memory = resources::buffer_memory(device, &buffer, memory_flags)?;
        let data = ConstantData::new(swapchain, None);
        Ok(Self { buffer, memory, data })
    }

    pub fn update(&mut self, swapchain: &Swapchain) {
        self.data = ConstantData::new(swapchain, Some(self.data));
    }

    pub fn buffer_write(&self) -> BufferWrite {
        BufferWrite {
            buffer: &self.buffer,
            data: bytemuck::bytes_of(&self.data),
        }
    }

    pub fn destroy(&self, device: &Device) {
        self.buffer.destroy(device);
        self.memory.free(device);
    }
}
