use std::mem;

use ash::vk;
use asset::DirectionalLight;
use eyre::Result;
use glam::{Mat4, UVec2, Vec4};

use crate::camera::Camera;
use crate::device::Device;
use crate::resources::{
    self, Buffer, BufferKind, BufferRequest, BufferWrite, Memory,
};
use crate::swapchain::Swapchain;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
pub(crate) struct ConstantData {
    pub proj: Mat4,
    pub view: Mat4,
    pub proj_view: Mat4,
    pub camera_position: Vec4,
    pub frustrum_planes: [Vec4; 6],
    pub sun: DirectionalLight,
    pub screen_size: UVec2,
    pub z_near: f32,
    pub z_far: f32,
}

impl ConstantData {
    pub fn new(
        swapchain: &Swapchain,
        camera: &Camera,
        _prev: Option<Self>,
    ) -> Self {
        let sun = DirectionalLight {
            direction: Vec4::new(0.0, 1.0, 0.0, 0.0),
            irradiance: Vec4::splat(6.0),
        };
        Self {
            screen_size: UVec2 {
                x: swapchain.extent.width,
                y: swapchain.extent.height,
            },
            proj: camera.proj,
            view: camera.view,
            frustrum_planes: camera.frustrum_planes(),
            camera_position: camera.position.extend(0.0),
            proj_view: camera.proj * camera.view,
            z_near: camera.z_near,
            z_far: camera.z_far,
            sun,
        }
    }
}

pub(crate) struct Constants {
    pub data: ConstantData,
    pub buffer: Buffer,
    pub memory: Memory,
}

impl Constants {
    pub fn new(
        device: &Device,
        swapchain: &Swapchain,
        camera: &Camera,
    ) -> Result<Self> {
        let buffer = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of::<ConstantData>() as vk::DeviceSize,
                kind: BufferKind::Uniform,
            },
        )?;
        let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let memory = resources::buffer_memory(device, &buffer, memory_flags)?;
        let data = ConstantData::new(swapchain, camera, None);
        Ok(Self { buffer, memory, data })
    }

    pub fn update(&mut self, swapchain: &Swapchain, camera: &Camera) {
        self.data = ConstantData::new(swapchain, camera, Some(self.data));
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
