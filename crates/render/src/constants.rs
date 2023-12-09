use std::mem;

use ash::vk;
use asset::DirectionalLight;
use eyre::Result;
use glam::{Mat4, UVec2, Vec4};

use crate::camera::Camera;
use crate::hal;

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
        swapchain: &hal::Swapchain,
        camera: &Camera,
        _prev: Option<Self>,
    ) -> Self {
        let sun = DirectionalLight {
            direction: Vec4::new(1.0, 1.0, 0.0, 0.0).normalize(),
            irradiance: Vec4::splat(6.0),
        };
        let proj = camera.proj();
        Self {
            screen_size: UVec2 {
                x: swapchain.extent.width,
                y: swapchain.extent.height,
            },
            proj,
            view: camera.view,
            frustrum_planes: camera.frustrum_planes(),
            camera_position: camera.position.extend(0.0),
            proj_view: proj * camera.view,
            z_near: camera.z_near,
            z_far: camera.z_far,
            sun,
        }
    }
}

pub(crate) struct Constants {
    pub data: ConstantData,
    pub buffer: hal::Buffer,
    pub memory: hal::Memory,
}

impl Constants {
    pub fn new(
        device: &hal::Device,
        swapchain: &hal::Swapchain,
        camera: &Camera,
    ) -> Result<Self> {
        let buffer = hal::Buffer::new(
            device,
            &hal::BufferRequest {
                size: mem::size_of::<ConstantData>() as vk::DeviceSize,
                kind: hal::BufferKind::Uniform,
            },
        )?;
        let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let memory = hal::buffer_memory(device, &buffer, memory_flags)?;
        let data = ConstantData::new(swapchain, camera, None);
        Ok(Self { buffer, memory, data })
    }

    pub fn update(&mut self, swapchain: &hal::Swapchain, camera: &Camera) {
        self.data = ConstantData::new(swapchain, camera, Some(self.data));
    }

    pub fn buffer_write(&self) -> hal::BufferWrite {
        hal::BufferWrite {
            buffer: &self.buffer,
            data: bytemuck::bytes_of(&self.data),
        }
    }

    pub fn destroy(&self, device: &hal::Device) {
        self.buffer.destroy(device);
        self.memory.free(device);
    }
}
