use ash::vk;
use glam::{Mat4, Vec2, Vec3, Vec4};
use winit::event::VirtualKeyCode;
use anyhow::Result;

use std::{mem, array};
use std::time::Duration;

use crate::core::*;
use crate::InputState;
use crate::resource::{Buffer, BufferReq, ResourcePool, Res};

pub struct Camera {
    pub pos: Vec3,
    front: Vec3,
    up: Vec3,
    pub proj: Mat4,
    pub view: Mat4,
    yaw: f32,
    pitch: f32,
    pub fov: f32,
    pub z_near: f32,
    pub z_far: f32,
    rotation_speed: f32,
    movement_speed: f32,
}

impl Camera {
    pub fn new(aspect_ratio: f32) -> Self {
        let pos = Vec3::new(10.0, 10.0, 10.0);
        let up = Vec3::new(0.0, -1.0, 0.0);
        let front = Vec3::default();

        let yaw = 0.0;
        let pitch = 0.0;

        let fov = 66.0_f32;
        let z_near = 0.1;
        let z_far = 100.0;

        let view = Mat4::look_at_rh(pos, pos + front, up);
        let proj = Mat4::perspective_rh(fov.to_radians(), aspect_ratio, z_near, z_far);

        Self {
            rotation_speed: 0.5,
            movement_speed: 60.0,
            pos,
            up,
            front,
            yaw,
            pitch,
            fov,
            z_near,
            z_far,
            view,
            proj,
        }

    }

    pub fn update(&mut self, input_state: &mut InputState, dt: Duration) {
        let speed = self.movement_speed * dt.as_secs_f32();

        if input_state.is_key_pressed(VirtualKeyCode::W) {
            self.pos += self.front * speed;
        }

        if input_state.is_key_pressed(VirtualKeyCode::S) {
            self.pos -= self.front * speed;
        }

        if input_state.is_key_pressed(VirtualKeyCode::A) {
            self.pos -= self.front.cross(self.up).normalize() * speed;
        }

        if input_state.is_key_pressed(VirtualKeyCode::D) {
            self.pos += self.front.cross(self.up).normalize() * speed;
        }

        let (x_delta, y_delta) = input_state.mouse_delta();
       
        self.yaw += x_delta as f32 * self.rotation_speed;
        self.pitch -= y_delta as f32 * self.rotation_speed;
        self.pitch = self.pitch.clamp(-89.0, 89.0);

        self.front = -Vec3::new(
            f32::cos(self.yaw.to_radians()) * f32::cos(self.pitch.to_radians()),
            f32::sin(self.pitch.to_radians()),
            f32::sin(self.yaw.to_radians()) * f32::cos(self.pitch.to_radians()),
        )
        .normalize();

        self.view = Mat4::look_at_rh(self.pos, self.pos + self.front, self.up);
    }

    pub fn update_proj(&mut self, aspect_ratio: f32) {
        self.proj = Mat4::perspective_rh(
            self.fov.to_radians(),
            aspect_ratio,
            self.z_near,
            self.z_far,
        );
    }
}

/// Data related to the camera view. This is updated every frame and has a copy per frame in
/// flight.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::NoUninit)]
pub struct ViewUniform {
    /// The position of the camera in world space.
    eye: Vec4,
    /// The view matrix.
    view: Mat4,
    /// `proj * view`. This is cached to save a dot product between two 4x4 matrices
    /// for each vertex.
    proj_view: Mat4,
}

impl ViewUniform {
    pub fn new(camera: &Camera) -> Self {
        let eye = Vec4::from((camera.pos, 0.0));
        let proj_view = camera.proj * camera.view;
        Self { eye, view: camera.view.clone(), proj_view }
    }
}

/// Data related to the projection matrix. This is only updated on screen resize or camera settings
/// changes. There is only one copy.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::NoUninit)]
pub struct ProjUniform {
    /// The projection matrix.
    proj: Mat4,
    /// The inverse projection.
    ///
    /// Used to transform points from screen to view space.
    inverse_proj: Mat4,
    /// The screen dimensions.
    dimensions: Vec2,
    /// z near and z far.
    z_plane: Vec2,
}

impl ProjUniform {
    pub fn new(renderer: &Renderer, camera: &Camera) -> Self {
        let dimensions = Vec2::new(
            renderer.swapchain.extent.width as f32,
            renderer.swapchain.extent.height as f32,
        );
        let inverse_proj = camera.proj.inverse();
        let z_plane = Vec2::new(camera.z_near, camera.z_far);
        Self { proj: camera.proj.clone(), inverse_proj, dimensions, z_plane }
    }
}

/// Uniform buffers containing camera information.
pub struct CameraUniforms {
    pub view_buffers: [Res<Buffer>; FRAMES_IN_FLIGHT],
    pub proj_buffer: Res<Buffer>,

    pub descriptor: Res<DescriptorSet>,
}

impl CameraUniforms {
    pub fn new(renderer: &Renderer, pool: &ResourcePool, camera: &Camera) -> Result<Self> {
        let memory_flags = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;

        let view_buffers: [_; FRAMES_IN_FLIGHT] = array::try_from_fn(|_| {
            Buffer::new(renderer, pool, memory_flags, &BufferReq {
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                size: mem::size_of::<ViewUniform>() as u64,
            })
        })?;

        let proj_buffer = Buffer::new(renderer, pool, memory_flags, &BufferReq {
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            size: mem::size_of::<ProjUniform>() as u64,
        })?;

        let layout = pool.alloc(DescriptorSetLayout::new(&renderer, &[
            LayoutBinding {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                stage: vk::ShaderStageFlags::COMPUTE
                    | vk::ShaderStageFlags::FRAGMENT
                    | vk::ShaderStageFlags::VERTEX,
                array_count: None,
            },
            LayoutBinding {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                stage: vk::ShaderStageFlags::COMPUTE
                    | vk::ShaderStageFlags::FRAGMENT
                    | vk::ShaderStageFlags::VERTEX,
                array_count: None,
            },
        ])?);

        let descriptor = pool.alloc(DescriptorSet::new_per_frame(&renderer, layout, &[
            DescriptorBinding::Buffer([proj_buffer.clone(), proj_buffer.clone()]),
            DescriptorBinding::Buffer(view_buffers.clone()),
        ])?);

        let uniforms = Self { view_buffers, proj_buffer, descriptor };

        uniforms.update_proj(renderer, camera)?;

        Ok(uniforms)
    }

    /// Update view uniform for frame with index `frame_index`.
    pub fn update_view(&self, frame_index: usize, camera: &Camera) -> Result<()> {
        self.view_buffers[frame_index]
            .get_mapped()?
            .fill(bytemuck::bytes_of(&ViewUniform::new(camera)));

        Ok(())
    }

    pub fn update_proj(&self, renderer: &Renderer, camera: &Camera) -> Result<()> {
        self.proj_buffer
            .get_mapped()?
            .fill(bytemuck::bytes_of(&ProjUniform::new(renderer, camera)));

        Ok(())
    }
}
