use ash::vk;
use glam::{Mat4, Vec2, Vec3, Vec4};
use anyhow::Result;

use std::mem;

use crate::core::*;
use crate::resource::*;

#[derive(Clone, Copy)]
pub struct View {
    pub pos: Vec3,
    pub front: Vec3,
    pub up: Vec3,
    pub yaw: f32,
    pub pitch: f32,

    pub mat: Mat4,
}

impl View {
    pub fn new() -> Self {
        let pos = Vec3::new(10.0, 10.0, 10.0);
        let up = Vec3::new(0.0, -1.0, 0.0);
        let front = Vec3::default();

        let yaw = 0.0;
        let pitch = 0.0;

        let mat = Mat4::look_at_rh(pos, pos + front, up);

        Self { yaw, pitch, front, pos, mat, up }
    }

    pub fn update(&mut self) {
        self.mat = Mat4::look_at_rh(self.pos, self.pos + self.front, self.up);
    }
}

pub enum ProjMode {
    Perspective {
        fov: f32,
    },
    Orthographic {
        left: f32,
        right: f32,
        top: f32,
        bottom: f32,
    }
}

pub struct Proj {
    pub z_near: f32,
    pub z_far: f32,
    pub surface_size: Vec2,

    pub mat: Mat4,
    pub inverse_mat: Mat4,

    pub mode: ProjMode,
}

impl Proj {
    pub fn new(surface_size: Vec2, mode: ProjMode) -> Self {
        let z_near = 0.1;
        let z_far = 100.0;

        let aspect_ratio = surface_size.x / surface_size.y;
        
        let mat = match mode {
            ProjMode::Perspective { fov } => {
                Mat4::perspective_rh(fov.to_radians(), aspect_ratio, z_near, z_far)
            }
            ProjMode::Orthographic { left, right, top, bottom } => {
                Mat4::orthographic_rh(left, right, bottom, top, z_near, z_far)
            }
        };

        let inverse_mat = mat.inverse();

        Self { surface_size, z_near, z_far, mat, inverse_mat, mode }
    }

    pub fn update(&mut self, surface_size: Vec2) {
        let aspect_ratio = surface_size.x / surface_size.y;

        self.surface_size = surface_size;
        self.mat = match self.mode {
            ProjMode::Perspective { fov } => {
                Mat4::perspective_rh(fov.to_radians(), aspect_ratio, self.z_near, self.z_far)
            }
            ProjMode::Orthographic { left, right, top, bottom } => {
                Mat4::orthographic_rh(left, right, bottom, top, self.z_near, self.z_far)
            }
        };
    }
}

pub fn frustrum_planes(proj: &Proj, view: &View) -> [Vec4; 6] {
    let proj_view = proj.mat * view.mat;

    let planes = [
        proj_view.row(3) + proj_view.row(0),
        proj_view.row(3) - proj_view.row(0),
        proj_view.row(3) + proj_view.row(1),
        proj_view.row(3) - proj_view.row(1),
        proj_view.row(3) - proj_view.row(2),

        Vec4::splat(0.0),
    ];

    planes.map(|plane| plane / plane.truncate().length())
}

/*
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
*/

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
    pub fn new(proj: &Proj, view: &View) -> Self {
        let eye = view.pos.extend(0.0);
        let proj_view = proj.mat * view.mat;

        Self { eye, view: view.mat, proj_view }
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
    surface_size: Vec2,
    /// z near and z far.
    z_plane: Vec2,
}

impl ProjUniform {
    pub fn new(proj: &Proj) -> Self {
        let z_plane = Vec2::new(proj.z_near, proj.z_far);

        Self {
            proj: proj.mat,
            inverse_proj: proj.inverse_mat,
            surface_size: proj.surface_size,
            z_plane,
        }
    }
}

/// Uniform buffers containing camera information.
pub struct CameraUniforms {
    pub view_buffers: PerFrame<Res<Buffer>>,
    pub proj_buffer: Res<Buffer>,
    pub descs: PerFrame<Res<DescSet>>,
}

impl CameraUniforms {
    pub fn new(renderer: &Renderer, proj: &Proj) -> Result<Self> {
        let pool = &renderer.static_pool;
        let memory_flags = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;

        let view_buffers = PerFrame::try_from_fn(|_| {
            pool.create_buffer(memory_flags, &BufferInfo {
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                size: mem::size_of::<ViewUniform>() as u64,
            })
        })?;

        let proj_buffer = pool.create_buffer(memory_flags, &BufferInfo {
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            size: mem::size_of::<ProjUniform>() as u64,
        })?;

        let layout = pool.create_desc_layout(&[
            DescLayoutSlot {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                stage: vk::ShaderStageFlags::COMPUTE
                    | vk::ShaderStageFlags::FRAGMENT
                    | vk::ShaderStageFlags::VERTEX,
                array_count: None,
            },
            DescLayoutSlot {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                stage: vk::ShaderStageFlags::COMPUTE
                    | vk::ShaderStageFlags::FRAGMENT
                    | vk::ShaderStageFlags::VERTEX,
                array_count: None,
            },
        ])?;
       
        let descs = PerFrame::try_from_fn(|frame_index| {
            pool.create_desc_set(layout.clone(), &[
                DescBinding::Buffer(proj_buffer.clone()),
                DescBinding::Buffer(view_buffers[frame_index].clone()),
            ])
        })?;

        let uniforms = Self { view_buffers, proj_buffer, descs };

        uniforms.update_proj(proj)?;

        Ok(uniforms)
    }

    /// Update view uniform for frame with index `frame_index`.
    pub fn update_view(&self, frame_index: FrameIndex, proj: &Proj, view: &View) -> Result<()> {
        self.view_buffers[frame_index]
            .get_mapped()?
            .fill(bytemuck::bytes_of(&ViewUniform::new(proj, view)));

        Ok(())
    }

    pub fn update_proj(&self, proj: &Proj) -> Result<()> {
        self.proj_buffer
            .get_mapped()?
            .fill(bytemuck::bytes_of(&ProjUniform::new(proj)));

        Ok(())
    }
}
