use rendi_math::prelude::*;

#[derive(Clone)]
pub struct Camera {
    /// The position of the camera in world space.
    pub pos: Vec3,

    /// The front vector.
    pub front: Vec3,

    /// The yaw of the camera (horizontal angle in degrees).
    pub yaw: f32,

    /// The pitch of the camera (vertical angle in degrees).
    pub pitch: f32,

    /// The fov of the view frustrum in degrees.
    pub fov: f32,

    /// The distance to the near z-plane.
    pub z_near: f32,

    /// The distance to the far z-plane.
    pub z_far: f32,

    /// The size of the surface in pixels.
    pub surface_size: Vec2,

    /// The view matrix.
    pub view: Mat4,

    /// The projection matrix.
    pub proj: Mat4,

    /// Inverse of the projection matrix.
    pub inverse_proj: Mat4,
}

impl Camera {
    /// The up vector.
    pub const UP: Vec3 = Vec3::new(0.0, -1.0, 0.0);

    pub fn new(surface_size: Vec2) -> Self {
        let pos = Vec3::new(10.0, 10.0, 10.0);
        let front = Vec3::X;

        let yaw = 0.0;
        let pitch = 0.0;

        let view = Mat4::look_at_rh(pos, pos + front, Self::UP);

        let fov = 66.0_f32;
        let z_near = 0.1;
        let z_far = 100.0;

        let aspect_ratio = surface_size.x / surface_size.y;
        
        let proj = Mat4::perspective_rh(fov.to_radians(), aspect_ratio, z_near, z_far);
        let inverse_proj = proj.inverse();

        Self { pos, yaw, pitch, view, z_near, z_far, fov, surface_size, inverse_proj, proj, front }
    }

    /// Update camera for new surface size.
    pub fn handle_resize(&mut self, surface_size: Vec2) {
        let aspect_ratio = surface_size.x / surface_size.y;
        self.surface_size = surface_size;

        self.proj =
            Mat4::perspective_rh(self.fov.to_radians(), aspect_ratio, self.z_near, self.z_far);

        self.inverse_proj = self.proj.inverse(); 
    }

    /// Get bounding planes of the frustrum in view space.
    pub fn frustrum_planes(&self) -> [Vec4; 6] {
        let proj_view = self.proj * self.view;

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

    /// Update the view matrix.
    pub fn update_view_matrix(&mut self) {
        self.view = Mat4::look_at_rh(self.pos, self.pos + self.front, Self::UP);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::NoUninit)]
pub struct ViewUniform {
    /// The position of the camera in world space.
    eye: Vec4,

    /// The view matrix.
    view: Mat4,

    /// `proj * view`. This is cached to save a dot product between two 4x4 matrices
    /// for each vertex shader invocation.
    proj_view: Mat4,
}

impl ViewUniform {
    pub fn new(camera: &Camera) -> Self {
        let eye = camera.pos.extend(0.0);
        let proj_view = camera.proj * camera.view;

        Self { eye, view: camera.view, proj_view }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::NoUninit)]
pub struct ProjUniform {
    /// The projection matrix.
    proj: Mat4,

    /// The inverse of `proj`.
    inverse_proj: Mat4,

    /// The screen dimensions.
    surface_size: Vec2,

    /// The camera near z-plane.
    z_near: f32,

    /// THe camera far z-plane.
    z_far: f32,
}

impl ProjUniform {
    pub fn new(camera: &Camera) -> Self {
        Self {
            proj: camera.proj,
            inverse_proj: camera.inverse_proj,
            surface_size: camera.surface_size,
            z_near: camera.z_near,
            z_far: camera.z_far,
        }
    }
}
