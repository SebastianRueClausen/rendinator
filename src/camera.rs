use glam::{Mat4, Vec2, Vec3, Vec4};

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
}

#[derive(Clone, Copy)]
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

#[derive(Clone, Copy)]
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

    z_near: f32,
    z_far: f32,
}

impl ProjUniform {
    pub fn new(proj: &Proj) -> Self {
        Self {
            proj: proj.mat,
            inverse_proj: proj.inverse_mat,
            surface_size: proj.surface_size,
            z_near: proj.z_near,
            z_far: proj.z_far,
        }
    }
}
