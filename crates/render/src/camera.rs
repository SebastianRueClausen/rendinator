use glam::{Mat4, Vec2, Vec3};

#[derive(Debug)]
pub(crate) struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub view: Mat4,
    pub proj: Mat4,
}

impl Camera {
    pub const UP: Vec3 = Vec3::Y;

    pub fn new(surface_size: Vec2) -> Self {
        let position = Vec3::ZERO;
        let forward = Vec3::X;
        let yaw = 0.0;
        let pitch = 0.0;
        let z_near = 0.1;
        let fov = std::f32::consts::FRAC_PI_2;
        let view = Mat4::look_at_rh(position, position + forward, Self::UP);
        let proj = proj(fov, surface_size.x / surface_size.y, z_near);
        Self { position, forward, yaw, pitch, view, proj }
    }

    pub fn move_by(&mut self, delta: CameraMove) {
        let horizontal = self.forward.cross(Self::UP).normalize();
        self.position += self.forward * delta.forward;
        self.position -= self.forward * delta.backward;
        self.position += horizontal * delta.right;
        self.position -= horizontal * delta.left;
        self.yaw = (self.yaw + delta.yaw) % std::f32::consts::TAU;
        self.pitch = (self.pitch - delta.pitch).clamp(-1.553, 1.553);
        self.forward = Vec3::new(
            f32::cos(self.yaw) * f32::cos(self.pitch),
            f32::sin(self.pitch),
            f32::sin(self.yaw) * f32::cos(self.pitch),
        )
        .normalize();
        self.view = Mat4::look_at_rh(
            self.position,
            self.position + self.forward,
            Self::UP,
        );
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CameraMove {
    pub left: f32,
    pub right: f32,
    pub forward: f32,
    pub backward: f32,
    pub yaw: f32,
    pub pitch: f32,
}

fn proj(fov: f32, aspect: f32, z_near: f32) -> Mat4 {
    let mut proj = Mat4::perspective_infinite_reverse_rh(fov, aspect, z_near);
    proj.z_axis[3] = 1.0;
    proj
}
