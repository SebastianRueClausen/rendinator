use glam::{Mat4, Vec3, Vec4};

pub struct Camera {
    pub pos: Vec3,
    pub front: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub proj: Mat4,
}

impl Camera {
    pub const UP: Vec3 = Vec3::Y;

    pub fn new(aspect_ratio: f32) -> Self {
        let z_near = 0.1;
        let z_far = 50.0;
        let fov = std::f32::consts::PI / 4.0;

        let proj = calc_proj(fov, aspect_ratio, z_near, z_far);

        Self {
            pos: Vec3::ZERO,
            front: Vec3::X,
            yaw: 0.0,
            pitch: 0.0,
            z_near,
            z_far,
            fov,
            proj,
        }
    }

    pub fn move_by_delta(&mut self, delta: CameraDelta) {
        let horizontal = self.front.cross(Self::UP).normalize();

        self.pos += self.front * delta.forward;
        self.pos -= self.front * delta.backward;
        self.pos += horizontal * delta.right;
        self.pos -= horizontal * delta.left;

        self.yaw = (self.yaw - delta.yaw) % 360.0;
        self.pitch = (self.pitch + delta.pitch).clamp(-89.0, 89.0);

        self.front = Vec3::new(
            f32::cos(self.yaw.to_radians()) * f32::cos(self.pitch.to_radians()),
            f32::sin(self.pitch.to_radians()),
            f32::sin(self.yaw.to_radians()) * f32::cos(self.pitch.to_radians()),
        )
        .normalize();
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.pos, self.pos + self.front, Self::UP)
    }

    pub fn proj(&self) -> Mat4 {
        self.proj
    }

    pub fn proj_view(&self) -> Mat4 {
        self.proj() * self.view()
    }

    pub fn resize_proj(&mut self, aspect_ratio: f32) {
        self.proj = calc_proj(self.fov, aspect_ratio, self.z_near, self.z_far);
    }

    pub fn frustrum(&self) -> Frustrum {
        let proj_view = self.proj_view();

        let planes = [
            proj_view.row(3) + proj_view.row(0),
            proj_view.row(3) - proj_view.row(0),
            proj_view.row(3) + proj_view.row(1),
            proj_view.row(3) - proj_view.row(1),
            proj_view.row(3) + proj_view.row(2),
            proj_view.row(3) - proj_view.row(2),
        ];

        let planes = planes.map(|plane| {
            let normal = plane.truncate();
            let length = normal.length();
            -plane / length
        });

        Frustrum {
            left: planes[0],
            right: planes[1],
            bottom: planes[2],
            top: planes[3],
            near: planes[4],
            far: planes[5],
        }
    }
}

fn calc_proj(fov: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Mat4 {
    Mat4::perspective_rh(fov, aspect_ratio, z_near, z_far)
}

#[derive(Clone, Debug, Default)]
pub struct CameraDelta {
    pub left: f32,
    pub right: f32,
    pub forward: f32,
    pub backward: f32,
    pub yaw: f32,
    pub pitch: f32,
}

pub struct Frustrum {
    pub top: Vec4,
    pub bottom: Vec4,
    pub right: Vec4,
    pub left: Vec4,
    pub far: Vec4,
    pub near: Vec4,
}
