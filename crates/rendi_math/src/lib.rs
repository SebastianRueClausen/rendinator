pub mod polynom;
pub mod rect;
pub mod bezier;

pub mod prelude {
    pub use glam::{Vec2, Vec3, Vec4, UVec2, UVec3, UVec4, Mat3, Mat4, Quat};
    pub use glam::swizzles::{Vec2Swizzles, Vec3Swizzles, Vec4Swizzles};
    pub use crate::rect::Rect;
    pub use crate::bezier::Bezier;
}
