pub mod normal;

use eyre::{Context, Result};
use std::{fs, path::Path};

use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
pub use meshopt::utilities;
use serde::{Deserialize, Serialize};

use normal::TangentFrame;

#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DirectionalLight {
    pub direction: Vec4,
    pub irradiance: Vec4,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec4::new(0.0, 1.0, 0.0, 1.0),
            irradiance: Vec4::ONE,
        }
    }
}

unsafe impl bytemuck::NoUninit for DirectionalLight {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Transform {
    pub scale: Vec4,
    pub rotation: Quat,
    pub translation: Vec4,
}

impl From<Mat4> for Transform {
    fn from(matrix: Mat4) -> Self {
        let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
        Self {
            scale: scale.extend(1.0),
            translation: translation.extend(1.0),
            rotation,
        }
    }
}

impl From<Transform> for Mat4 {
    fn from(transform: Transform) -> Self {
        Self::from_scale_rotation_translation(
            transform.scale.truncate(),
            transform.rotation,
            transform.translation.truncate(),
        )
    }
}

unsafe impl bytemuck::NoUninit for Transform {}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BoundingSphere {
    pub center: Vec3,
    pub radius: f32,
}

impl BoundingSphere {
    pub fn transformed(self, transform: Transform) -> Self {
        let scale = transform.scale.abs().max_element();
        let transform: Mat4 = transform.into();
        let center = (transform * self.center.extend(1.0)).truncate();

        Self {
            radius: scale * self.radius,
            center,
        }
    }
}

unsafe impl bytemuck::NoUninit for BoundingSphere {}

#[repr(C)]
#[repr(align(16))]
#[derive(Clone, Default, Copy, Debug, Serialize, Deserialize)]
pub struct Meshlet {
    pub bounding_sphere: BoundingSphere,
    pub cone_axis: [i8; 3],
    pub cone_cutoff: i8,
    pub data_offset: u32,
    pub vertex_count: u8,
    pub triangle_count: u8,
}

unsafe impl bytemuck::NoUninit for Meshlet {}

#[repr(C)]
#[derive(Clone, Default, Copy, Debug, Serialize, Deserialize)]
pub struct Lod {
    pub index_offset: u32,
    pub index_count: u32,
    pub meshlet_offset: u32,
    pub meshlet_count: u32,
}

unsafe impl bytemuck::NoUninit for Lod {}

#[repr(C)]
#[repr(align(16))]
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Mesh {
    pub bounding_sphere: BoundingSphere,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub material: u32,
    pub lod_count: u32,
    pub lods: [Lod; 8],
}

unsafe impl bytemuck::NoUninit for Mesh {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub mesh_indices: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instance {
    pub transform: Transform,
    pub mesh_index: Option<u32>,
    pub children: Vec<Instance>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum TextureKind {
    Albedo,
    Normal,
    Specular,
    Emissive,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Texture {
    pub kind: TextureKind,
    pub width: u32,
    pub height: u32,
    pub mips: Vec<Box<[u8]>>,
}

#[repr(C)]
#[repr(align(16))]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Material {
    pub albedo_texture: u32,
    pub normal_texture: u32,
    pub specular_texture: u32,
    pub emissive_texture: u32,
    pub base_color: Vec4,
    pub emissive: Vec4,
    pub metallic: f32,
    pub roughness: f32,
    pub ior: f32,
    pub padding: [u32; 1],
}

unsafe impl bytemuck::NoUninit for Material {}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct Position {
    encoded: [i16; 3],
}

impl Position {
    fn new(position: Vec3, bounding_sphere: &BoundingSphere) -> Self {
        let position = (position - bounding_sphere.center) / bounding_sphere.radius;
        let encoded = position
            .to_array()
            .map(|value| utilities::quantize_snorm(value, 16) as i16);
        Self { encoded }
    }
}

unsafe impl bytemuck::NoUninit for Position {}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct Vertex {
    pub texcoord: [u16; 2],
    pub position: Position,
    pub material: u16,
    pub tangent_frame: TangentFrame,
}

impl Vertex {
    pub fn encode(
        bounding_sphere: &BoundingSphere,
        position: Vec3,
        normal: Vec3,
        texcoord: Vec2,
        tangent: Vec4,
        material: u32,
    ) -> Self {
        let tangent_frame = TangentFrame::new(normal, tangent);

        debug_assert!({
            let (decoded_normal, _) = tangent_frame.into_normal_tangent();
            decoded_normal.abs_diff_eq(normal, 0.1)
        });

        let texcoord = texcoord.to_array().map(utilities::quantize_half);
        let position = Position::new(position, bounding_sphere);

        Self {
            material: material as u16,
            tangent_frame,
            position,
            texcoord,
        }
    }
}

unsafe impl bytemuck::NoUninit for Vertex {}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Scene {
    pub directional_light: DirectionalLight,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub textures: Vec<Texture>,
    pub materials: Vec<Material>,
    pub meshes: Vec<Mesh>,
    pub models: Vec<Model>,
    pub instances: Vec<Instance>,
    pub meshlets: Vec<Meshlet>,
    pub meshlet_data: Vec<u32>,
}

impl Scene {
    pub fn add_texture(&mut self, texture: Texture) -> u32 {
        let index = self.textures.len();
        self.textures.push(texture);

        index as u32
    }

    pub fn add_material(&mut self, material: Material) -> u32 {
        let index = self.materials.len();
        self.materials.push(material);

        index as u32
    }

    pub fn add_mesh(&mut self, mesh: Mesh) -> u32 {
        let index = self.meshes.len();
        self.meshes.push(mesh);

        index as u32
    }

    pub fn serialize(&self, path: &Path) -> Result<()> {
        let bytes = bincode::serialize(&self).wrap_err("failed to serialize scene")?;
        fs::write(path, &bytes).wrap_err_with(|| format!("failed to write to {path:?}"))
    }
}
