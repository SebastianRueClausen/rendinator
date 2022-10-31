use anyhow::Result;
use ash::vk;
use serde::{Serialize, Deserialize};

use rendi_math::prelude::*;

use std::path::Path;
use std::fs;

/// The vertex format used by the [`Mesh`].
#[repr(C)]
#[derive(Default, Clone, Copy, bytemuck::NoUninit, Serialize, Deserialize)]
pub struct Vertex {
    /// x, y, z contains the vertex position as half precision floats.
    ///
    /// w is the tangent angle compared to a fixed orthogonal vector to the normal.
    /// The w sign bit indicates the sign of the bitangent.
    pub position: [u16; 4],

    /// The textexture coordinates as half precision floats.
    pub texcoord: [u16; 2],

    /// Octahedron encoded normal.
    pub normal: [u16; 2],

    pub tangent: [u16; 4],
}

/// All mesh and texture data for at scene.
#[derive(Serialize, Deserialize)]
pub struct Scene {
    pub meshes: Vec<Mesh>,

    pub instances: Vec<Instance>,

    /// All materials used in the scene.
    pub materials: Vec<Material>,

    pub textures: Vec<Image>,

    /// All the vertices of the while scene.
    pub vertices: Vec<Vertex>,

    /// The index data for all meshes.
    pub indices: Vec<u32>,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum BcFormat {
    Bc5Unorm = vk::Format::BC5_UNORM_BLOCK.as_raw() as isize,
    Bc1Unorm = vk::Format::BC1_RGBA_UNORM_BLOCK.as_raw() as isize,
    Bc1Srgb = vk::Format::BC1_RGBA_SRGB_BLOCK.as_raw() as isize,
}

impl BcFormat {
    pub fn block_size(self) -> usize {
        match self {
            BcFormat::Bc5Unorm => 16,
            BcFormat::Bc1Unorm | BcFormat::Bc1Srgb => 8,
        }
    }
}

impl Into<vk::Format> for BcFormat {
    fn into(self) -> vk::Format {
        vk::Format::from_raw(self as i32) 
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum RawFormat {
    Rgba8Unorm = vk::Format::R8G8B8A8_UNORM.as_raw() as isize,
    Rgba8Srgb = vk::Format::R8G8B8A8_SRGB.as_raw() as isize,
    Rg8Unorm = vk::Format::R8G8_UNORM.as_raw() as isize,
    R8Unorm = vk::Format::R8_UNORM.as_raw() as isize,
}

impl Into<vk::Format> for RawFormat {
    fn into(self) -> vk::Format {
        vk::Format::from_raw(self as i32) 
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum ImageFormat {
    Raw(RawFormat),
    Bc(BcFormat),
}

impl Into<vk::Format> for ImageFormat {
    fn into(self) -> vk::Format {
        match self {
            ImageFormat::Raw(format) => format.into(),
            ImageFormat::Bc(format) => format.into(),
        }
    }
}

/// A raw image without specified format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    pub mips: Vec<Vec<u8>>,

    pub format: ImageFormat,

    pub width: u32,
    pub height: u32,
}

impl Image {
    pub fn mip_levels(&self) -> u32 {
        self.mips.len() as u32
    }

    pub fn width(&self, mip_level: u32) -> u32 {
        self.width >> mip_level
    }

    pub fn height(&self, mip_level: u32) -> u32 {
        self.height >> mip_level
    }

    pub fn base_image_data(&self) -> &[u8] {
        self.mips[0].as_slice()
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct BoundingSphere {
    pub center: Vec3,
    pub radius: f32,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Lod {
    pub index_start: u32,
    pub index_count: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Primitive {
    /// The material of the mesh.
    pub material: usize,

    pub lods: Vec<Lod>,

    /// This is the first vertex index 0 points to.
    pub vertex_start: u32,

    pub bounding_sphere: BoundingSphere,
}

#[derive(Serialize, Deserialize)]
pub struct Mesh {
    pub primitives: Vec<Primitive> 
}

#[derive(Serialize, Deserialize)]
pub struct Instance {
    /// The index of the model used.
    pub mesh: usize,

    /// The transform matrix of the instance.
    pub transform: Mat4,
}

#[derive(Serialize, Deserialize)]
pub struct Material {
    pub albedo_map: usize,
    pub specular_map: usize,
    pub normal_map: usize,
}

pub fn load<T: for<'a> Deserialize<'a>>(path: &Path) -> Result<T> {
    bincode::deserialize(&fs::read(path)?).map_err(|err| {
        anyhow::anyhow!("failed to store asset: {err}")
    })
}

pub fn store<T: Serialize>(asset: &T, path: &Path) -> Result<()> {
    fs::write(path, bincode::serialize(asset)?).map_err(|err| {
        anyhow::anyhow!("failed to load asset: {err}")
    })
}

