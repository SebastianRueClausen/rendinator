use anyhow::Result;
use ash::vk;
use glam::{Vec2, Vec3, Vec4, Mat4};
use serde::{Serialize, Deserialize};

use std::path::Path;
use std::fs;

/// The vertex format used by the [`Mesh`].
#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit, Serialize, Deserialize)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub texcoord: Vec2,
    pub tangent: Vec4,
}


#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum IndexFormat {
    U16 = vk::IndexType::UINT16.as_raw() as isize,
    U32 = vk::IndexType::UINT32.as_raw() as isize,
}

impl IndexFormat {
    pub fn byte_size(&self) -> usize {
        match self {
            IndexFormat::U16 => 2,
            IndexFormat::U32 => 4,
        }
    }
}

impl Into<vk::IndexType> for IndexFormat {
    fn into(self) -> vk::IndexType {
        vk::IndexType::from_raw(self as i32) 
    }
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
    ///
    /// The format is indicated by `index_format`.
    pub indices: Vec<u8>,

    /// The format of `indices`.
    pub index_format: IndexFormat,
}

impl Scene {
    pub fn load(path: &Path) -> Result<Self> {
       Ok(bincode::deserialize(&fs::read(path)?)?)
    }

    pub fn store(&self, path: &Path) -> Result<()> {
        fs::write(path, bincode::serialize(self)?).map_err(|err| err.into())
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum BcFormat {
    Bc5Unorm = vk::Format::BC5_UNORM_BLOCK.as_raw() as isize,
    Bc7Unorm = vk::Format::BC7_UNORM_BLOCK.as_raw() as isize,
    Bc7Srgb = vk::Format::BC7_SRGB_BLOCK.as_raw() as isize,
}

impl BcFormat {
    pub fn block_size(self) -> usize {
        match self {
            BcFormat::Bc5Unorm | BcFormat::Bc7Unorm | BcFormat::Bc7Srgb => 16,
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
#[derive(Debug, Serialize, Deserialize)]
pub struct Image {
    pub data: Vec<u8>,

    pub format: ImageFormat,

    /// The pixel width of the image.
    pub width: u32,

    /// The pixel height of the image.
    pub height: u32,
}

#[derive(Serialize, Deserialize)]
pub struct Skybox {
    pub images: [Image; 6],
}

impl Skybox {
    pub fn width(&self) -> u32 {
        self.images[0].width
    }

    pub fn height(&self) -> u32 {
        self.images[0].height
    }

    pub fn load(path: &Path) -> Result<Self> {
       Ok(bincode::deserialize(&fs::read(path)?)?)
    }

    pub fn store(&self, path: &Path) -> Result<()> {
        fs::write(path, bincode::serialize(self)?).map_err(|err| err.into())
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct BoundingSphere {
    pub center: Vec3,
    pub radius: f32,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Primitive {
    /// The material of the mesh.
    pub material: usize,

    /// The first index from `Scene::indices` used.
    pub index_start: u32,

    /// The number of indices in the mesh.
    pub index_count: u32,

    /// The vertex start / vertex offset.
    ///
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

/// Metadata of a single character glyph.
#[derive(Clone, Serialize, Deserialize)]
pub struct Glyph {
    pub codepoint: char,

    /// The dimensions of the glyph scaled to the font size.
    pub scaled_dim: Vec2,

    /// The offset from the draw position the glyph should be draw at, scaled to the font size.
    pub scaled_offset: Vec2,

    /// The minimum texcoord in the glyph atlas.
    pub texcoord_min: Vec2,

    /// The maximum texcoord in the glyph atlas.
    pub texcoord_max: Vec2,

    /// How much the draw position should be advanced on the x-axis.
    pub advance: f32,
}

#[derive(Serialize, Deserialize)]
pub struct Font {
    /// The size of the font.
    pub size: u32,

    /// The SDF atlas.
    ///
    /// This is an unormalized r8 image.
    pub atlas: Image,

    /// All the glyphs of the font.
    pub glyphs: Vec<Glyph>,
}

impl Font {
    pub fn load(path: &Path) -> Result<Self> {
       Ok(bincode::deserialize(&fs::read(path)?)?)
    }

    pub fn store(&self, path: &Path) -> Result<()> {
        fs::write(path, bincode::serialize(self)?).map_err(|err| err.into())
    }
}

