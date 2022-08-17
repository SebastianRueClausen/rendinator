use anyhow::Result;
use glam::{Vec2, Vec3, Vec4, Mat4};

use std::path::Path;
use std::fs;

/// The vertex format used by the [`Mesh`].
#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit, serde::Serialize, serde::Deserialize)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub texcoord: Vec2,
    pub tangent: Vec4,
}


#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum IndexFormat {
    U16,
    U32,
}

impl IndexFormat {
    pub fn byte_size(&self) -> usize {
        match self {
            IndexFormat::U16 => 2,
            IndexFormat::U32 => 4,
        }
    }
}

/// All mesh and texture data for at scene.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Scene {
    pub meshes: Vec<Mesh>,

    /// All materials used in the scene.
    pub materials: Vec<Material>,

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

/// A raw image without specified format.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Image {
    pub data: Vec<u8>,

    /// The pixel width of the image.
    pub width: u32,

    /// The pixel height of the image.
    pub height: u32,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Mesh {
    /// The index of the material in [`Scene`].
    pub material: usize,

    /// The transform matrix of the mesh.
    pub transform: Mat4,

    /// The first index from `Mesh::indices` used.
    pub index_start: u32,

    /// The number of indices in the mesh.
    pub index_count: u32,

    /// The vertex start / vertex offset.
    ///
    /// This is the first vertex index 0 points to.
    pub vertex_start: u32,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Material {
    /// Base color / albedo texture image.
    ///
    /// This is an rgba8 image.
    pub base_color: Image,

    /// The normal map of the material.
    ///
    /// This is an unormalized rgba8 image.
    pub normal: Image,

    /// The metallic roughness map.
    ///
    /// This is an unormalized rg8 image.
    pub metallic_roughness: Image,
}

/// Metadata of a single character glyph.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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

#[derive(serde::Serialize, serde::Deserialize)]
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

