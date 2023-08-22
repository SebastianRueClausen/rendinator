mod gltf;
mod normal;
mod quantize;

use std::{
    fs,
    ops::Range,
    path::{Path, PathBuf},
};

use bytemuck::{Pod, Zeroable};
use eyre::{Result, WrapErr};
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use half::f16;
use serde::{Deserialize, Serialize};

use normal::TangentFrame;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Serialize, Deserialize)]
pub struct DirectionalLight {
    direction: Vec4,
    irradiance: Vec4,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec4::new(0.0, 1.0, 0.0, 1.0),
            irradiance: Vec4::splat(1.0),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Transform {
    pub scale: Vec3,
    pub rotation: Quat,
    pub translation: Vec3,
}

impl From<Mat4> for Transform {
    fn from(matrix: Mat4) -> Self {
        let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
        Self {
            scale,
            rotation,
            translation,
        }
    }
}

impl From<Transform> for Mat4 {
    fn from(transform: Transform) -> Self {
        Self::from_scale_rotation_translation(
            transform.scale,
            transform.rotation,
            transform.translation,
        )
    }
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug, Zeroable, Pod, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Primitive {
    pub indices: Range<u32>,
    pub bounding_sphere: BoundingSphere,
    pub material: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instance {
    pub name: Option<String>,
    pub mesh: Option<u32>,
    pub transform: Transform,
    pub children: Vec<Instance>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Texture {
    pub format: wgpu::TextureFormat,
    pub extent: wgpu::Extent3d,
    pub mip_level_count: u32,
    pub mips: Box<[u8]>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Serialize, Deserialize)]
pub struct Material {
    albedo_texture: u32,
    normal_texture: u32,
    specular_texture: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Mesh {
    pub primitives: Vec<Primitive>,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, Serialize, Deserialize)]
pub struct Position {
    encoded: [i16; 3],
}

impl Position {
    fn new(position: Vec3, bounding_sphere: &BoundingSphere) -> Self {
        let position = (position - bounding_sphere.center) / bounding_sphere.radius;

        Self {
            encoded: position
                .to_array()
                .map(|value| quantize::quantize_snorm::<16>(value) as i16),
        }
    }

    #[cfg(test)]
    fn to_vec3(self, bounding_sphere: &BoundingSphere) -> Vec3 {
        let vec = Vec3::from_array(
            self.encoded
                .map(|value| quantize::dequantize_snorm::<16>(value as i32)),
        );

        (vec * bounding_sphere.radius) + bounding_sphere.center
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable, Serialize, Deserialize)]
pub struct Vertex {
    pub texcoord: [f16; 2],
    pub position: Position,
    pub material: u16,
    pub tangent_frame: TangentFrame,
}

impl Vertex {
    fn new(
        bounding_sphere: &BoundingSphere,
        position: Vec3,
        normal: Vec3,
        texcoord: Vec2,
        tangent: Vec4,
        material: u32,
    ) -> Self {
        let tangent_frame = TangentFrame::new(normal, tangent);
        let texcoord = texcoord.to_array().map(f16::from_f32);
        let position = Position::new(position, bounding_sphere);

        debug_assert!({
            let (decoded_normal, _) = tangent_frame.into_normal_tangent();
            decoded_normal.abs_diff_eq(normal, 0.1)
        });

        Self {
            material: material as u16,
            tangent_frame,
            position,
            texcoord,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AssetPath {
    pub asset: PathBuf,
    pub cache: PathBuf,
}

impl AssetPath {
    pub fn new<A, B>(asset: A, cache: B) -> Self
    where
        A: AsRef<Path>,
        B: AsRef<Path>,
    {
        Self {
            asset: asset.as_ref().to_path_buf(),
            cache: cache.as_ref().to_path_buf(),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Scene {
    pub directional_light: DirectionalLight,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub textures: Vec<Texture>,
    pub materials: Vec<Material>,
    pub meshes: Vec<Mesh>,
    pub instances: Vec<Instance>,
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

    pub fn visit_instances<F, R>(&self, mut cb: F)
    where
        F: FnMut(&Instance, Option<&R>) -> R,
    {
        fn visit<F, R>(instance: &Instance, ret: Option<&R>, cb: &mut F)
        where
            F: FnMut(&Instance, Option<&R>) -> R,
        {
            let ret = cb(instance, ret);

            for child in &instance.children {
                visit(child, Some(&ret), cb);
            }
        }

        for instance in &self.instances {
            visit(instance, None, &mut cb);
        }
    }

    pub fn from_gltf(path: &AssetPath) -> Result<Self> {
        let scene = fs::read(&path.cache)
            .ok()
            .and_then(|bytes| bincode::deserialize(&bytes).ok());

        if let Some(scene) = scene {
            Ok(scene)
        } else {
            let importer = gltf::Importer::new(&path.asset).wrap_err_with(|| {
                format!("failed creating gltf importer for file at {:?}", path.asset)
            })?;

            let scene = importer.load_scene().wrap_err_with(|| {
                format!("failed loading gltf scene for file at {:?}", path.asset)
            })?;

            scene.serialize(&path.cache);

            Ok(scene)
        }
    }

    fn serialize(&self, path: &Path) {
        match bincode::serialize(&self) {
            Ok(bytes) => {
                if let Err(err) = fs::write(path, &bytes) {
                    eprintln!("failed to cache scene: {err}");
                } else {
                    println!("cached scene to: {:?}", path);
                }
            }
            Err(err) => {
                eprintln!("failed serialize scene: {err}");
            }
        }
    }
}
