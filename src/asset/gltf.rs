use eyre::{Result, WrapErr};
use glam::{Mat4, Vec2, Vec3, Vec4};
use gltf::Gltf;
use std::ops::Range;
use std::path::{Path, PathBuf};

use std::{fs, io, mem};

use crate::asset::{Primitive, Vertex};

use super::{
    normal, quantize, BoundingSphere, Instance, Material, Mesh, Scene, Texture, Transform,
};

#[derive(Default)]
struct FallbackTextures {
    albedo_fallback_texture: Option<u32>,
    emissive_fallback_texture: Option<u32>,
    normal_fallback_texture: Option<u32>,
    specular_fallback_texture: Option<u32>,
}

impl FallbackTextures {
    fn albedo_fallback_texture(&mut self, scene: &mut Scene) -> u32 {
        *self.albedo_fallback_texture.get_or_insert_with(|| {
            scene.add_texture(fallback_texture(ALBEDO_MAP_RAW_FORMAT, [u8::MAX; 4]))
        })
    }

    fn emissive_fallback_texture(&mut self, scene: &mut Scene) -> u32 {
        *self.emissive_fallback_texture.get_or_insert_with(|| {
            scene.add_texture(fallback_texture(EMISSIVE_MAP_RAW_FORMAT, [255; 4]))
        })
    }

    fn normal_fallback_texture(&mut self, scene: &mut Scene) -> u32 {
        *self.normal_fallback_texture.get_or_insert_with(|| {
            let mut normal = [128, 128, 255, 255];
            octahedron_encode_pixel(&mut normal);

            scene.add_texture(fallback_texture(NORMAL_MAP_RAW_FORMAT, normal))
        })
    }

    fn specular_fallback_texture(&mut self, scene: &mut Scene) -> u32 {
        *self.specular_fallback_texture.get_or_insert_with(|| {
            scene.add_texture(fallback_texture(SPECULAR_MAP_RAW_FORMAT, [255; 2]))
        })
    }

    fn fallback_material(&mut self, scene: &mut Scene) -> Material {
        Material {
            albedo_texture: self.albedo_fallback_texture(scene),
            normal_texture: self.normal_fallback_texture(scene),
            specular_texture: self.specular_fallback_texture(scene),
            emissive_texture: self.emissive_fallback_texture(scene),
            base_color: DEFAULT_COLOR,
            emissive: DEFAULT_EMISSIVE,
            metallic: DEFAULT_METALLIC,
            roughness: DEFAULT_ROUGHNESS,
            ior: DEFAULT_IOR,
            padding: [0; 1],
        }
    }
}

pub struct Importer {
    gltf: Gltf,
    buffer_data: Vec<Box<[u8]>>,
    parent_path: PathBuf,
}

impl Importer {
    pub fn new(path: &Path) -> Result<Self> {
        let file = fs::File::open(path)?;
        let gltf = Gltf::from_reader(io::BufReader::new(file))?;
        let parent_path = path
            .parent()
            .ok_or_else(|| eyre::eyre!("path has no parent directory"))?
            .to_owned();

        let buffer_data: Result<Vec<_>, _> = gltf
            .buffers()
            .map(|buffer| match buffer.source() {
                gltf::buffer::Source::Bin => gltf
                    .blob
                    .as_ref()
                    .ok_or_else(|| eyre::eyre!("failed to load inline binary data"))
                    .cloned()
                    .map(Vec::into_boxed_slice),
                gltf::buffer::Source::Uri(uri) => {
                    let binary_path: PathBuf =
                        [parent_path.as_path(), Path::new(uri)].iter().collect();
                    fs::read(&binary_path)
                        .wrap_err_with(|| {
                            eyre::eyre!("failed to load binary data from {binary_path:?}")
                        })
                        .map(Vec::into_boxed_slice)
                }
            })
            .collect();

        Ok(Self {
            buffer_data: buffer_data?,
            parent_path,
            gltf,
        })
    }

    fn image(&self, source: gltf::image::Source) -> Result<image::DynamicImage> {
        match source {
            gltf::image::Source::View { view, mime_type } => {
                let format = match mime_type {
                    "image/png" => image::ImageFormat::Png,
                    "image/jpeg" => image::ImageFormat::Jpeg,
                    _ => return Err(eyre::eyre!("invalid image type, must be png or jpg")),
                };

                let input = self.buffer_data(&view, None, 0);
                image::load(io::Cursor::new(&input), format).wrap_err("failed to load inline image")
            }
            gltf::image::Source::Uri { uri, .. } => {
                let uri = Path::new(uri);
                let path: PathBuf = [self.parent_path.as_path(), Path::new(uri)]
                    .iter()
                    .collect();
                image::open(&path).wrap_err_with(|| format!("failed to load image from {uri:?}"))
            }
        }
    }

    fn buffer_data(
        &self,
        view: &gltf::buffer::View,
        byte_count: Option<usize>,
        offset: usize,
    ) -> &[u8] {
        let start = view.offset() + offset;
        let end = view.offset() + offset + byte_count.unwrap_or(view.length() - offset);
        let index = view.buffer().index();

        &self.buffer_data[index][start..end]
    }

    fn accessor_data(&self, accessor: &gltf::Accessor) -> &[u8] {
        let view = accessor.view().unwrap();
        let bytes = accessor.count() * accessor.size();
        self.buffer_data(&view, Some(bytes), accessor.offset())
    }

    fn load_material(
        &self,
        scene: &mut Scene,
        fallback_textures: &mut FallbackTextures,
        material: gltf::Material,
    ) -> Result<Material> {
        let albedo_texture = {
            if let Some(accessor) = material.pbr_metallic_roughness().base_color_texture() {
                let image = self.image(accessor.texture().source().source())?;
                let texture = create_texture(image, ALBEDO_MAP_FORMAT, true, |_| ())?;
                scene.add_texture(texture)
            } else {
                fallback_textures.albedo_fallback_texture(scene)
            }
        };

        let emissive_texture = {
            if let Some(accessor) = material.emissive_texture() {
                let image = self.image(accessor.texture().source().source())?;
                let texture = create_texture(image, EMISSIVE_MAP_FORMAT, true, |_| ())?;
                scene.add_texture(texture)
            } else {
                fallback_textures.emissive_fallback_texture(scene)
            }
        };

        let normal_texture = {
            if let Some(accessor) = material.normal_texture() {
                let image = self.image(accessor.texture().source().source())?;
                let texture =
                    create_texture(image, NORMAL_MAP_FORMAT, true, octahedron_encode_pixel)?;
                scene.add_texture(texture)
            } else {
                fallback_textures.normal_fallback_texture(scene)
            }
        };

        let specular_texture = {
            let accessor = material
                .pbr_metallic_roughness()
                .metallic_roughness_texture();

            if let Some(accessor) = accessor {
                let image = self.image(accessor.texture().source().source())?;
                let texture = create_texture(image, SPECULAR_MAP_FORMAT, true, |rgba| {
                    // Change metallic channel to red.
                    rgba[0] = rgba[2];
                })?;
                scene.add_texture(texture)
            } else {
                fallback_textures.specular_fallback_texture(scene)
            }
        };

        let metallic = material.pbr_metallic_roughness().metallic_factor();
        let roughness = material.pbr_metallic_roughness().roughness_factor();
        let ior = material.ior().unwrap_or(DEFAULT_IOR);

        let base_color = Vec4::from_array(material.pbr_metallic_roughness().base_color_factor());

        let emissive = Vec3::from_array(material.emissive_factor()).extend(1.0)
            * material.emissive_strength().unwrap_or(1.0);

        Ok(Material {
            albedo_texture,
            emissive_texture,
            normal_texture,
            specular_texture,
            base_color,
            emissive,
            metallic,
            roughness,
            ior,
            padding: [0; 1],
        })
    }

    fn load_indices(&self, primitive: &gltf::Primitive) -> Result<Vec<u32>> {
        use gltf::accessor::{DataType, Dimensions};

        let accessor = primitive
            .indices()
            .ok_or_else(|| eyre::eyre!("primitive doesn't have indices"))?;

        if accessor.dimensions() != Dimensions::Scalar {
            return Err(eyre::eyre!("index attribute must be scalar",));
        }

        let index_data = self.accessor_data(&accessor);

        let indices = match accessor.data_type() {
            DataType::U32 => index_data
                .chunks(4)
                .map(|bytes| bytemuck::pod_read_unaligned(bytes))
                .collect(),
            DataType::U16 => index_data
                .chunks(2)
                .map(|bytes| bytemuck::pod_read_unaligned::<u16>(bytes) as u32)
                .collect(),
            ty => {
                return Err(eyre::eyre!("invalid index type {ty:?}"));
            }
        };

        Ok(indices)
    }

    /// Note: Call this after loading materials.
    fn load_mesh(
        &self,
        scene: &mut Scene,
        fallback_textures: &mut FallbackTextures,
        mesh: gltf::Mesh,
    ) -> Result<Mesh> {
        use gltf::accessor::{DataType, Dimensions};

        let primitives: Result<Vec<_>> = mesh
            .primitives()
            .map(|primitive| {
                let material = primitive
                    .material()
                    .index()
                    .map(|material| material as u32)
                    .unwrap_or_else(|| {
                        let material = fallback_textures.fallback_material(scene);
                        scene.add_material(material)
                    });

                let indices = self.load_indices(&primitive)?;

                let position_accessor = primitive
                    .get(&gltf::Semantic::Positions)
                    .ok_or_else(|| eyre::eyre!("primitive doesn't have vertex positions"))?;

                verify_accessor(
                    "positions",
                    &position_accessor,
                    DataType::F32,
                    Dimensions::Vec3,
                )?;

                let positions: Vec<Vec3> = self
                    .accessor_data(&position_accessor)
                    .chunks(mem::size_of::<Vec3>())
                    .map(bytemuck::pod_read_unaligned)
                    .collect();

                let normals = match primitive.get(&gltf::Semantic::Normals) {
                    None => generate_normals(&positions, &indices),
                    Some(accessor) => {
                        verify_accessor("normals", &accessor, DataType::F32, Dimensions::Vec3)?;
                        self.accessor_data(&accessor)
                            .chunks(mem::size_of::<Vec3>())
                            .map(bytemuck::pod_read_unaligned)
                            .collect()
                    }
                };

                let texcoords = match primitive.get(&gltf::Semantic::TexCoords(0)) {
                    None => vec![Vec2::ZERO; normals.len()],
                    Some(accessor) => {
                        verify_accessor("texcoords", &accessor, DataType::F32, Dimensions::Vec2)?;
                        self.accessor_data(&accessor)
                            .chunks(mem::size_of::<Vec2>())
                            .map(bytemuck::pod_read_unaligned)
                            .collect()
                    }
                };

                let tangents = match primitive.get(&gltf::Semantic::Tangents) {
                    None => generate_tangents(&positions, &texcoords, &normals, &indices)?,
                    Some(accessor) => {
                        verify_accessor("tangents", &accessor, DataType::F32, Dimensions::Vec4)?;
                        self.accessor_data(&accessor)
                            .chunks(mem::size_of::<Vec4>())
                            .map(bytemuck::pod_read_unaligned)
                            .collect()
                    }
                };

                let indices = load_indices(scene, &indices);
                let bounding_sphere = bounding_sphere(&primitive);

                scene.vertices.extend(
                    texcoords
                        .iter()
                        .cloned()
                        .zip(normals.iter().cloned())
                        .zip(tangents.iter().cloned())
                        .zip(positions.iter().cloned())
                        .map(|(((texcoord, normal), tangent), position)| {
                            Vertex::new(
                                &bounding_sphere,
                                position,
                                normal,
                                texcoord,
                                tangent,
                                material,
                            )
                        }),
                );

                Ok(Primitive {
                    bounding_sphere,
                    indices,
                    material,
                })
            })
            .collect();

        Ok(Mesh {
            primitives: primitives?,
        })
    }

    pub fn load_scene(self) -> Result<Scene> {
        let mut scene = Scene::default();
        let mut fallback_textures = FallbackTextures::default();

        scene.instances = load_instances(self.gltf.scenes().flat_map(|scene| scene.nodes()));
        scene.materials = self
            .gltf
            .materials()
            .map(|material| self.load_material(&mut scene, &mut fallback_textures, material))
            .collect::<Result<_>>()?;
        scene.meshes = self
            .gltf
            .meshes()
            .map(|mesh| self.load_mesh(&mut scene, &mut fallback_textures, mesh))
            .collect::<Result<_>>()?;

        Ok(scene)
    }
}

fn load_indices(scene: &mut Scene, indices: &[u32]) -> Range<u32> {
    let offset = scene.vertices.len() as u32;

    let start = scene.indices.len() as u32;
    let end = start + indices.len() as u32;

    scene
        .indices
        .extend(indices.iter().map(|index| index + offset));

    start..end
}

fn bounding_sphere(primitive: &gltf::Primitive) -> BoundingSphere {
    let bounding_box = primitive.bounding_box();
    let min = Vec3::from(bounding_box.min);
    let max = Vec3::from(bounding_box.max);

    let center = min + (max - min) * 0.5;

    BoundingSphere {
        radius: (center - max).length(),
        center,
    }
}

fn load_instances<'a>(nodes: impl Iterator<Item = gltf::Node<'a>>) -> Vec<Instance> {
    let nodes = nodes.map(|node| {
        let mesh = node.mesh().map(|mesh| mesh.index() as u32);
        let transform = Transform::from(Mat4::from_cols_array_2d(&node.transform().matrix()));

        let children = load_instances(node.children());
        let name = node.name().map(String::from);

        Instance {
            name,
            mesh,
            transform,
            children,
        }
    });

    nodes.collect()
}

fn create_texture(
    mut image: image::DynamicImage,
    format: wgpu::TextureFormat,
    create_mips: bool,
    mut encode: impl FnMut(&mut [u8]) + Clone + Copy,
) -> Result<Texture> {
    let mut mip_level_count = if create_mips {
        let extent = u32::max(image.width(), image.height()) as f32;
        extent.log2().floor() as u32 + 1
    } else {
        1
    };

    let extent = wgpu::Extent3d {
        width: image.width(),
        height: image.height(),
        depth_or_array_layers: 1,
    };

    let filter_type = image::imageops::FilterType::Lanczos3;

    let texture = match format {
        wgpu::TextureFormat::Bc1RgbaUnorm
        | wgpu::TextureFormat::Bc1RgbaUnormSrgb
        | wgpu::TextureFormat::Bc5RgSnorm
        | wgpu::TextureFormat::Bc5RgUnorm => {
            fn round_to_block_size(extent: u32) -> u32 {
                const BLOCK_SIZE: u32 = 4;
                let quarter = (extent + BLOCK_SIZE - 1) / BLOCK_SIZE;
                let rounded = quarter * BLOCK_SIZE;

                u32::max(rounded, BLOCK_SIZE)
            }

            if create_mips {
                // Block compressed textures can't have 1x1 and 2x2 mips.
                mip_level_count -= 2;
            }

            let mut mips = Vec::new();

            for level in 0..mip_level_count {
                image = if level == 0 {
                    image.resize_exact(
                        round_to_block_size(image.width()),
                        round_to_block_size(image.height()),
                        filter_type,
                    )
                } else {
                    image.resize_exact(
                        round_to_block_size(image.width() / 2),
                        round_to_block_size(image.height() / 2),
                        filter_type,
                    )
                };

                let mut mip = image.clone().into_rgba8();
                mip.pixels_mut().for_each(|pixel| encode(&mut pixel.0));
                mips.append(&mut compress_image(format, mip)?);
            }

            Texture {
                mips: mips.into_boxed_slice(),
                mip_level_count,
                extent,
                format,
            }
        }
        format => {
            if format.is_compressed() {
                return Err(eyre::eyre!("can't compress image with format {format:?}"));
            }

            let mut mips = Vec::new();
            for level in 0..mip_level_count {
                if level != 0 {
                    image = image.resize_exact(image.width() / 2, image.height() / 2, filter_type);
                }

                let mut raw = match format {
                    wgpu::TextureFormat::R8Unorm => {
                        let mut mip = image.clone().into_luma8();
                        mip.pixels_mut().for_each(|pixel| encode(&mut pixel.0));
                        mip.into_raw()
                    }
                    wgpu::TextureFormat::Rg8Unorm => {
                        let mut mip = image.clone().into_rgb8();
                        mip.pixels_mut().for_each(|pixel| encode(&mut pixel.0));

                        let size = mip.width() * mip.height() * 2;
                        let mut raw = Vec::with_capacity(size as usize);

                        for pixel in mip.pixels() {
                            raw.push(pixel[1]);
                            raw.push(pixel[2]);
                        }

                        raw
                    }
                    wgpu::TextureFormat::Rgba8UnormSrgb => {
                        let mut mip = image.clone().into_rgba8();
                        mip.pixels_mut().for_each(|pixel| encode(&mut pixel.0));
                        mip.into_raw()
                    }
                    format => {
                        return Err(eyre::eyre!("invalid format: {format:?}"));
                    }
                };

                mips.append(&mut raw);
            }

            Texture {
                mips: mips.into_boxed_slice(),
                mip_level_count,
                extent,
                format,
            }
        }
    };

    Ok(texture)
}

fn fallback_texture<const N: usize>(format: wgpu::TextureFormat, pixel: [u8; N]) -> Texture {
    let extent = wgpu::Extent3d {
        depth_or_array_layers: 1,
        width: 4,
        height: 4,
    };

    let mips = (0..16).flat_map(|_| pixel).collect();

    Texture {
        mip_level_count: 1,
        extent,
        mips,
        format,
    }
}

fn octahedron_encode_pixel(pixel: &mut [u8]) {
    let normal = Vec3 {
        x: (pixel[0] as f32) / 255.0,
        y: (pixel[1] as f32) / 255.0,
        z: (pixel[2] as f32) / 255.0,
    };

    let normal = normal * 2.0 - 1.0;
    let uv = normal::encode_octahedron(normal);

    pixel[0] = quantize::quantize_unorm::<8>(uv.x) as u8;
    pixel[1] = quantize::quantize_unorm::<8>(uv.y) as u8;
}

fn compress_image(format: wgpu::TextureFormat, image: image::RgbaImage) -> Result<Vec<u8>> {
    let format = match format {
        wgpu::TextureFormat::Bc1RgbaUnorm | wgpu::TextureFormat::Bc1RgbaUnormSrgb => {
            texpresso::Format::Bc1
        }
        wgpu::TextureFormat::Bc5RgSnorm | wgpu::TextureFormat::Bc5RgUnorm => texpresso::Format::Bc5,
        format => {
            panic!("invalid format {format:?}");
        }
    };

    let params = texpresso::Params {
        algorithm: texpresso::Algorithm::IterativeClusterFit,
        ..Default::default()
    };

    let (width, height) = (image.width() as usize, image.height() as usize);
    let size = format.compressed_size(width, height);

    let mut output = vec![0x0; size];
    format.compress(&image.into_raw(), width, height, params, &mut output);

    Ok(output)
}

fn verify_accessor(
    name: &str,
    accessor: &gltf::Accessor,
    data_type: gltf::accessor::DataType,
    dimensions: gltf::accessor::Dimensions,
) -> Result<()> {
    if accessor.data_type() != data_type {
        return Err(eyre::eyre!(
            "{name} attribute should be of type {:?} but is {:?}",
            data_type,
            accessor.data_type(),
        ));
    }

    if accessor.dimensions() != dimensions {
        return Err(eyre::eyre!(
            "{name} attribute should have dimensions {:?} but is {:?}",
            dimensions,
            accessor.dimensions(),
        ));
    }

    Ok(())
}

fn generate_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];

    for triangle in indices.chunks(3) {
        let [idx0, idx1, idx2] = triangle else {
            panic!("indices isn't multiple of 3");
        };

        let a = positions[*idx0 as usize];
        let b = positions[*idx1 as usize];
        let c = positions[*idx2 as usize];

        let normal = (b - a).cross(c - a);
        normals[*idx0 as usize] += normal;
        normals[*idx1 as usize] += normal;
        normals[*idx2 as usize] += normal;
    }

    for normal in &mut normals {
        *normal = normal.normalize();
    }

    normals
}

fn generate_tangents(
    positions: &[Vec3],
    texcoords: &[Vec2],
    normals: &[Vec3],
    indices: &[u32],
) -> Result<Vec<Vec4>> {
    let tangents = vec![Vec4::ZERO; positions.len()];

    let mut generator = TangentGenerator {
        positions,
        texcoords,
        normals,
        indices,
        tangents,
    };

    if !mikktspace::generate_tangents(&mut generator) {
        return Err(eyre::eyre!("failed to generate tangents"));
    }

    for tangent in &mut generator.tangents {
        tangent[3] *= -1.0;
    }

    Ok(generator.tangents)
}

struct TangentGenerator<'a> {
    positions: &'a [Vec3],
    texcoords: &'a [Vec2],
    normals: &'a [Vec3],
    indices: &'a [u32],
    tangents: Vec<Vec4>,
}

impl<'a> TangentGenerator<'a> {
    fn index(&self, face: usize, vertex: usize) -> usize {
        self.indices[face * 3 + vertex] as usize
    }
}

impl<'a> mikktspace::Geometry for TangentGenerator<'a> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.index(face, vert)].into()
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.index(face, vert)].into()
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.texcoords[self.index(face, vert)].into()
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        let index = self.index(face, vert);
        self.tangents[index] = tangent.into();
    }
}

const ALBEDO_MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bc1RgbaUnormSrgb;
const ALBEDO_MAP_RAW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

const NORMAL_MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bc5RgUnorm;
const NORMAL_MAP_RAW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;

const SPECULAR_MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bc5RgUnorm;
const SPECULAR_MAP_RAW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;

const EMISSIVE_MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bc1RgbaUnormSrgb;
const EMISSIVE_MAP_RAW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

const DEFAULT_IOR: f32 = 1.4;
const DEFAULT_METALLIC: f32 = 0.0;
const DEFAULT_ROUGHNESS: f32 = 1.0;

const DEFAULT_COLOR: Vec4 = Vec4::splat(1.0);
const DEFAULT_EMISSIVE: Vec4 = Vec4::splat(0.0);
