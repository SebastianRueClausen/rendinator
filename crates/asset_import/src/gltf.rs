use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::{fs, io, mem};

use asset::{
    BoundingSphere, Instance, Lod, Material, Mesh, Meshlet, Model, Scene,
    Texture, TextureKind, Transform, Vertex,
};
use eyre::{Result, WrapErr};
use glam::{Mat4, Vec2, Vec3, Vec4};
use gltf::Gltf;

use crate::{normals, Progress, Stage};

#[derive(Default)]
struct FallbackTextures {
    albedo: Option<u32>,
    emissive: Option<u32>,
    normal: Option<u32>,
    specular: Option<u32>,
}

impl FallbackTextures {
    fn insert_fallback(scene: &mut Scene, specs: &TextureSpecs) -> u32 {
        let mip: Vec<_> =
            specs.fallback.iter().copied().cycle().take(16 * 4).collect();
        let mip = compress_bytes(specs.kind, 4, 4, &mip).into_boxed_slice();
        scene.add_texture(Texture {
            kind: specs.kind,
            width: 4,
            height: 4,
            mips: vec![mip],
        })
    }

    fn fallback_texture(
        &mut self,
        scene: &mut Scene,
        kind: TextureKind,
    ) -> u32 {
        let (field, specs) = match kind {
            TextureKind::Albedo => (&mut self.albedo, &ALBEDO_TEXTURE_SPECS),
            TextureKind::Normal => (&mut self.normal, &NORMAL_TEXTURE_SPECS),
            TextureKind::Specular => {
                (&mut self.specular, &SPECULAR_TEXTURE_SPECS)
            }
            TextureKind::Emissive => {
                (&mut self.emissive, &EMISSIVE_TEXTURE_SPECS)
            }
        };
        *field.get_or_insert_with(|| Self::insert_fallback(scene, specs))
    }

    fn fallback_material(&mut self, scene: &mut Scene) -> Material {
        Material {
            albedo_texture: self.fallback_texture(scene, TextureKind::Albedo),
            normal_texture: self.fallback_texture(scene, TextureKind::Normal),
            specular_texture: self
                .fallback_texture(scene, TextureKind::Specular),
            emissive_texture: self
                .fallback_texture(scene, TextureKind::Emissive),
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
    progress: Option<Arc<Mutex<Progress>>>,
    work_amount: f32,
}

impl Importer {
    pub fn new(
        path: &Path,
        progress: Option<Arc<Mutex<Progress>>>,
    ) -> Result<Self> {
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
                    .ok_or_else(|| {
                        eyre::eyre!("failed to load inline binary data")
                    })
                    .cloned()
                    .map(Vec::into_boxed_slice),
                gltf::buffer::Source::Uri(uri) => {
                    let binary_path: PathBuf =
                        [parent_path.as_path(), Path::new(uri)]
                            .iter()
                            .collect();
                    fs::read(&binary_path)
                        .wrap_err_with(|| {
                            eyre::eyre!(
                                "failed to load binary data from \
                                 {binary_path:?}"
                            )
                        })
                        .map(Vec::into_boxed_slice)
                }
            })
            .collect();
        let work_amount =
            gltf.meshes().len() as f32 + gltf.materials().len() as f32;
        Ok(Self {
            buffer_data: buffer_data?,
            parent_path,
            gltf,
            progress,
            work_amount,
        })
    }

    fn image(
        &self,
        source: gltf::image::Source,
    ) -> Result<image::DynamicImage> {
        match source {
            gltf::image::Source::View { view, mime_type } => {
                let format = match mime_type {
                    "image/png" => image::ImageFormat::Png,
                    "image/jpeg" => image::ImageFormat::Jpeg,
                    _ => {
                        return Err(eyre::eyre!(
                            "invalid image type, must be png or jpg"
                        ))
                    }
                };
                let input = self.buffer_data(&view, None, 0);
                let cursor = io::Cursor::new(&input);
                image::load(cursor, format)
                    .wrap_err("failed to load inline image")
            }
            gltf::image::Source::Uri { uri, .. } => {
                let uri = Path::new(uri);
                let path: PathBuf =
                    [self.parent_path.as_path(), Path::new(uri)]
                        .iter()
                        .collect();
                image::open(&path).wrap_err_with(|| {
                    format!("failed to load image from {uri:?}")
                })
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
        let end = start + byte_count.unwrap_or(view.length() - offset);
        &self.buffer_data[view.buffer().index()][start..end]
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
            if let Some(accessor) =
                material.pbr_metallic_roughness().base_color_texture()
            {
                let image = self.image(accessor.texture().source().source())?;
                let texture =
                    create_texture(image, &ALBEDO_TEXTURE_SPECS, true, |_| ());
                scene.add_texture(texture)
            } else {
                fallback_textures.fallback_texture(scene, TextureKind::Albedo)
            }
        };

        let emissive_texture = {
            if let Some(accessor) = material.emissive_texture() {
                let image = self.image(accessor.texture().source().source())?;
                let texture = create_texture(
                    image,
                    &EMISSIVE_TEXTURE_SPECS,
                    true,
                    |_| (),
                );
                scene.add_texture(texture)
            } else {
                fallback_textures.fallback_texture(scene, TextureKind::Emissive)
            }
        };

        let normal_texture = {
            if let Some(accessor) = material.normal_texture() {
                let image = self.image(accessor.texture().source().source())?;
                let texture = create_texture(
                    image,
                    &NORMAL_TEXTURE_SPECS,
                    true,
                    |pixel| {
                        let normal = Vec3 {
                            x: (pixel[0] as f32) / 255.0,
                            y: (pixel[1] as f32) / 255.0,
                            z: (pixel[2] as f32) / 255.0,
                        };

                        let normal = normal * 2.0 - 1.0;
                        let uv = asset::normal::encode_octahedron(normal);

                        pixel[0] =
                            asset::utilities::quantize_unorm(uv.x, 8) as u8;
                        pixel[1] =
                            asset::utilities::quantize_unorm(uv.y, 8) as u8;
                    },
                );

                scene.add_texture(texture)
            } else {
                fallback_textures.fallback_texture(scene, TextureKind::Normal)
            }
        };

        let specular_texture = {
            let accessor =
                material.pbr_metallic_roughness().metallic_roughness_texture();
            if let Some(accessor) = accessor {
                let image = self.image(accessor.texture().source().source())?;
                let texture = create_texture(
                    image,
                    &SPECULAR_TEXTURE_SPECS,
                    true,
                    |rgba| {
                        // Change metallic channel to red.
                        rgba[0] = rgba[2];
                    },
                );
                scene.add_texture(texture)
            } else {
                fallback_textures.fallback_texture(scene, TextureKind::Specular)
            }
        };

        Ok(Material {
            metallic: material.pbr_metallic_roughness().metallic_factor(),
            roughness: material.pbr_metallic_roughness().roughness_factor(),
            ior: material.ior().unwrap_or(DEFAULT_IOR),
            base_color: material
                .pbr_metallic_roughness()
                .base_color_factor()
                .into(),
            emissive: Vec3::from_array(material.emissive_factor()).extend(1.0)
                * material.emissive_strength().unwrap_or(1.0),
            albedo_texture,
            normal_texture,
            specular_texture,
            emissive_texture,
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
    fn load_model(
        &self,
        scene: &mut Scene,
        fallback_textures: &mut FallbackTextures,
        mesh: gltf::Mesh,
    ) -> Result<Model> {
        use gltf::accessor::{DataType, Dimensions};
        let meshes: Result<Vec<_>> = mesh
            .primitives()
            .map(|primitive| {
                let material = primitive
                    .material()
                    .index()
                    .map(|material| material as u32)
                    .unwrap_or_else(|| {
                        let material =
                            fallback_textures.fallback_material(scene);
                        scene.add_material(material)
                    });

                let mut indices = self.load_indices(&primitive)?;

                let position_accessor = primitive
                    .get(&gltf::Semantic::Positions)
                    .ok_or_else(|| {
                        eyre::eyre!("primitive doesn't have vertex positions")
                    })?;

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
                    None => normals::generate_normals(&positions, &indices),
                    Some(accessor) => {
                        verify_accessor(
                            "normals",
                            &accessor,
                            DataType::F32,
                            Dimensions::Vec3,
                        )?;
                        self.accessor_data(&accessor)
                            .chunks(mem::size_of::<Vec3>())
                            .map(bytemuck::pod_read_unaligned)
                            .collect()
                    }
                };

                let texcoords =
                    match primitive.get(&gltf::Semantic::TexCoords(0)) {
                        None => vec![Vec2::ZERO; normals.len()],
                        Some(accessor) => {
                            verify_accessor(
                                "texcoords",
                                &accessor,
                                DataType::F32,
                                Dimensions::Vec2,
                            )?;
                            self.accessor_data(&accessor)
                                .chunks(mem::size_of::<Vec2>())
                                .map(bytemuck::pod_read_unaligned)
                                .collect()
                        }
                    };

                let tangents = match primitive.get(&gltf::Semantic::Tangents) {
                    None => normals::generate_tangents(
                        &positions, &texcoords, &normals, &indices,
                    )?,
                    Some(accessor) => {
                        verify_accessor(
                            "tangents",
                            &accessor,
                            DataType::F32,
                            Dimensions::Vec4,
                        )?;
                        self.accessor_data(&accessor)
                            .chunks(mem::size_of::<Vec4>())
                            .map(bytemuck::pod_read_unaligned)
                            .collect()
                    }
                };

                let bounding_sphere = bounding_sphere(&primitive);
                let vertices: Vec<_> = texcoords
                    .iter()
                    .cloned()
                    .zip(normals.iter().cloned())
                    .zip(tangents.iter().cloned())
                    .zip(positions.iter().cloned())
                    .map(|(((texcoord, normal), tangent), position)| {
                        Vertex::encode(
                            &bounding_sphere,
                            position,
                            normal,
                            texcoord,
                            tangent,
                            material,
                        )
                    })
                    .collect();

                let vertex_adapter = meshopt::VertexDataAdapter {
                    reader: Cursor::new(bytemuck::cast_slice(&positions)),
                    vertex_stride: mem::size_of::<Vec3>(),
                    vertex_count: positions.len(),
                    position_offset: 0,
                };

                let vertex_offset = scene.vertices.len();
                scene.vertices.extend(vertices);

                let mut lods = [Lod::default(); 8];
                let mut lod_count = 0;

                for lod in &mut lods {
                    lod.index_offset = scene.indices.len() as u32;
                    lod.index_count = indices.len() as u32;
                    lod_count += 1;

                    scene.indices.extend_from_slice(&indices);

                    lod.meshlet_count =
                        build_meshlets(scene, &indices, &vertex_adapter) as u32;
                    lod.meshlet_offset = scene.meshlets.len() as u32;

                    if lod_count < 8 {
                        let target_count =
                            (indices.len() as f32 * 0.75).ceil() as usize;
                        let new_indices = meshopt::simplify(
                            &indices,
                            &vertex_adapter,
                            target_count,
                            1e-2,
                        );

                        if new_indices.len() == indices.len() {
                            break;
                        }

                        indices = new_indices;
                    }
                }

                while scene.meshlets.len() % 64 != 0 {
                    scene.meshlets.push(Meshlet::default());
                }

                Ok(Mesh {
                    vertex_offset: vertex_offset as u32,
                    vertex_count: texcoords.len() as u32,
                    bounding_sphere,
                    material,
                    lods,
                    lod_count,
                })
            })
            .collect();
        let mesh_indices =
            meshes?.into_iter().map(|mesh| scene.add_mesh(mesh)).collect();
        Ok(Model { mesh_indices })
    }

    fn progress_tick(&self) {
        if let Some(progress) = &self.progress {
            progress.lock().unwrap().percentage += self.work_amount.recip();
        }
    }

    fn progress_stage(&self, stage: Stage) {
        if let Some(progress) = &self.progress {
            progress.lock().unwrap().stage = stage;
        }
    }

    pub fn load_scene(self) -> Result<Scene> {
        let mut scene = Scene::default();
        let mut fallback_textures = FallbackTextures::default();

        scene.instances =
            load_instances(self.gltf.scenes().flat_map(|scene| scene.nodes()));

        self.progress_stage(Stage::Textures);
        scene.materials = self
            .gltf
            .materials()
            .map(|material| {
                self.progress_tick();
                self.load_material(&mut scene, &mut fallback_textures, material)
            })
            .collect::<Result<_>>()?;

        self.progress_stage(Stage::Meshes);
        scene.models = self
            .gltf
            .meshes()
            .map(|mesh| {
                self.progress_tick();
                self.load_model(&mut scene, &mut fallback_textures, mesh)
            })
            .collect::<Result<_>>()?;

        Ok(scene)
    }
}

fn build_meshlets(
    scene: &mut Scene,
    indices: &[u32],
    vertices: &meshopt::VertexDataAdapter,
) -> usize {
    let max_vertices = 64;
    let max_triangles = 64;
    let meshlets = meshopt::clusterize::build_meshlets(
        indices,
        vertices.vertex_count,
        max_vertices,
        max_triangles,
    );
    let mut meshlets: Vec<_> = meshlets
        .iter()
        .map(|meshlet| {
            let data_offset = scene.meshlet_data.len() as u32;
            let meshlet_vertices =
                &meshlet.vertices[..meshlet.vertex_count as usize];
            let meshlet_indices =
                &meshlet.indices[..meshlet.triangle_count as usize];
            scene.meshlet_data.extend_from_slice(meshlet_vertices);
            for triangle in meshlet_indices {
                let [a, b, c] = triangle.map(|i| i as u32);
                scene.meshlet_data.push(a << 24 | b << 16 | c << 8);
            }
            let bounds = meshopt::compute_meshlet_bounds(meshlet, vertices);
            let bounding_sphere = BoundingSphere {
                center: Vec3::from_array(bounds.center),
                radius: bounds.radius,
            };
            Meshlet {
                bounding_sphere,
                cone_axis: bounds.cone_axis_s8,
                cone_cutoff: bounds.cone_cutoff_s8,
                vertex_count: meshlet.vertex_count,
                triangle_count: meshlet.triangle_count,
                data_offset,
            }
        })
        .collect();
    let count = meshlets.len();
    scene.meshlets.append(&mut meshlets);
    count
}

fn bounding_sphere(primitive: &gltf::Primitive) -> BoundingSphere {
    let bounding_box = primitive.bounding_box();
    let min = Vec3::from(bounding_box.min);
    let max = Vec3::from(bounding_box.max);

    let center = min + (max - min) * 0.5;

    BoundingSphere { radius: (center - max).length(), center }
}

fn load_instances<'a>(
    nodes: impl Iterator<Item = gltf::Node<'a>>,
) -> Vec<Instance> {
    nodes
        .map(|node| {
            let model_index = node.mesh().map(|mesh| mesh.index() as u32);
            let transform = Transform::from(Mat4::from_cols_array_2d(
                &node.transform().matrix(),
            ));
            let children = load_instances(node.children());
            Instance { model_index, transform, children }
        })
        .collect()
}

struct TextureSpecs {
    kind: TextureKind,
    fallback: [u8; 4],
}

fn create_texture(
    mut image: image::DynamicImage,
    specs: &TextureSpecs,
    create_mips: bool,
    mut encode: impl FnMut(&mut [u8]) + Clone + Copy,
) -> Texture {
    let width = image.width().next_multiple_of(4);
    let height = image.height().next_multiple_of(4);

    let mip_level_count = if create_mips {
        let extent = u32::min(width, height) as f32;
        let count = extent.log2().floor() as u32;
        count.saturating_sub(2) + 1
    } else {
        1
    };

    let min_width = width >> (mip_level_count - 1);
    let min_height = width >> (mip_level_count - 1);
    assert!(
        min_width >= 4 && min_height >= 4,
        "smallest mip is too small: {min_width} x {min_height}",
    );

    let filter = image::imageops::FilterType::Lanczos3;

    let mips = (0..mip_level_count)
        .map(|level| {
            image = if level == 0 {
                image.resize_exact(width, height, filter)
            } else {
                let width = (image.width() / 2).next_multiple_of(4);
                let height = (image.height() / 2).next_multiple_of(4);
                image.resize_exact(width, height, filter)
            };

            let mut mip = image.clone().into_rgba8();
            mip.pixels_mut().for_each(|pixel| encode(&mut pixel.0));

            compress_bytes(
                specs.kind,
                mip.width(),
                mip.height(),
                &mip.into_raw(),
            )
            .into_boxed_slice()
        })
        .collect();
    Texture { kind: specs.kind, mips, width, height }
}

fn compress_bytes(
    kind: TextureKind,
    width: u32,
    height: u32,
    bytes: &[u8],
) -> Vec<u8> {
    let format = match kind {
        TextureKind::Albedo | TextureKind::Emissive => texpresso::Format::Bc1,
        TextureKind::Normal | TextureKind::Specular => texpresso::Format::Bc5,
    };
    let params = texpresso::Params {
        algorithm: texpresso::Algorithm::ClusterFit,
        ..Default::default()
    };
    let (width, height) = (width as usize, height as usize);
    let size = format.compressed_size(width, height);
    let mut output = vec![0x0; size];
    format.compress(bytes, width, height, params, &mut output);
    output
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

const ALBEDO_TEXTURE_SPECS: TextureSpecs =
    TextureSpecs { kind: TextureKind::Albedo, fallback: [u8::MAX; 4] };

const NORMAL_TEXTURE_SPECS: TextureSpecs = TextureSpecs {
    kind: TextureKind::Normal,
    // Octahedron encoded normal pointing straight out.
    fallback: [128; 4],
};

const SPECULAR_TEXTURE_SPECS: TextureSpecs =
    TextureSpecs { kind: TextureKind::Specular, fallback: [u8::MAX; 4] };

const EMISSIVE_TEXTURE_SPECS: TextureSpecs =
    TextureSpecs { kind: TextureKind::Emissive, fallback: [u8::MAX; 4] };

const DEFAULT_IOR: f32 = 1.4;
const DEFAULT_METALLIC: f32 = 0.0;
const DEFAULT_ROUGHNESS: f32 = 1.0;

const DEFAULT_COLOR: Vec4 = Vec4::splat(1.0);
const DEFAULT_EMISSIVE: Vec4 = Vec4::splat(0.0);
