#![feature(iterator_try_collect)]

use anyhow::{anyhow, Result};
use glam::{Vec2, Vec3, Vec4, Mat4, Vec2Swizzles, Vec3Swizzles};
use image::imageops::FilterType;

use std::path::{Path, PathBuf};
use std::{fs, io, mem};

use asset::*;

#[repr(C)]
#[derive(Default, Clone, Copy, bytemuck::NoUninit)]
struct RawVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub texcoord: Vec2,
    pub tangent: Vec4,
}

#[derive(Clone, clap::ValueEnum)]
enum AssetKind {
    Font,
    Gltf,
}

#[derive(clap::Parser)]
struct Args {
    /// The kind of asset.
    #[clap(arg_enum)]
    asset_kind: AssetKind,

    /// The input files.
    /// 
    /// Font:
    ///
    ///   A single json file in the angelcode bm font format. The bitmap is expected to be encoded
    ///   as signed distance fields.
    ///
    /// Skybox:
    ///
    ///   A list of 6 images.
    ///
    ///   The images are expected to be in the order:
    ///     - positive x
    ///     - negaive x
    ///     - positive y
    ///     - negaive y
    ///     - positive z
    ///     - negaive z
    ///
    /// Gltf:
    ///
    ///  A gltf scene file,
    ///
    inputs: Vec<PathBuf>,

    /// The ouput file.
    #[clap(short, long)]
    ouput: Option<PathBuf>, 
}

fn main() -> Result<()> {
    use clap::Parser;

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()?;

    let args = Args::parse();

    match &args.asset_kind {
        AssetKind::Gltf => {
            let Some(input) = args.inputs.first() else {
                return Err(anyhow!("expected a single gltf input file"));
            };
            let output = args.ouput
                .as_ref()
                .map(|path| path.as_path())
                .unwrap_or(&Path::new("out.scene"));
            let res = load_scene_from_gltf(&input)?.store(output);
            if let Err(err) = res {
                return Err(anyhow!("failed to store scene to {output:?}: {err}"));
            }
        },
        AssetKind::Font => {
            let Some(input) = args.inputs.first() else {
                return Err(anyhow!("expected a single font metadata file"));
            };
            let output = args.ouput
                .as_ref()
                .map(|path| path.as_path())
                .unwrap_or(&Path::new("out.font"));
            let res = load_font(&input)?.store(output);
            if let Err(err) = res {
                return Err(anyhow!("failed to store font to {output:?}: {err}"));
            }
        }
    }

    Ok(())
}

fn create_image<F>(
    mut image: image::DynamicImage,
    format: ImageFormat,
    encode: F,
    mip_levels: usize,
) -> Image
where
    F: FnMut(&mut image::Rgba<u8>) + Clone + Copy
{
    match format {
        ImageFormat::Bc(bc) => {
            // Subtract 2 from the mip count to exclude mip levels of size 1x1 and 2x2, which
            // aren't allowed for block compressed textures.
            let mip_levels = mip_levels - 2;

            let (width, height) = (image.width(), image.height());
            let mips: Vec<Vec<u8>> = (0..mip_levels)
                .map(|level| {
                    let round_to_block = |val: u32| -> u32 {
                        (((val + 3) / 4) * 4).max(4) 
                    };

                    image = if level == 0 {
                        // Make sure the image dimensions are aligned to the right block size.
                        image.resize_exact(
                            round_to_block(image.width()),
                            round_to_block(image.height()),
                            FilterType::Lanczos3,
                        )
                    } else {
                        image.resize_exact(
                            round_to_block(image.width() / 2),
                            round_to_block(image.height() / 2),
                            FilterType::Lanczos3,
                        )
                    };

                    // Not effecient at all!
                    let mut mip = image.clone().into_rgba8();
                    let (width, height) = (mip.width() as usize, mip.height() as usize);

                    mip.pixels_mut().for_each(encode);

                    let format = match bc {
                        BcFormat::Bc5Unorm => texpresso::Format::Bc5,
                        BcFormat::Bc1Unorm | BcFormat::Bc1Srgb => texpresso::Format::Bc1,
                    };

                    let size = format.compressed_size(width as usize, height as usize);
                    let data = mip.into_raw();

                    let params = texpresso::Params {
                        algorithm: texpresso::Algorithm::IterativeClusterFit,
                        ..Default::default()
                    };

                    let mut output = vec![0x0; size];
                    format.compress(&data, width as usize, height as usize, params, &mut output);

                    output
                })
                .collect();

            Image { width, height, format, mips }
        }
        ImageFormat::Raw(raw) => {
            // The width and height of mip level 0.
            let (width, height) = (image.width(), image.height());

            let mips: Vec<_> = (0..mip_levels)
                .map(|level| level as u32)
                .map(|level| {
                    if level != 0 {
                        image = image.resize_exact(
                            image.width() / 2,
                            image.height() / 2,
                            FilterType::Lanczos3
                        );
                    }

                    let mip = image.clone();

                    match raw {
                        RawFormat::Rgba8Unorm | RawFormat::Rgba8Srgb => mip
                            .into_rgba8()
                            .into_raw(),
                        RawFormat::Rg8Unorm => {
                            let mip = mip.into_rgba8();
                            let mut data = Vec::with_capacity((mip.width() * mip.height() * 2) as usize);

                            for px in mip.pixels() {
                                data.push(px[1]);
                                data.push(px[2]);
                            } 
                        
                            data
                        }
                        RawFormat::R8Unorm => mip
                            .into_luma8()
                            .into_raw(),
                    }
                })
                .collect();

            Image { width, height, format, mips }
        }
    }
}

fn fallback_image<const N: usize>(format: RawFormat, rgba: [u8; N]) -> Image {
    let format = ImageFormat::Raw(format);
    let (width, height) = (4, 4);

    let mips = vec![(0..4 * 4)
        .flat_map(|_| rgba)
        .collect()];

    Image { width, height, format, mips }
}

fn fallback_albedo_map() -> Image {
    fallback_image(RawFormat::Rgba8Srgb, [255; 4])
}

fn fallback_normal_map() -> Image {
    fallback_image(RawFormat::Rgba8Unorm, [128, 128, 255, 255])
}

fn fallback_specular_map() -> Image {
    fallback_image(RawFormat::Rg8Unorm, [0, 0])
}

fn mip_level_count(image: &image::DynamicImage) -> usize {
    (image.width().max(image.height()) as f32).log2().floor() as usize + 1
}

struct GltfImporter {
    buffer_data: Vec<Box<[u8]>>,
    parent_path: PathBuf,
    gltf: gltf::Gltf,
}

impl GltfImporter {
    fn new(path: &Path) -> Result<Self> {
        let file = match fs::File::open(path) {
            Ok(file) => file,
            Err(err) => {
                return Err(anyhow!("can't read file {path:?}: {err}"));
            }
        };

        let reader = io::BufReader::new(file);
        let gltf = gltf::Gltf::from_reader(reader)?;

        let parent_path = path
            .parent()
            .expect("`path` doesn't have a parent directory")
            .to_path_buf();

        let buffer_data: Result<Vec<_>> = gltf
            .buffers()
            .map(|buffer| {
                Ok(match buffer.source() {
                    gltf::buffer::Source::Bin => gltf.blob
                        .as_ref()
                        .map(|blob| blob.clone().into_boxed_slice())
                        .ok_or_else(|| anyhow!("no binary blob in gltf scene"))?,
                    gltf::buffer::Source::Uri(uri) => {
                        let path: PathBuf = [parent_path.as_path(), Path::new(uri)].iter().collect();
                        fs::read(&path)
                            .map_err(|err| anyhow!("can't read file {path:?}: {err}"))?
                            .into_boxed_slice()
                    }
                })
            })
            .collect();

        Ok(Self { gltf, buffer_data: buffer_data?, parent_path })
    }

    fn get_buffer_data(
        &self,
        view: &gltf::buffer::View,
        offset: usize,
        size: Option<usize>,
    ) -> &[u8] {
        let start = view.offset() + offset;
        let end = view.offset() + offset + size.unwrap_or(view.length() - offset);
        &self.buffer_data[view.buffer().index()][start..end]
    }

    fn load_image(&self, source: &gltf::image::Source) -> Result<image::DynamicImage> {
        let image = match source {
            gltf::image::Source::View { view, mime_type } => {
                let format = match *mime_type {
                    "image/png" => image::ImageFormat::Png,
                    "image/jpeg" => image::ImageFormat::Jpeg,
                    _ => return Err(anyhow!("image must be either png of jpeg")),
                };

                let input = self.get_buffer_data(view, 0, None);
                image::load(io::Cursor::new(&input), format)?
            }
            gltf::image::Source::Uri { uri, .. } => {
                let uri = Path::new(uri);

                let path: PathBuf = [self.parent_path.as_path(), Path::new(uri)]
                    .iter()
                    .collect();

                image::open(&path)?
            }
        };

        Ok(image)
    }

    fn load_scene(self) -> Result<Scene> {
        let mut textures = Vec::default();

        for mat in self.gltf.materials() {
            let albedo_map = mat
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|texture| {
                    let image = texture
                        .texture()
                        .source()
                        .source();
                    self.load_image(&image).map(|image| {
                        let mip_levels = mip_level_count(&image);
                        create_image(image, ImageFormat::Bc(BcFormat::Bc1Srgb), |_| {}, mip_levels)
                    })
                })
                .unwrap_or_else(|| {
                    let base_color = mat
                        .pbr_metallic_roughness()
                        .base_color_factor()
                        .map(|factor| (255.0 / factor) as u8);

                    Ok(fallback_image(RawFormat::Rgba8Srgb, base_color))
                })?;

            let normal_map = mat
                .normal_texture()
                .map(|texture| {
                    let image = texture
                        .texture()
                        .source()
                        .source();
                    self.load_image(&image).map(|image| {
                        let mip_levels = mip_level_count(&image);
                        create_image(image, ImageFormat::Bc(BcFormat::Bc1Unorm), |_| {}, mip_levels)
                    })
                })
                .unwrap_or_else(|| {
                    Ok(fallback_normal_map())
                })?;

            let specular_map = mat
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
                .map(|texture| {
                    let image = texture
                        .texture()
                        .source()
                        .source();
                    self.load_image(&image).map(|image| {
                        let mip_count = mip_level_count(&image);
                        let format = ImageFormat::Bc(BcFormat::Bc5Unorm);

                        let encode = |px: &mut image::Rgba<u8>| {
                            px.0[0] = px.0[1];
                            px.0[1] = px.0[2];
                        };

                        create_image(image, format, encode, mip_count)
                    })
                })
                .unwrap_or_else(|| {
                    let metallic_factor = mat
                        .pbr_metallic_roughness()
                        .metallic_factor();
                    let roughness_factor = mat
                        .pbr_metallic_roughness()
                        .roughness_factor();
            
                    let metallic = (255.0 / metallic_factor) as u8;
                    let roughness = (255.0 / roughness_factor) as u8;

                    Ok(fallback_image(RawFormat::Rg8Unorm, [metallic, roughness]))
                })?;

            textures.extend([albedo_map, specular_map, normal_map].into_iter());
        }

        let mut materials: Vec<_> = self.gltf
            .materials()
            .enumerate()
            .map(|(i, _)| i * 3)
            .map(|base| Material {
                albedo_map: base,
                specular_map: base + 1,
                normal_map: base + 2,
            })
            .collect();

        let instances: Vec<_> = self.gltf
            .nodes()
            .filter_map(|node| {
                node.mesh().map(|mesh| (node, mesh.index()))
            })
            .map(|(node, mesh)| Instance {
                transform: Mat4::from_cols_array_2d(&node.transform().matrix()),
                mesh
            })
            .collect();

        let mut vertex_buffer: Vec<Vertex> = Vec::new();
        let mut index_buffer: Vec<u32> = Vec::new();

        let meshes: Result<Vec<_>> = self.gltf
            .meshes()
            .map(|mesh| {
                let mut primitives = Vec::default();

                for primitive in mesh.primitives() {
                    let material = primitive.material().index().unwrap_or_else(|| {
                        textures.extend([
                            fallback_albedo_map(),
                            fallback_specular_map(),
                            fallback_normal_map(),
                        ]);

                        let index = materials.len();

                        materials.push(Material {
                            albedo_map: index * 3,
                            specular_map: index * 3 + 1,
                            normal_map: index * 3 + 2,
                        });

                        index
                    });

                    let bounding_sphere = {
                        let bounding_box = primitive.bounding_box();

                        let min = Vec3::from(bounding_box.min);
                        let max = Vec3::from(bounding_box.max);

                        let center = min + (max - min) * 0.5;
                        let radius = (center - max).length();

                        BoundingSphere { center, radius }
                    };

                    let vertices = {
                        let positions = primitive.get(&gltf::Semantic::Positions);
                        let texcoords = primitive.get(&gltf::Semantic::TexCoords(0));
                        let normals = primitive.get(&gltf::Semantic::Normals);
                        let tangents = primitive.get(&gltf::Semantic::Tangents);

                        let (Some(positions), Some(texcoords), Some(normals), Some(tangents)) =
                            (positions, texcoords, normals, tangents) else
                        {
                            return Err(anyhow!(
                                "prmitive {} doesn't have both positions, texcoords, normals and tangents",
                                primitive.index(),
                            ));
                        };

                        use gltf::accessor::{DataType, Dimensions};

                        fn check_format(
                            acc: &gltf::Accessor,
                            dt: DataType,
                            d: Dimensions,
                        ) -> Result<()> {
                            if dt != acc.data_type() {
                                return Err(anyhow!( "accessor {} must be a {dt:?}", acc.index()));
                            }

                            if d != acc.dimensions() {
                                return Err(anyhow!( "accessor {} must be a {d:?}", acc.index()));
                            }

                            Ok(())
                        }
                    
                        check_format(&positions, DataType::F32, Dimensions::Vec3)?;
                        check_format(&texcoords, DataType::F32, Dimensions::Vec2)?;
                        check_format(&normals, DataType::F32, Dimensions::Vec3)?;
                        check_format(&tangents, DataType::F32, Dimensions::Vec4)?;

                        let get_accessor_data = |acc: &gltf::Accessor| -> Result<&[u8]> {
                            let Some(view) = acc.view() else {
                                return Err(anyhow!("no view on accessor {}", acc.index()));
                            };
                            Ok(self.get_buffer_data(&view, acc.offset(), Some(acc.count() * acc.size())))
                        };

                        let positions = get_accessor_data(&positions)?;
                        let texcoords = get_accessor_data(&texcoords)?;
                        let normals = get_accessor_data(&normals)?;
                        let tangents = get_accessor_data(&tangents)?;

                        assert_eq!(positions.len() / 3, texcoords.len() / 2);
                        assert_eq!(positions.len() / 3, normals.len() / 3);
                        assert_eq!(positions.len() / 3, tangents.len() / 4);

                        positions
                            .chunks(4 * 3)
                            .zip(normals.chunks(4 * 3))
                            .zip(tangents.chunks(4 * 4))
                            .zip(texcoords.chunks(4 * 2))
                            .map(|(((p, n), t), c)| RawVertex {
                                position: Vec3::new(
                                    f32::from_le_bytes([p[0], p[1], p[2], p[3]]),
                                    f32::from_le_bytes([p[4], p[5], p[6], p[7]]),
                                    f32::from_le_bytes([p[8], p[9], p[10], p[11]]),
                                ),
                                normal: Vec3::new(
                                    f32::from_le_bytes([n[0], n[1], n[2], n[3]]),
                                    f32::from_le_bytes([n[4], n[5], n[6], n[7]]),
                                    f32::from_le_bytes([n[8], n[9], n[10], n[11]]),
                                ),
                                texcoord: Vec2::new(
                                    f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                                    f32::from_le_bytes([c[4], c[5], c[6], c[7]]),
                                ),
                                tangent: Vec4::new(
                                    f32::from_le_bytes([t[0], t[1], t[2], t[3]]),
                                    f32::from_le_bytes([t[4], t[5], t[6], t[7]]),
                                    f32::from_le_bytes([t[8], t[9], t[10], t[11]]),
                                    f32::from_le_bytes([t[12], t[13], t[14], t[15]]),
                                ),
                            })
                            .collect::<Vec<_>>()
                    };

                    let indices = {
                        let Some(accessor) = primitive.indices() else {
                            return Err(anyhow!("primtive {} has no indices", primitive.index()));
                        };

                        use gltf::accessor::{DataType, Dimensions};
                        let Dimensions::Scalar = accessor.dimensions() else {
                            return Err(anyhow!(
                                "index attribute, accessor {}, must be scalar", accessor.index(),
                            ));
                        };

                        let Some(view) = accessor.view() else {
                            return Err(anyhow!("no view on accessor {}", accessor.index()));
                        };


                        let offset = accessor.offset();
                        let size = accessor.count() * accessor.size();

                        let index_slice = self.get_buffer_data(&view, offset, Some(size));
                        let mut indices: Vec<u32> = Vec::with_capacity(index_slice.len() / 4);

                        match accessor.data_type() {
                            DataType::U32 => {
                                indices.extend(index_slice
                                    .chunks(4)
                                    .map(|bytes| bytemuck::from_bytes(bytes))
                                );
                            }
                            DataType::U16 => {
                                indices.extend(index_slice
                                    .chunks(2)
                                    .map(|bytes| 0_u32
                                        | (bytes[1] as u32) << 8
                                        | (bytes[0] as u32)
                                    )
                                );
                            }
                            format => {
                                return Err(anyhow!("invalid index format {format:?}"));
                            }
                        }

                        indices
                    };

                    let (remap_count, remap_table) =
                        meshopt::remap::generate_vertex_remap(&vertices, Some(&indices));

                    let mut indices = meshopt::remap::remap_index_buffer(
                        Some(&indices),
                        remap_count,
                        &remap_table,
                    );
        
                    let mut vertices = meshopt::remap::remap_vertex_buffer(
                        &vertices,
                        remap_count,
                        &remap_table,
                    );

                    meshopt::optimize::optimize_vertex_cache_in_place(&indices, vertices.len());

                    use meshopt::utilities::VertexDataAdapter;

                    let vertex_adapter = VertexDataAdapter::new(
                        bytemuck::cast_slice(&vertices),
                        mem::size_of::<RawVertex>(),
                        0,
                    )
                    .expect("failed to create vertex data adapter");

                    const OVERDRAW_THRESHOLD: f32 = 1.05;

                    meshopt::optimize::optimize_overdraw_in_place(
                        &indices,
                        &vertex_adapter,
                        OVERDRAW_THRESHOLD,
                    );

                    meshopt::optimize::optimize_vertex_fetch_in_place(
                        &mut indices,
                        &mut vertices,
                    );

                     let index_count = indices.len() as f32;

                    let mut lods = Vec::<asset::Lod>::with_capacity(TARGET_LOD_COUNT);
                    let mut sloppy = false;

                    for i in 0..TARGET_LOD_COUNT {
                        let index_target =
                            (index_count * LOD_SHRINK_FACTOR.powi(i as i32)) as usize;

                        // Should never fail.
                        let vertex_adapter = &VertexDataAdapter::new(
                            bytemuck::cast_slice(&vertices),
                            mem::size_of::<RawVertex>(),
                            0,
                        )
                        .expect("failed to create vertex adapter");

                        let mut indices = if !sloppy {
                            meshopt::simplify::simplify(
                                &indices,
                                &vertex_adapter,
                                index_target,
                                1e-2,
                            )
                        } else {
                            meshopt::simplify::simplify_sloppy(
                                &indices,
                                &vertex_adapter,
                                index_target,
                            )
                        };

                        if i != 0 && indices.len() < 64 {
                            break;
                        }

                        if Some(indices.len()) == lods.last().map(|lod| lod.index_count as usize) {
                            if sloppy {
                                break;
                            } else {
                                sloppy = true;
                                continue;
                            }
                        }
                     
                        let index_start = index_buffer.len() as u32;
                        let index_count = indices.len() as u32;

                        index_buffer.append(&mut indices);
                        
                        lods.push(Lod { index_count, index_start });
                    }

                    let mut vertices: Vec<_> = vertices
                        .iter()
                        .map(|vertex| {
                            let normal = octahedron_encode_normal(vertex.normal);

                            let normal = normal
                                .to_array()
                                .map(|val| {
                                    meshopt::utilities::quantize_half(val)
                                });

                            let texcoord = vertex.texcoord
                                .to_array()
                                .map(|val| {
                                    meshopt::utilities::quantize_half(val)
                                });

                            let position = vertex.position
                                .extend(1.0)
                                .to_array()
                                .map(|val| {
                                    meshopt::utilities::quantize_half(val)
                                });

                            let tangent = vertex.tangent
                                .to_array()
                                .map(|val| {
                                    meshopt::utilities::quantize_half(val)
                                });

                            Vertex { position, normal, texcoord, tangent }
                        })
                        .collect();

                    let vertex_start = vertex_buffer.len() as u32;
                    vertex_buffer.append(&mut vertices);

                    primitives.push(Primitive {
                        bounding_sphere,
                        vertex_start,
                        material,
                        lods,
                    })
                }

                Ok(Mesh { primitives })
            })
            .collect();

        let meshes = meshes?;

        Ok(Scene { vertices: vertex_buffer, indices: index_buffer, meshes, textures, materials, instances })
    }
}

fn load_scene_from_gltf(path: &Path) -> Result<Scene> {
    GltfImporter::new(path)?.load_scene()
}

/// Glyph of the angelcode bitmap format.
///
/// https://www.angelcode.com/products/bmfont/
#[derive(serde::Deserialize)]
struct BmChar {
    #[serde(rename = "char")]
    codepoint: String,

    width: u32,
    height: u32,

    xoffset: i32,
    yoffset: i32,

    xadvance: i32,
    
    x: u32,
    y: u32,
}

/// Font info for the angelcode bitmap format.
///
/// https://www.angelcode.com/products/bmfont/
#[derive(serde::Deserialize)]
struct BmInfo {
    size: u32,
}

/// Font metadata for the angelcode bitmap format.
///
/// https://www.angelcode.com/products/bmfont/
#[derive(serde::Deserialize)]
struct BmFont {
    pages: Vec<String>,
    chars: Vec<BmChar>,
    info: BmInfo,
}

pub fn load_font(metadata: &Path) -> Result<Font> {
    let file = match fs::File::open(metadata) {
        Ok(file) => file,
        Err(err) => {
            return Err(anyhow!("can't read file {metadata:?}: {err}"));
        }
    };

    let reader = io::BufReader::new(file);
    let font: BmFont = serde_json::from_reader(reader)?;

    let Some(atlas_name) = font.pages.iter().next() else {
        return Err(anyhow!("no pages in font"));
    };

    let parent_path = metadata
        .parent()
        .expect("`path` doesn't have a parent directory")
        .to_path_buf();

    let atlas_path: PathBuf = [parent_path.as_path(), &Path::new(&atlas_name)]
        .iter()
        .collect();

    let image = image::open(&atlas_path)?;

    let width = image.width();
    let height = image.height();

    let atlas_dim = Vec2::new(width as f32, height as f32);

    let size = font.info.size as f32;
    let glyphs: Result<Vec<_>> = font.chars
        .iter()
        .map(|c| {
            let Some(codepoint) = c.codepoint.chars().next() else {
                return Err(anyhow!("empty char"));
            };

            let dim = Vec2::new(c.width as f32, c.height as f32);
            let pos = Vec2::new(c.x as f32, c.y as f32);

            let scaled_dim = dim / Vec2::splat(size);
            let scaled_offset = Vec2::new(c.xoffset as f32, c.yoffset as f32) / Vec2::splat(size);

            let dim = dim / atlas_dim;
            let pos = pos / atlas_dim;

            let texcoord_min = Vec2::new(pos.x, pos.y);
            let texcoord_max = texcoord_min + dim;

            Ok(Glyph {
                codepoint,
                texcoord_min,
                texcoord_max,
                scaled_dim,
                scaled_offset,
                advance: c.xadvance as f32 / size,
            })
        })
        .collect();

    let glyphs = glyphs?;
    let atlas = create_image(image, ImageFormat::Raw(RawFormat::R8Unorm), |_| {}, 1);

    Ok(Font { size: font.info.size, atlas, glyphs })
}

fn octahedron_encode_normal(normal: Vec3) -> Vec2 {
    let t = normal.xy() * (1.0 / (normal.x.abs() + normal.y.abs() + normal.z.abs()));

    fn sign_not_zero(v: Vec2) -> Vec2 {
        let x = if v.x >= 0.0 { 1.0 } else { -1.0 };
        let y = if v.y >= 0.0 { 1.0 } else { -1.0 };

        Vec2 { x, y }
    }

    if normal.z <= 0.0 {
        (Vec2::splat(1.0) - t.yx().abs()) * sign_not_zero(t)
    } else {
        t
    }
}

const TARGET_LOD_COUNT: usize = 8;
const LOD_SHRINK_FACTOR: f32 = 0.75;
