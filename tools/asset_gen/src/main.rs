#![feature(let_else, iterator_try_collect)]

use anyhow::{anyhow, Result};
use glam::{Vec2, Vec3, Vec4, Mat4};
use intel_tex_2::{bc5, bc7};
use image::imageops::FilterType;

use std::path::{Path, PathBuf};
use std::{fs, io};

use asset::*;

#[derive(Clone, clap::ValueEnum)]
enum AssetKind {
    Font,
    Skybox,
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
        AssetKind::Skybox => {
            let inputs: Vec<&Path> = args.inputs
                .iter()
                .take(6)
                .map(|buf| buf.as_ref())
                .collect();
            let Ok(inputs) = inputs.try_into() else {
                return Err(anyhow!("expected 6 input image files"));
            };
            let output = args.ouput
                .as_ref()
                .map(|path| path.as_path())
                .unwrap_or(&Path::new("out.skybox"));
            let res = load_skybox(inputs)?.store(output);
            if let Err(err) = res {
                return Err(anyhow!("failed to store skybox to {output:?}: {err}"));
            }
        }
    }

    Ok(())
}

fn create_image(mut image: image::DynamicImage, format: ImageFormat, mip_levels: usize) -> Image {
    match format {
        ImageFormat::Bc(bc) => {
            use intel_tex_2::{RgbaSurface, divide_up_by_multiple};

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

                    let block_bytes = bc.block_size();
                    let (width, height, stride) = (
                        mip.width(),
                        mip.height(),
                        mip.width() * 4,
                    );

                    let block_count =
                        divide_up_by_multiple(width * height, block_bytes as u32) as usize;

                    let mut compressed = vec![0x0; block_bytes * block_count];

                    match bc {
                        BcFormat::Bc5Unorm => {
                            // Swizzle.
                            for px in mip.pixels_mut() {
                                px.0[0] = px.0[1];
                                px.0[1] = px.0[2];
                                px.0[2] = px.0[0];
                            } 
                      
                            let data = mip.into_raw();
                            let surface = RgbaSurface { width, height, stride, data: &data };

                            bc5::compress_blocks_into(&surface, &mut compressed);
                        }
                        BcFormat::Bc7Unorm | BcFormat::Bc7Srgb => {
                            let settings = if mip.pixels().any(|px| px.0[3] != u8::MAX) {
                                bc7::alpha_basic_settings()
                            } else {
                                bc7::opaque_basic_settings()
                            };

                            let data = mip.into_raw();
                            let surface = RgbaSurface { width, height, stride, data: &data };

                            bc7::compress_blocks_into(&settings, &surface, &mut compressed);      
                        }
                    }

                    compressed
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
                        RawFormat::Rg8Unorm => mip
                            .into_luma_alpha8()
                            .into_raw(),
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

    fn load_image(
        &self,
        format: ImageFormat,
        source: &gltf::image::Source,
    ) -> Result<Image> {
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

        let mip_levels = (image.width().max(image.height()) as f32).log2().floor() as usize;

        Ok(create_image(image, format, mip_levels))
    }

    fn load_scene(self) -> Result<Scene> {
        let mut textures = Vec::default();

        for mat in self.gltf.materials() {
            let albedo_map = mat
                .pbr_metallic_roughness()
                .base_color_texture()
                .ok_or_else(|| anyhow!("no base color texture on material"))?
                .texture()
                .source()
                .source();
            let normal_map = mat
                .normal_texture()
                .ok_or_else(|| anyhow!("no normal texture in material"))?
                .texture()
                .source()
                .source();
            let specular_map = mat
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
                .ok_or_else(|| anyhow!("no metallic roughness texture in material"))?
                .texture()
                .source()
                .source();

            textures.extend([
                self.load_image(
                    ImageFormat::Bc(BcFormat::Bc7Srgb),
                    &albedo_map,
                )?,
                self.load_image(
                    ImageFormat::Bc(BcFormat::Bc5Unorm),
                    &specular_map,
                )?,
                self.load_image(
                    ImageFormat::Bc(BcFormat::Bc7Unorm),
                    &normal_map,
                )?,
            ].into_iter());
        }

        let materials: Vec<_> = self.gltf
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
            .filter_map(|node| node.mesh().map(|mesh| (node, mesh.index())))
            .map(|(node, mesh)| {
                let transform = Mat4::from_cols_array_2d(&node.transform().matrix());
                Instance { mesh, transform }
            })
            .collect();

        let mut vertices = Vec::<Vertex>::new();
        let mut indices = Vec::<u8>::new();

        let index_format = self.gltf
            .meshes()
            .map(|mesh| mesh.primitives())
            .flatten()
            .filter_map(|prim| prim
                .indices()
                .map(|acc| acc.data_type())
            )
            .any(|ty| ty == gltf::accessor::DataType::U32)
            .then_some(IndexFormat::U32)
            .unwrap_or(IndexFormat::U16);

        let meshes: Result<Vec<_>> = self.gltf
            .meshes()
            .map(|mesh| {
                let mut primitives = Vec::default();

                for primitive in mesh.primitives() {
                    let Some(material) = primitive.material().index() else {
                        return Err(anyhow!("primitive {} doesn't have a material", primitive.index()));
                    };

                    let bounding_sphere = {
                        let bounding_box = primitive.bounding_box();

                        let min = Vec3::from(bounding_box.min);
                        let max = Vec3::from(bounding_box.max);

                        let center = ((min - max) * 0.5) + max;
                        let radius = (center - max).length();

                        BoundingSphere { center, radius }
                    };

                    let vertex_start = {
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

                        let vertex_start = vertices.len() as u32;

                        vertices.extend(positions
                            .chunks(12)
                            .zip(normals.chunks(4 * 3))
                            .zip(tangents.chunks(4 * 4))
                            .zip(texcoords.chunks(4 * 2))
                            .map(|(((p, n), t), c)| Vertex {
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
                        );
                        
                        vertex_start
                    };

                    let (index_start, index_count) = {
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

                        let index_start = (indices.len() / index_format.byte_size()) as u32;
                        let index_count = (index_slice.len() / index_format.byte_size()) as u32;

                        match (index_format, accessor.data_type()) {
                            (IndexFormat::U32, DataType::U32) => {
                                indices.extend_from_slice(index_slice)
                            }
                            (IndexFormat::U16, DataType::U16) => {
                                indices.extend_from_slice(index_slice)
                            }
                            (IndexFormat::U32, DataType::U16) => {
                                for bytes in index_slice.chunks(2) {
                                    indices.extend_from_slice(&[0, 0]);
                                    indices.extend_from_slice(bytes);  
                                }
                            }
                            (_, format) => {
                                return Err(anyhow!("invalid index format {format:?}"));
                            }
                        }

                        (index_start, index_count)
                    };

                    primitives.push(Primitive {
                        bounding_sphere,
                        vertex_start,
                        index_start,
                        index_count,
                        material,
                    })
                }

                Ok(Mesh { primitives })
            })
            .collect();

        let meshes = meshes?;

        Ok(Scene { vertices, indices, meshes, textures, materials, instances, index_format })
    }
}

fn load_scene_from_gltf(path: &Path) -> Result<Scene> {
    GltfImporter::new(path)?.load_scene()
}

fn load_skybox(images: [&Path; 6]) -> Result<Skybox> {
    let images: Result<Vec<_>> = images
        .iter()
        .map(|path| {
            let image = image::open(path)?.into_rgba8();

            let width = image.width() as u32;
            let height = image.height() as u32;

            let data = image.into_raw();
            let format = ImageFormat::Raw(RawFormat::Rgba8Srgb);

            Ok(Image { width, height, mips: vec![data], format, })
        })
        .collect();

    let images = images?.try_into().unwrap();

    Ok(Skybox { images })
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
    let atlas = create_image(image, ImageFormat::Raw(RawFormat::R8Unorm), 1);

    Ok(Font { size: font.info.size, atlas, glyphs })
}
