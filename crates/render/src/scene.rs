use std::mem;

use ash::vk::{self};
use asset::BoundingSphere;
use eyre::Result;
use glam::{Mat4, Quat, Vec3};

use crate::hal;

pub(super) struct Scene {
    pub indices: hal::Buffer,
    pub vertices: hal::Buffer,
    pub meshlets: hal::Buffer,
    pub meshlet_data: hal::Buffer,
    pub materials: hal::Buffer,
    pub meshes: hal::Buffer,
    pub instances: hal::Buffer,
    pub draws: hal::Buffer,
    pub draw_commands: hal::Buffer,
    pub draw_count: hal::Buffer,
    pub textures: Vec<hal::Image>,
    pub texture_sampler: hal::Sampler,
    pub memory: hal::Memory,
    pub node_tree: NodeTree,
    pub blases: Vec<hal::Blas>,
    pub tree_draws: Vec<Draw>,
    pub tlas: hal::Tlas,
    pub total_draw_count: u32,
}

impl Scene {
    pub fn new(device: &hal::Device, scene: &asset::Scene) -> Result<Self> {
        let node_tree = NodeTree::from_instances(scene);
        let tree_draws = node_tree.draws(&scene);
        let total_draw_count = tree_draws.len() as u32;

        let draws = hal::Buffer::new(
            device,
            &hal::BufferRequest {
                size: mem::size_of_val(tree_draws.as_slice()) as u64,
                kind: hal::BufferKind::Indirect,
            },
        )?;
        let draw_commands = hal::Buffer::new(
            device,
            &hal::BufferRequest {
                size: mem::size_of::<DrawCommand>() as u64
                    * total_draw_count as u64,
                kind: hal::BufferKind::Indirect,
            },
        )?;
        let draw_count = hal::Buffer::new(
            device,
            &hal::BufferRequest {
                size: mem::size_of::<u32>() as u64,
                kind: hal::BufferKind::Indirect,
            },
        )?;
        let instances = hal::Buffer::new(
            device,
            &hal::BufferRequest {
                size: node_tree.instance_count as u64
                    * mem::size_of::<Instance>() as u64,
                kind: hal::BufferKind::Storage,
            },
        )?;
        let indices = hal::Buffer::new(
            device,
            &hal::BufferRequest {
                size: mem::size_of_val(scene.indices.as_slice())
                    as vk::DeviceSize,
                kind: hal::BufferKind::Index,
            },
        )?;
        let vertices = hal::Buffer::new(
            device,
            &hal::BufferRequest {
                size: mem::size_of_val(scene.vertices.as_slice())
                    as vk::DeviceSize,
                kind: hal::BufferKind::AccStructInput,
            },
        )?;
        let meshlets = hal::Buffer::new(
            device,
            &hal::BufferRequest {
                size: mem::size_of_val(scene.meshlets.as_slice())
                    as vk::DeviceSize,
                kind: hal::BufferKind::Storage,
            },
        )?;
        let meshlet_data = hal::Buffer::new(
            device,
            &hal::BufferRequest {
                size: mem::size_of_val(scene.meshlet_data.as_slice())
                    as vk::DeviceSize,
                kind: hal::BufferKind::Storage,
            },
        )?;
        let meshes = hal::Buffer::new(
            device,
            &hal::BufferRequest {
                size: mem::size_of_val(scene.meshes.as_slice())
                    as vk::DeviceSize,
                kind: hal::BufferKind::Storage,
            },
        )?;
        let materials = hal::Buffer::new(
            device,
            &hal::BufferRequest {
                size: mem::size_of_val(scene.materials.as_slice())
                    as vk::DeviceSize,
                kind: hal::BufferKind::Storage,
            },
        )?;

        let mut textures: Vec<_> = scene
            .textures
            .iter()
            .map(|texture| {
                let mip_level_count = texture.mips.len() as u32;
                let usage = vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED;
                let request = hal::ImageRequest {
                    format: texture_kind_format(texture.kind),
                    extent: vk::Extent3D {
                        width: texture.width,
                        height: texture.height,
                        depth: 1,
                    },
                    mip_level_count,
                    usage,
                };
                hal::Image::new(device, &request)
            })
            .collect::<Result<_>>()?;

        let blas_meshes = scene.models.iter().flat_map(|model| {
            model
                .mesh_indices
                .iter()
                .map(|mesh_index| &scene.meshes[*mesh_index as usize])
        });

        let blases: Vec<_> = blas_meshes
            .clone()
            .map(|mesh| {
                hal::Blas::new(
                    device,
                    &hal::BlasRequest {
                        vertex_stride: mem::size_of::<asset::Vertex>()
                            as vk::DeviceSize,
                        vertex_format: vk::Format::R16G16B16_SNORM,
                        first_vertex: mesh.vertex_offset,
                        vertex_count: mesh.vertex_count,
                        triangle_count: mesh.lods[0].index_count / 3,
                    },
                )
            })
            .collect::<Result<_>>()?;

        let tlas = hal::Tlas::new(device, tree_draws.len() as u32)?;

        let mut allocator = hal::Allocator::new(device);
        for texture in &textures {
            allocator.alloc_image(texture);
        }

        for blas in &blases {
            allocator.alloc_blas(blas);
        }

        let memory = allocator
            .alloc_tlas(&tlas)
            .alloc_buffer(&instances)
            .alloc_buffer(&draws)
            .alloc_buffer(&draw_commands)
            .alloc_buffer(&draw_count)
            .alloc_buffer(&indices)
            .alloc_buffer(&vertices)
            .alloc_buffer(&meshlets)
            .alloc_buffer(&meshlet_data)
            .alloc_buffer(&meshes)
            .alloc_buffer(&materials)
            .finish(vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

        for (image, texture) in textures.iter_mut().zip(scene.textures.iter()) {
            image.add_view(
                device,
                hal::ImageViewRequest {
                    mip_level_count: texture.mips.len() as u32,
                    base_mip_level: 0,
                },
            )?;
        }

        let texture_writes: Vec<_> = textures
            .iter()
            .zip(scene.textures.iter())
            .map(|(image, texture)| hal::ImageWrite {
                offset: vk::Offset3D::default(),
                extent: image.extent,
                mips: &texture.mips,
                image,
            })
            .collect();

        let buffer_writes = [
            hal::BufferWrite {
                buffer: &indices,
                data: bytemuck::cast_slice(&scene.indices),
            },
            hal::BufferWrite {
                buffer: &vertices,
                data: bytemuck::cast_slice(&scene.vertices),
            },
            hal::BufferWrite {
                buffer: &meshlets,
                data: bytemuck::cast_slice(&scene.meshlets),
            },
            hal::BufferWrite {
                buffer: &meshlet_data,
                data: bytemuck::cast_slice(&scene.meshlet_data),
            },
            hal::BufferWrite {
                buffer: &meshes,
                data: bytemuck::cast_slice(&scene.meshes),
            },
            hal::BufferWrite {
                buffer: &materials,
                data: bytemuck::cast_slice(&scene.materials),
            },
            hal::BufferWrite {
                buffer: &draws,
                data: bytemuck::cast_slice(&tree_draws),
            },
        ];

        let scratch = hal::command::quickie(device, |command_buffer| {
            command_buffer.ensure_image_layouts(
                device,
                hal::ImageLayouts {
                    layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src: hal::Access::NONE,
                    dst: hal::Access::ALL,
                },
                textures.iter().map(|texture| texture),
            );
            let buffer_scratch = hal::upload_buffer_data(
                device,
                &command_buffer,
                &buffer_writes,
            )?;
            let image_scratch = hal::upload_image_data(
                device,
                &command_buffer,
                &texture_writes,
            )?;
            Ok([buffer_scratch, image_scratch])
        })?;

        for scratch in scratch {
            scratch.destroy(device);
        }

        let texture_sampler = hal::Sampler::new(
            device,
            &hal::SamplerRequest {
                filter: vk::Filter::LINEAR,
                max_anisotropy: Some(device.limits.max_sampler_anisotropy),
                address_mode: vk::SamplerAddressMode::REPEAT,
                ..Default::default()
            },
        )?;

        let blas_builds: Vec<_> = blases
            .iter()
            .zip(blas_meshes)
            .map(|(blas, mesh)| hal::BlasBuild {
                vertices: hal::BufferRange {
                    buffer: &vertices,
                    offset: bytemuck::offset_of!(asset::Vertex, position)
                        as vk::DeviceSize,
                },
                indices: hal::BufferRange {
                    buffer: &indices,
                    offset: mesh.lods[0].index_offset as vk::DeviceSize
                        * mem::size_of::<u32>() as vk::DeviceSize,
                },
                blas,
            })
            .collect();

        hal::build_blases(device, &blas_builds)?;

        let tlas_instances = tlas_instances(&tree_draws, &blases, &node_tree);
        let tlas_update = tlas.update(
            device,
            vk::BuildAccelerationStructureModeKHR::BUILD,
            &tlas_instances,
        );
        hal::command::quickie(device, |command_buffer| {
            let scratch = hal::upload_buffer_data(
                device,
                command_buffer,
                &[tlas_update.buffer_write(&tlas)],
            )?;
            command_buffer.pipeline_barriers(
                device,
                &[],
                &[hal::BufferBarrier {
                    buffer: &tlas.instances,
                    src: hal::Access::ALL,
                    dst: hal::Access::ALL,
                }],
            );
            tlas_update.update(device, command_buffer, &tlas);
            Ok(scratch)
        })?
        .destroy(device);

        Ok(Self {
            indices,
            vertices,
            meshlets,
            meshlet_data,
            materials,
            meshes,
            textures,
            texture_sampler,
            memory,
            node_tree,
            instances,
            draws,
            draw_commands,
            draw_count,
            total_draw_count,
            tlas,
            tree_draws,
            blases,
        })
    }

    pub fn update_tlas(&self, device: &hal::Device) -> Result<()> {
        let tlas_instances =
            tlas_instances(&self.tree_draws, &self.blases, &self.node_tree);
        let tlas_update = self.tlas.update(
            device,
            vk::BuildAccelerationStructureModeKHR::UPDATE,
            &tlas_instances,
        );
        hal::command::quickie(device, |command_buffer| {
            let scratch = hal::upload_buffer_data(
                device,
                command_buffer,
                &[tlas_update.buffer_write(&self.tlas)],
            )?;
            command_buffer.pipeline_barriers(
                device,
                &[],
                &[hal::BufferBarrier {
                    buffer: &self.tlas.instances,
                    src: hal::Access::ALL,
                    dst: hal::Access::ALL,
                }],
            );
            tlas_update.update(device, command_buffer, &self.tlas);
            Ok(scratch)
        })?
        .destroy(device);
        Ok(())
    }

    pub fn destroy(&self, device: &hal::Device) {
        self.instances.destroy(device);
        self.draws.destroy(device);
        self.draw_commands.destroy(device);
        self.draw_count.destroy(device);
        self.indices.destroy(device);
        self.vertices.destroy(device);
        self.meshlets.destroy(device);
        self.meshlet_data.destroy(device);
        self.materials.destroy(device);
        self.meshes.destroy(device);
        for texture in &self.textures {
            texture.destroy(device);
        }
        self.texture_sampler.destroy(device);
        for blas in &self.blases {
            blas.destroy(device);
        }
        self.tlas.destroy(device);
        self.memory.free(device);
    }

    pub(super) fn update(&self) -> SceneUpdate {
        SceneUpdate { instances: self.node_tree.instances() }
    }
}

fn tlas_instances<'a>(
    draws: &[Draw],
    blases: &'a [hal::Blas],
    node_tree: &NodeTree,
) -> Vec<hal::TlasInstance<'a>> {
    let instance_transform = node_tree.instances();
    draws
        .iter()
        .map(|draw| {
            let blas = &blases[draw.mesh_index as usize];
            let something = Mat4::from_scale_rotation_translation(
                Vec3::splat(draw.radius),
                Quat::IDENTITY,
                draw.center,
            );
            let transform = instance_transform[draw.instance_index as usize]
                .transform
                * something;
            hal::TlasInstance { transform, blas }
        })
        .collect()
}

#[repr(C)]
#[repr(align(16))]
#[derive(Debug, Clone, Copy, bytemuck::NoUninit)]
struct Instance {
    transform: Mat4,
    normal_transform: Mat4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Draw {
    center: Vec3,
    radius: f32,
    mesh_index: u32,
    instance_index: u32,
    visible: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct DrawCommand {
    command: vk::DrawIndexedIndirectCommand,
    center: Vec3,
    radius: f32,
}

unsafe impl bytemuck::NoUninit for Draw {}

#[derive(Debug)]
struct NodeDraw {
    instance_index: u32,
    model_index: u32,
}

#[derive(Debug)]
pub struct Node {
    pub transform: Mat4,
    parent: Option<u32>,
    sub_tree_end: u32,
    draw: Option<NodeDraw>,
}

impl Node {
    // Returns the index of the parent node, `None` if it's a root node.
    pub fn parent(&self) -> Option<usize> {
        self.parent.map(|parent| parent as usize)
    }

    // Returns the index of the next node not in this nodes subgraph.
    pub fn sub_tree_end(&self) -> usize {
        self.sub_tree_end as usize
    }
}

#[derive(Debug, Default)]
pub struct NodeTree {
    nodes: Vec<Node>,
    instance_count: u32,
}

impl NodeTree {
    fn from_instances(scene: &asset::Scene) -> Self {
        scene.instances.iter().fold(Self::default(), |tree, instance| {
            build_node_tree(instance, &scene.meshes, tree, None)
        })
    }

    pub fn nodes_mut(&mut self) -> &mut [Node] {
        &mut self.nodes
    }

    fn instances(&self) -> Vec<Instance> {
        let mut global_transforms: Vec<Mat4> =
            Vec::with_capacity(self.nodes.len());
        self.nodes
            .iter()
            .filter_map(|node| {
                let transform = if let Some(parent) = node.parent {
                    global_transforms[parent as usize] * node.transform
                } else {
                    node.transform
                };

                global_transforms.push(transform);
                node.draw.is_some().then_some(Instance {
                    normal_transform: transform.inverse().transpose(),
                    transform,
                })
            })
            .collect()
    }

    fn draws(&self, scene: &asset::Scene) -> Vec<Draw> {
        self.nodes.iter().flat_map(|node| node_draws(node, scene)).collect()
    }
}

fn node_draws<'a>(
    node: &'a Node,
    scene: &'a asset::Scene,
) -> impl Iterator<Item = Draw> + 'a {
    node.draw.iter().flat_map(|draw| {
        model_draws(
            &scene.models[draw.model_index as usize],
            scene,
            draw.instance_index,
        )
    })
}

fn model_draws<'a>(
    model: &'a asset::Model,
    scene: &'a asset::Scene,
    instance_index: u32,
) -> impl Iterator<Item = Draw> + 'a {
    model.mesh_indices.iter().copied().map(move |mesh_index| {
        let mesh = &scene.meshes[mesh_index as usize];
        let BoundingSphere { center, radius } = mesh.bounding_sphere;
        Draw { center, radius, mesh_index, instance_index, visible: 1 }
    })
}

fn build_node_tree(
    instance: &asset::Instance,
    meshes: &[asset::Mesh],
    mut tree: NodeTree,
    parent: Option<u32>,
) -> NodeTree {
    let transform = instance.transform.into();
    let draw = instance.model_index.map(|model_index| {
        let draw =
            NodeDraw { instance_index: tree.instance_count, model_index };
        tree.instance_count += 1;
        draw
    });

    let node_index = tree.nodes.len() as u32;
    let node = Node { transform, parent, draw, sub_tree_end: node_index + 1 };
    tree.nodes.push(node);

    let mut tree = instance.children.iter().fold(tree, |tree, instance| {
        build_node_tree(instance, meshes, tree, Some(node_index))
    });

    let subtree_end = tree.nodes.len() as u32;
    tree.nodes[node_index as usize].sub_tree_end = subtree_end;
    tree
}

pub(super) struct SceneUpdate {
    instances: Vec<Instance>,
}

pub(super) fn buffer_writes<'a>(
    scene: &'a Scene,
    update: &'a SceneUpdate,
) -> impl Iterator<Item = hal::BufferWrite<'a>> {
    [hal::BufferWrite {
        data: bytemuck::cast_slice(&update.instances),
        buffer: &scene.instances,
    }]
    .into_iter()
}

fn texture_kind_format(kind: asset::TextureKind) -> vk::Format {
    match kind {
        asset::TextureKind::Albedo => vk::Format::BC1_RGBA_SRGB_BLOCK,
        asset::TextureKind::Normal => vk::Format::BC5_UNORM_BLOCK,
        asset::TextureKind::Specular => vk::Format::BC5_UNORM_BLOCK,
        asset::TextureKind::Emissive => vk::Format::BC1_RGB_SRGB_BLOCK,
    }
}
