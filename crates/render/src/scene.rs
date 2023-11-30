use std::mem;

use ash::vk::{self};
use asset::BoundingSphere;
use eyre::Result;
use glam::{Mat4, Vec3};

use crate::command::{self, Access, ImageLayouts};
use crate::device::Device;
use crate::resources::{
    self, Allocator, Buffer, BufferKind, BufferRequest, BufferWrite, Image,
    ImageRequest, ImageViewRequest, ImageWrite, Memory, Sampler,
    SamplerRequest,
};

pub(super) struct Scene {
    pub indices: Buffer,
    pub vertices: Buffer,
    pub meshlets: Buffer,
    pub meshlet_data: Buffer,
    pub materials: Buffer,
    pub meshes: Buffer,
    pub instances: Buffer,
    pub draws: Buffer,
    pub draw_commands: Buffer,
    pub draw_count: Buffer,
    pub textures: Vec<Image>,
    pub texture_sampler: Sampler,
    pub memory: Memory,
    pub node_tree: NodeTree,
    pub total_draw_count: u32,
}

impl Scene {
    pub fn new(device: &Device, scene: &asset::Scene) -> Result<Self> {
        let node_tree = NodeTree::from_instances(scene);
        let tree_draws = node_tree.draws(&scene);
        let total_draw_count = tree_draws.len() as u32;

        let draws = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(tree_draws.as_slice()) as u64,
                kind: BufferKind::Indirect,
            },
        )?;
        let draw_commands = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of::<DrawCommand>() as u64
                    * total_draw_count as u64,
                kind: BufferKind::Indirect,
            },
        )?;
        let draw_count = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of::<u32>() as u64,
                kind: BufferKind::Indirect,
            },
        )?;
        let instances = Buffer::new(
            device,
            &BufferRequest {
                size: node_tree.instance_count as u64
                    * mem::size_of::<Instance>() as u64,
                kind: BufferKind::Storage,
            },
        )?;
        let indices = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(scene.indices.as_slice())
                    as vk::DeviceSize,
                kind: BufferKind::Index,
            },
        )?;
        let vertices = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(scene.vertices.as_slice())
                    as vk::DeviceSize,
                kind: BufferKind::Storage,
            },
        )?;
        let meshlets = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(scene.meshlets.as_slice())
                    as vk::DeviceSize,
                kind: BufferKind::Storage,
            },
        )?;
        let meshlet_data = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(scene.meshlet_data.as_slice())
                    as vk::DeviceSize,
                kind: BufferKind::Storage,
            },
        )?;
        let meshes = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(scene.meshes.as_slice())
                    as vk::DeviceSize,
                kind: BufferKind::Storage,
            },
        )?;
        let materials = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(scene.materials.as_slice())
                    as vk::DeviceSize,
                kind: BufferKind::Storage,
            },
        )?;

        let mut textures: Vec<_> = scene
            .textures
            .iter()
            .map(|texture| {
                let mip_level_count = texture.mips.len() as u32;
                let usage = vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED;
                let request = ImageRequest {
                    format: texture_kind_format(texture.kind),
                    extent: vk::Extent3D {
                        width: texture.width,
                        height: texture.height,
                        depth: 1,
                    },
                    mip_level_count,
                    usage,
                };
                Image::new(device, &request)
            })
            .collect::<Result<_>>()?;

        let mut allocator = Allocator::new(device);
        for texture in &textures {
            allocator.alloc_image(texture);
        }

        let memory = allocator
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
                ImageViewRequest {
                    mip_level_count: texture.mips.len() as u32,
                    base_mip_level: 0,
                },
            )?;
        }

        let texture_writes: Vec<_> = textures
            .iter()
            .zip(scene.textures.iter())
            .map(|(image, texture)| ImageWrite {
                offset: vk::Offset3D::default(),
                extent: image.extent,
                mips: &texture.mips,
                image,
            })
            .collect();

        let buffer_writes = [
            BufferWrite {
                buffer: &indices,
                data: bytemuck::cast_slice(&scene.indices),
            },
            BufferWrite {
                buffer: &vertices,
                data: bytemuck::cast_slice(&scene.vertices),
            },
            BufferWrite {
                buffer: &meshlets,
                data: bytemuck::cast_slice(&scene.meshlets),
            },
            BufferWrite {
                buffer: &meshlet_data,
                data: bytemuck::cast_slice(&scene.meshlet_data),
            },
            BufferWrite {
                buffer: &meshes,
                data: bytemuck::cast_slice(&scene.meshes),
            },
            BufferWrite {
                buffer: &materials,
                data: bytemuck::cast_slice(&scene.materials),
            },
            BufferWrite {
                buffer: &draws,
                data: bytemuck::cast_slice(&tree_draws),
            },
        ];

        let scratch = command::quickie(device, |command_buffer| {
            command_buffer.ensure_image_layouts(
                device,
                ImageLayouts {
                    layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src: Access::NONE,
                    dst: Access::ALL,
                },
                textures.iter().map(|texture| texture),
            );
            let buffer_scratch = resources::upload_buffer_data(
                device,
                &command_buffer,
                &buffer_writes,
            )?;
            let image_scratch = resources::upload_image_data(
                device,
                &command_buffer,
                &texture_writes,
            )?;
            Ok([buffer_scratch, image_scratch])
        })?;

        for scratch in scratch {
            scratch.destroy(device);
        }

        let texture_sampler = Sampler::new(
            device,
            &SamplerRequest {
                filter: vk::Filter::LINEAR,
                max_anisotropy: Some(device.limits.max_sampler_anisotropy),
                address_mode: vk::SamplerAddressMode::REPEAT,
                ..Default::default()
            },
        )?;

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
        })
    }

    pub fn destroy(&self, device: &Device) {
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
        self.memory.free(device);
    }

    pub(super) fn update(&self) -> SceneUpdate {
        SceneUpdate { instances: self.node_tree.instances() }
    }
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
) -> impl Iterator<Item = BufferWrite<'a>> {
    [BufferWrite {
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
