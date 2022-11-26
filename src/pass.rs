use ahash::HashMap;
use ash::vk;
use smallvec::SmallVec;
use thiserror::Error;

use std::rc::Rc;

use crate::resource::{
    Buffer, BufferKind, ComputePipeline, ComputeProg, DescLayout, DescPool, Image, ImageLayout,
    ImageView, ResourcePool, Sampler,
};
use rendi_data_structs::SortedMap;
use rendi_res::Res;
use rendi_shader::{AccessFlags, BindSlot, DescBind, DescCount, DescKind};

#[derive(Error, Debug)]
pub enum PassBuildError {
    /// A descriptor bind slot in the shader wasn't bound.
    #[error("missing resource at {0}")]
    MissingBinding(BindSlot),

    /// Trying to bind a resource to a nonexistent bind slot.
    #[error("no shader slot at {0}")]
    NoShaderSlot(BindSlot),

    /// Trying to bind a resource to a bind slot with a different kind.
    #[error("binding at {slot} is {is}, expected {expected}")]
    WrongBindKind {
        slot: BindSlot,
        is: DescKind,
        expected: DescKind,
    },

    /// Trying to bind the wrong number of resources to bind slot.
    #[error("binding at {slot} has a count of {is}, expected {expected}")]
    WrongBindCount {
        slot: BindSlot,
        is: DescCount,
        expected: DescCount,
    },

    #[error("image is bound multiple times to pass with different layouts, {first} and {second}")]
    MultiImageLayout {
        first: ImageLayout,
        second: ImageLayout,
    },

    #[error("both storage and uniform buffers bound to {slot}")]
    MultiBufferKind { slot: BindSlot },

    #[error("failed creating pass: {0}")]
    VulkanError(#[from] anyhow::Error),
}

enum BoundResource {
    ImageView {
        sampler: Option<Res<Sampler>>,
        image_view: Res<ImageView>,
    },
    Buffer {
        buffer: Res<Buffer>,
    },
    ImageViews {
        sampler: Option<Res<Sampler>>,
        image_views: Vec<Res<ImageView>>,
    },
    Buffers {
        buffers: Vec<Res<Buffer>>,
    },
}

impl BoundResource {
    fn variable_length(&self) -> Option<usize> {
        match self {
            BoundResource::ImageViews { image_views, .. } => image_views.len().into(),
            BoundResource::Buffers { buffers, .. } => buffers.len().into(),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct ImageAccess {
    layout: ImageLayout,
    flags: AccessFlags,
}

#[derive(Clone, Copy)]
struct BufferAccess {
    flags: AccessFlags,
}

pub struct ComputePassBuilder<'a> {
    pool: &'a ResourcePool,

    pipeline: Res<ComputePipeline>,
    prog: Res<ComputeProg>,

    bound: Vec<(BindSlot, BoundResource)>,

    image_access: HashMap<Res<Image>, ImageAccess>,
    buffer_access: HashMap<Res<Buffer>, BufferAccess>,

    error: Option<PassBuildError>,
}

impl<'a> ComputePassBuilder<'a> {
    pub fn new(pool: &'a ResourcePool, pipeline: Res<ComputePipeline>) -> Self {
        Self {
            bound: Vec::with_capacity(20),
            image_access: HashMap::default(),
            buffer_access: HashMap::default(),
            prog: pipeline.prog(),
            error: None,
            pipeline,
            pool,
        }
    }

    #[inline]
    fn shader_binding(&mut self, slot: BindSlot) -> Option<DescBind> {
        let binding = self.prog.reflection().descs().get_bind(slot).copied();

        if binding.is_none() {
            self.error = Some(PassBuildError::NoShaderSlot(slot));
        }

        binding
    }

    #[inline]
    fn check_binding_kind(&mut self, slot: BindSlot, is: DescKind, expected: DescKind) {
        if expected != is {
            self.error = Some(PassBuildError::WrongBindKind { expected, slot, is });
        }
    }

    #[inline]
    fn check_binding_count(
        &mut self,
        slot: BindSlot,
        is: DescCount,
        expected: DescCount,
        count: u32,
    ) {
        if is != expected && is != DescCount::Bound(count) {
            self.error = Some(PassBuildError::WrongBindCount { expected, slot, is });
        }
    }

    #[inline]
    fn update_buffer_access(&mut self, buffer: Res<Buffer>, flags: AccessFlags) {
        let flags = self
            .buffer_access
            .get(&buffer)
            .map(|current| current.flags | flags)
            .unwrap_or(flags);

        self.buffer_access.insert(buffer, BufferAccess { flags });
    }

    #[inline]
    fn update_image_access(&mut self, image: Res<Image>, flags: AccessFlags, layout: ImageLayout) {
        let (flags, layout) = if let Some(access) = self.image_access.get(&image) {
            if access.layout != layout {
                self.error = Some(PassBuildError::MultiImageLayout {
                    first: access.layout,
                    second: layout,
                });
            }

            (access.flags | flags, layout)
        } else {
            (flags, layout)
        };

        self.image_access
            .insert(image, ImageAccess { flags, layout });
    }

    pub fn bind_sampled_image_view(
        mut self,
        slot: impl Into<BindSlot>,
        sampler: Res<Sampler>,
        image_view: Res<ImageView>,
    ) -> Self {
        let slot = slot.into();

        let Some(binding) = self.shader_binding(slot) else {
            return self;
        };

        self.check_binding_kind(slot, DescKind::SampledImage, binding.kind());
        self.check_binding_count(slot, DescCount::Single, binding.count(), 1);

        self.update_image_access(
            image_view.image().clone(),
            binding.access_flags(),
            ImageLayout::ShaderRead,
        );

        self.bound.push((
            slot,
            BoundResource::ImageView {
                sampler: Some(sampler),
                image_view,
            },
        ));

        self
    }

    pub fn bind_image_view(
        mut self,
        slot: impl Into<BindSlot>,
        image_view: Res<ImageView>,
    ) -> Self {
        let slot = slot.into();

        let Some(binding) = self.shader_binding(slot) else {
            return self;
        };

        self.check_binding_kind(slot, DescKind::StorageImage, binding.kind());
        self.check_binding_count(slot, DescCount::Single, binding.count(), 1);

        self.update_image_access(
            image_view.image().clone(),
            binding.access_flags(),
            ImageLayout::General,
        );

        self.bound.push((
            slot,
            BoundResource::ImageView {
                sampler: None,
                image_view,
            },
        ));

        self
    }

    pub fn bind_buffer(mut self, slot: impl Into<BindSlot>, buffer: Res<Buffer>) -> Self {
        let slot = slot.into();

        let Some(binding) = self.shader_binding(slot) else {
            return self;
        };

        self.check_binding_kind(slot, buffer.kind().into(), binding.kind());
        self.check_binding_count(slot, DescCount::Single, binding.count(), 1);
        self.update_buffer_access(buffer.clone(), binding.access_flags());

        self.bound.push((slot, BoundResource::Buffer { buffer }));

        self
    }

    pub fn bind_sampled_images(
        mut self,
        slot: impl Into<BindSlot>,
        sampler: Res<Sampler>,
        image_views: &[Res<ImageView>],
    ) -> Self {
        let slot = slot.into();
        let count = image_views.len() as u32;

        let Some(binding) = self.shader_binding(slot) else {
            return self;
        };

        self.check_binding_kind(slot, DescKind::SampledImage, binding.kind());
        self.check_binding_count(slot, DescCount::Unbound, binding.count(), count);

        for view in image_views {
            self.update_image_access(
                view.image().clone(),
                binding.access_flags(),
                ImageLayout::ShaderRead,
            );
        }

        self.bound.push((
            slot,
            BoundResource::ImageViews {
                sampler: sampler.into(),
                image_views: image_views.to_vec(),
            },
        ));

        self
    }

    pub fn bind_image_views(
        mut self,
        slot: impl Into<BindSlot>,
        image_views: &[Res<ImageView>],
    ) -> Self {
        let slot = slot.into();
        let count = image_views.len() as u32;

        let Some(binding) = self.shader_binding(slot) else {
            return self;
        };

        self.check_binding_kind(slot, DescKind::SampledImage, binding.kind());
        self.check_binding_count(slot, DescCount::Unbound, binding.count(), count);

        for view in image_views {
            self.update_image_access(
                view.image().clone(),
                binding.access_flags(),
                ImageLayout::General,
            );
        }

        self.bound.push((
            slot,
            BoundResource::ImageViews {
                image_views: image_views.to_vec(),
                sampler: None,
            },
        ));

        self
    }

    pub fn bind_buffers(mut self, slot: impl Into<BindSlot>, buffers: &[Res<Buffer>]) -> Self {
        let slot = slot.into();
        let count = buffers.len() as u32;

        let Some(binding) = self.shader_binding(slot) else {
            return self;
        };

        // Check that all buffers are the same kind.
        if !buffers.windows(2).all(|w| w[0].kind() == w[1].kind()) {
            self.error = Some(PassBuildError::MultiBufferKind { slot });
        }

        self.check_binding_count(slot, DescCount::Bound(count), binding.count(), 0);

        for buffer in buffers {
            self.check_binding_kind(slot, buffer.kind().into(), binding.kind());
            self.update_buffer_access(buffer.clone(), binding.access_flags());
        }

        self.bound.push((
            slot,
            BoundResource::Buffers {
                buffers: buffers.to_vec(),
            },
        ));

        self
    }

    pub fn build(self) -> Result<ComputePass, PassBuildError> {
        let descs = self.prog.reflection().descs();
        let bound_resources = SortedMap::from_unsorted(self.bound);

        let max_set = descs.max_set().map_or(0, |s| s + 1);

        let desc_sets: Vec<Option<DescSet>> = (0..max_set)
            .map(|set_id| {
                let Some(set) = descs.get_set(set_id) else {
                    return Ok(None);
                };

                let max_bind = set.max_bind().map_or(0, |b| b + 1);

                let resources: Vec<_> = (0..max_bind)
                    .filter_map(|binding| {
                        if set.get_bind(binding).is_none() {
                            return None;
                        }

                        let bind_slot = BindSlot::new(set_id, binding);

                        let Some(resource) = bound_resources.get(&bind_slot) else {
                            return Some(Err(
                                PassBuildError::MissingBinding(bind_slot)
                            ));
                        };

                        Some(Ok((binding, resource)))
                    })
                    .collect::<Result<_, _>>()?;

                let variable_len = if let Some(res) = resources.last() {
                    res.1.variable_length().unwrap_or(0) as u32
                } else {
                    0
                };

                let layout = self
                    .pipeline
                    .layout()
                    .get_desc_layout(set_id)
                    .cloned()
                    .unwrap_or_else(|| panic!("pipeline doesn't have desc set {set_id}"));

                let (handle, pool) = self.pool.desc_alloc(variable_len, &layout)?;

                let desc_set = DescSet {
                    layout,
                    handle,
                    pool,
                };

                let desc_write_infos: SmallVec<[DescWriteInfo; 12]> = resources
                    .into_iter()
                    .filter_map(|(binding, resource)| {
                        let access = set
                            .get_bind(binding)
                            .expect("should have shader binding")
                            .access_flags();

                        fn image_view_desc_type(has_sampler: bool) -> vk::DescriptorType {
                            if has_sampler {
                                vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                            } else {
                                vk::DescriptorType::STORAGE_IMAGE
                            }
                        }

                        fn image_layout(access: AccessFlags) -> vk::ImageLayout {
                            if !access.contains(AccessFlags::WRITE) {
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                            } else {
                                vk::ImageLayout::GENERAL
                            }
                        }

                        let write = match resource {
                            BoundResource::ImageView {
                                sampler,
                                image_view,
                            } => {
                                let ty = image_view_desc_type(sampler.is_some());
                                let image_layout = image_layout(access);

                                let sampler = sampler
                                    .as_ref()
                                    .map(|sampler| sampler.handle)
                                    .unwrap_or(vk::Sampler::null());

                                DescWriteInfo {
                                    binding,
                                    image_writes: DescWrites::from([vk::DescriptorImageInfo {
                                        image_view: image_view.handle,
                                        image_layout,
                                        sampler,
                                    }]),
                                    buffer_writes: DescWrites::from([
                                        vk::DescriptorBufferInfo::default(),
                                    ]),
                                    ty,
                                }
                            }
                            BoundResource::Buffer { buffer } => {
                                let desc_kind: DescKind = buffer.kind().into();

                                DescWriteInfo {
                                    ty: desc_kind.into(),
                                    image_writes: DescWrites::from([
                                        vk::DescriptorImageInfo::default(),
                                    ]),
                                    buffer_writes: DescWrites::from([vk::DescriptorBufferInfo {
                                        buffer: buffer.handle,
                                        range: buffer.size(),
                                        offset: 0,
                                    }]),
                                    binding,
                                }
                            }
                            BoundResource::Buffers { buffers } => {
                                let desc_kind: DescKind =
                                    buffers.first().map(|buffer| buffer.kind())?.into();

                                let buffer_writes = buffers
                                    .iter()
                                    .map(|buffer| vk::DescriptorBufferInfo {
                                        buffer: buffer.handle,
                                        range: buffer.size(),
                                        offset: 0,
                                    })
                                    .collect();

                                DescWriteInfo {
                                    buffer_writes,
                                    binding,
                                    ty: desc_kind.into(),
                                    image_writes: DescWrites::from([
                                        vk::DescriptorImageInfo::default(),
                                    ]),
                                }
                            }
                            BoundResource::ImageViews {
                                image_views,
                                sampler,
                            } => {
                                let ty = image_view_desc_type(sampler.is_some());
                                let image_layout = image_layout(access);

                                let sampler = sampler
                                    .as_ref()
                                    .map(|sampler| sampler.handle)
                                    .unwrap_or(vk::Sampler::null());

                                let image_writes = image_views
                                    .iter()
                                    .map(|image_view| vk::DescriptorImageInfo {
                                        image_view: image_view.handle,
                                        image_layout,
                                        sampler,
                                    })
                                    .collect();

                                DescWriteInfo {
                                    binding,
                                    image_writes,
                                    buffer_writes: DescWrites::from([
                                        vk::DescriptorBufferInfo::default(),
                                    ]),
                                    ty,
                                }
                            }
                        };

                        Some(write)
                    })
                    .collect();

                let writes: SmallVec<[_; 12]> = desc_write_infos
                    .iter()
                    .enumerate()
                    .map(|(binding, info)| {
                        vk::WriteDescriptorSet::builder()
                            .dst_binding(binding as u32)
                            .descriptor_type(info.ty)
                            .buffer_info(&info.buffer_writes)
                            .image_info(&info.image_writes)
                            .dst_set(desc_set.handle)
                            .build()
                    })
                    .collect();

                unsafe {
                    self.pool.device.handle.update_descriptor_sets(&writes, &[]);
                }

                Ok(Some(desc_set))
            })
            .collect::<Result<_, PassBuildError>>()?;

        Ok(ComputePass {
            image_access: self.image_access,
            buffer_access: self.buffer_access,
            desc_sets,
        })
    }
}

type DescWrites<T> = SmallVec<[T; 1]>;

struct DescWriteInfo {
    binding: u32,
    ty: vk::DescriptorType,
    buffer_writes: DescWrites<vk::DescriptorBufferInfo>,
    image_writes: DescWrites<vk::DescriptorImageInfo>,
}

struct DescSet {
    handle: vk::DescriptorSet,

    layout: Res<DescLayout>,
    pool: Rc<DescPool>,
}

pub struct ComputePass {
    image_access: HashMap<Res<Image>, ImageAccess>,
    buffer_access: HashMap<Res<Buffer>, BufferAccess>,
    desc_sets: Vec<Option<DescSet>>,
}
