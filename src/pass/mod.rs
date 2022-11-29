use ahash::HashMap;
use ash::vk;
use smallvec::SmallVec;
use thiserror::Error;

use std::rc::Rc;

use crate::resource::{
    Buffer, ComputePipeline, ComputeProg, DescLayout, DescPool, Image, ImageLayout, ImageView,
    PipelineLayout, Prog, RasterPipeline, RasterProg, ResourcePool, Sampler,
};
use rendi_data_structs::SortedMap;
use rendi_res::{Bump, Res};
use rendi_shader::{BindSlot, DescAccess, DescBind, DescCount, DescKind, ShaderStage};

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

    #[error("duplicate binding to {slot}")]
    DuplicateBinding { slot: BindSlot },

    #[error("image is bound multiple times to pass with different layouts, {first} and {second}")]
    MultiImageLayout {
        first: ImageLayout,
        second: ImageLayout,
    },

    #[error("multiple barries for same image in same pass")]
    MultiImageBarrier,

    #[error("both storage and uniform buffers bound to {slot}")]
    MultiBufferKind { slot: BindSlot },

    #[error("failed creating pass: {0}")]
    VulkanError(#[from] anyhow::Error),
}

enum BoundRes {
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

impl BoundRes {
    fn variable_length(&self) -> Option<usize> {
        match self {
            BoundRes::ImageViews { image_views, .. } => image_views.len().into(),
            BoundRes::Buffers { buffers, .. } => buffers.len().into(),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    None,
    Shader { stage: ShaderStage },
}

impl From<ShaderStage> for Stage {
    fn from(stage: ShaderStage) -> Self {
        Stage::Shader { stage }
    }
}

#[derive(Clone, Copy)]
struct ImageAccess {
    input_layout: ImageLayout,
    output_layout: ImageLayout,

    access: DescAccess,
    stage: Stage,
}

impl ImageAccess {
    #[inline]
    fn has_static_layout(&self) -> bool {
        self.input_layout == self.output_layout
    }
}

#[derive(Clone, Copy)]
struct BufferAccess {
    access: DescAccess,
    stage: Stage,
}

#[derive(Default)]
pub struct Accesses {
    images: HashMap<Res<Image>, ImageAccess>,
    buffers: HashMap<Res<Buffer>, BufferAccess>,
}

fn is_access_dependecy(src: DescAccess, dst: DescAccess) -> bool {
    match (src, dst) {
        (DescAccess::WRITE, DescAccess::READ)
        | (DescAccess::WRITE, DescAccess::WRITE)
        | (DescAccess::READ, DescAccess::WRITE) => true,
        _ => false,
    }
}

pub struct BufferBarrier {
    pub buffer: Res<Buffer>,

    pub src_access: DescAccess,
    pub dst_access: DescAccess,

    pub src_stage: Stage,
    pub dst_stage: Stage,
}

pub struct ImageBarrier {
    pub image: Res<Image>,

    pub src_access: DescAccess,
    pub dst_access: DescAccess,

    pub src_layout: ImageLayout,
    pub dst_layout: ImageLayout,

    pub src_stage: Stage,
    pub dst_stage: Stage,
}

impl Accesses {
    #[inline]
    #[must_use]
    fn image_accesses(&self) -> &HashMap<Res<Image>, ImageAccess> {
        &self.images
    }

    #[inline]
    #[must_use]
    fn buffer_accesses(&self) -> &HashMap<Res<Buffer>, BufferAccess> {
        &self.buffers
    }

    #[inline]
    #[must_use]
    fn get_buffer_access(&self, buffer: &Res<Buffer>) -> Option<&BufferAccess> {
        self.buffers.get(buffer)
    }

    #[inline]
    #[must_use]
    fn get_image_access(&self, image: &Res<Image>) -> Option<&ImageAccess> {
        self.images.get(image)
    }

    #[inline]
    fn insert_buffer_access(
        &mut self,
        buffer: Res<Buffer>,
        access: BufferAccess,
    ) -> Option<BufferAccess> {
        self.buffers.insert(buffer, access)
    }

    #[inline]
    fn insert_image_access(
        &mut self,
        image: Res<Image>,
        access: ImageAccess,
    ) -> Option<ImageAccess> {
        self.images.insert(image, access)
    }

    fn join(&mut self, other: &Self) -> (Vec<BufferBarrier>, Vec<ImageBarrier>) {
        let buffer_barriers = other
            .buffers
            .iter()
            .filter_map(|(buffer, dst)| {
                let Some(src) = self.buffers.get_mut(buffer) else {
                    self.insert_buffer_access(buffer.clone(), dst.clone());

                    return None;
                };

                let barrier = is_access_dependecy(src.access, dst.access).then(|| BufferBarrier {
                    buffer: buffer.clone(),
                    src_access: src.access,
                    dst_access: dst.access,
                    src_stage: src.stage,
                    dst_stage: dst.stage,
                });

                src.access |= dst.access;
                src.stage = dst.stage;

                barrier
            })
            .collect();

        let image_barriers = other
            .images
            .iter()
            .filter_map(|(image, dst)| {
                let Some(src) = self.images.get_mut(image) else {
                    self.insert_image_access(image.clone(), dst.clone());

                    return None;
                };

                let is_dependency = is_access_dependecy(src.access, dst.access);
                let barrier = (src.output_layout != dst.input_layout || is_dependency).then(|| {
                    ImageBarrier {
                        image: image.clone(),
                        src_layout: src.output_layout,
                        dst_layout: src.input_layout,
                        src_access: src.access,
                        dst_access: dst.access,
                        src_stage: src.stage,
                        dst_stage: dst.stage,
                    }
                });

                src.access |= dst.access;
                src.stage = dst.stage;
                src.output_layout = dst.output_layout;

                barrier
            })
            .collect();

        (buffer_barriers, image_barriers)
    }
}

struct PassDescs<T> {
    prog: Res<T>,
    layout: Res<PipelineLayout>,
    bound: SortedMap<BindSlot, BoundRes>,
    accesses: Accesses,
    error: Option<PassBuildError>,
}

impl<T: Prog> PassDescs<T> {
    fn new(layout: Res<PipelineLayout>, prog: Res<T>) -> Self {
        Self {
            accesses: Accesses::default(),
            bound: SortedMap::new(),
            error: None,
            layout,
            prog,
        }
    }

    fn update_buffer_access(&mut self, buffer: Res<Buffer>, bind: DescBind) {
        let flags = self
            .accesses
            .get_buffer_access(&buffer)
            .map_or(bind.access(), |current| current.access | bind.access());

        let access = BufferAccess {
            stage: bind.stage().into(),
            access: bind.access(),
        };

        self.accesses.insert_buffer_access(buffer, access);
    }

    fn update_image_access(&mut self, image: Res<Image>, bind: DescBind, layout: ImageLayout) {
        let (access, layout) = if let Some(access) = self.accesses.get_image_access(&image) {
            debug_assert!(
                access.has_static_layout(),
                "image must have static layout in compute or raster pass",
            );

            if access.input_layout != layout {
                self.error = Some(PassBuildError::MultiImageLayout {
                    first: access.input_layout,
                    second: layout,
                });
            }

            (access.access | bind.access(), layout)
        } else {
            (bind.access(), layout)
        };

        let access = ImageAccess {
            stage: bind.stage().into(),
            input_layout: layout,
            output_layout: layout,
            access,
        };

        self.accesses.insert_image_access(image, access);
    }

    #[inline]
    fn expect_desc_bind(&mut self, slot: BindSlot) -> Option<DescBind> {
        let binding = self.prog.descs().get_bind(slot).copied();

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
    fn bind_resource(&mut self, bind_slot: BindSlot, resource: BoundRes) {
        if self.bound.insert(bind_slot, resource).is_some() {
            self.error = Some(PassBuildError::DuplicateBinding { slot: bind_slot });
        }
    }

    fn bind_sampled_image_view(
        &mut self,
        slot: BindSlot,
        sampler: Res<Sampler>,
        image_view: Res<ImageView>,
    ) {
        let Some(binding) = self.expect_desc_bind(slot) else {
            return;
        };

        let image = image_view.image().clone();

        self.check_binding_kind(slot, DescKind::SampledImage, binding.kind());
        self.check_binding_count(slot, DescCount::Single, binding.count(), 1);
        self.update_image_access(image, binding, ImageLayout::ShaderRead);

        self.bind_resource(
            slot,
            BoundRes::ImageView {
                sampler: Some(sampler),
                image_view,
            },
        );
    }

    fn bind_image_view(&mut self, slot: BindSlot, image_view: Res<ImageView>) {
        let Some(binding) = self.expect_desc_bind(slot) else {
            return;
        };

        let image = image_view.image().clone();

        self.check_binding_kind(slot, DescKind::StorageImage, binding.kind());
        self.check_binding_count(slot, DescCount::Single, binding.count(), 1);
        self.update_image_access(image, binding, ImageLayout::General);

        self.bind_resource(
            slot,
            BoundRes::ImageView {
                sampler: None,
                image_view,
            },
        );
    }

    fn bind_buffer(&mut self, slot: BindSlot, buffer: Res<Buffer>) {
        let Some(binding) = self.expect_desc_bind(slot) else {
            return;
        };

        self.check_binding_kind(slot, buffer.kind().into(), binding.kind());
        self.check_binding_count(slot, DescCount::Single, binding.count(), 1);
        self.update_buffer_access(buffer.clone(), binding);

        self.bind_resource(slot, BoundRes::Buffer { buffer });
    }

    fn bind_sampled_images(
        &mut self,
        slot: BindSlot,
        sampler: Res<Sampler>,
        image_views: &[Res<ImageView>],
    ) {
        let count = image_views.len() as u32;

        let Some(binding) = self.expect_desc_bind(slot) else {
            return;
        };

        self.check_binding_kind(slot, DescKind::SampledImage, binding.kind());
        self.check_binding_count(slot, DescCount::Unbound, binding.count(), count);

        for view in image_views {
            self.update_image_access(view.image().clone(), binding, ImageLayout::ShaderRead);
        }

        self.bind_resource(
            slot,
            BoundRes::ImageViews {
                sampler: sampler.into(),
                image_views: image_views.to_vec(),
            },
        );
    }

    fn bind_image_views(&mut self, slot: BindSlot, image_views: &[Res<ImageView>]) {
        let count = image_views.len() as u32;

        let Some(binding) = self.expect_desc_bind(slot) else {
            return;
        };

        self.check_binding_kind(slot, DescKind::SampledImage, binding.kind());
        self.check_binding_count(slot, DescCount::Unbound, binding.count(), count);

        for view in image_views {
            self.update_image_access(view.image().clone(), binding, ImageLayout::General);
        }

        self.bind_resource(
            slot,
            BoundRes::ImageViews {
                image_views: image_views.to_vec(),
                sampler: None,
            },
        );
    }

    fn bind_buffers(&mut self, slot: BindSlot, buffers: &[Res<Buffer>]) {
        let count = buffers.len() as u32;

        let Some(binding) = self.expect_desc_bind(slot) else {
            return;
        };

        // Check that all buffers are the same kind.
        if !buffers.windows(2).all(|w| w[0].kind() == w[1].kind()) {
            self.error = Some(PassBuildError::MultiBufferKind { slot });
        }

        self.check_binding_count(slot, DescCount::Bound(count), binding.count(), 0);

        for buffer in buffers {
            self.check_binding_kind(slot, buffer.kind().into(), binding.kind());
            self.update_buffer_access(buffer.clone(), binding);
        }

        self.bind_resource(
            slot,
            BoundRes::Buffers {
                buffers: buffers.to_vec(),
            },
        );
    }

    fn build(self, pool: &ResourcePool) -> Result<(DescSets, Accesses), PassBuildError> {
        if let Some(error) = self.error {
            return Err(error);
        }

        let descs = self.prog.descs();
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

                        let Some(resource) = self.bound.get(&bind_slot) else {
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
                    .layout
                    .get_desc_layout(set_id)
                    .cloned()
                    .unwrap_or_else(|| panic!("pipeline doesn't have desc set {set_id}"));

                let (handle, desc_pool) = pool.desc_alloc(variable_len, &layout)?;

                let desc_set = DescSet {
                    pool: desc_pool,
                    layout,
                    handle,
                };

                let desc_write_infos: SmallVec<[DescWriteInfo; 12]> = resources
                    .into_iter()
                    .filter_map(|(binding, resource)| {
                        let access = set
                            .get_bind(binding)
                            .expect("should have shader binding")
                            .access();

                        fn image_view_desc_type(has_sampler: bool) -> vk::DescriptorType {
                            if has_sampler {
                                vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                            } else {
                                vk::DescriptorType::STORAGE_IMAGE
                            }
                        }

                        fn image_layout(access: DescAccess) -> vk::ImageLayout {
                            if !access.contains(DescAccess::WRITE) {
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                            } else {
                                vk::ImageLayout::GENERAL
                            }
                        }

                        let write = match resource {
                            BoundRes::ImageView {
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
                            BoundRes::Buffer { buffer } => {
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
                            BoundRes::Buffers { buffers } => {
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
                            BoundRes::ImageViews {
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
                    pool.device.handle.update_descriptor_sets(&writes, &[]);
                }

                Ok(Some(desc_set))
            })
            .collect::<Result<_, PassBuildError>>()?;

        Ok((desc_sets, self.accesses))
    }
}

pub struct ComputePassBuilder<'a> {
    pool: &'a ResourcePool,
    pipeline: Res<ComputePipeline>,
    prog: Res<ComputeProg>,
    pass_descs: PassDescs<ComputeProg>,
}

pub struct RasterPassBuilder<'a> {
    pool: &'a ResourcePool,
    pipeline: Res<RasterPipeline>,
    prog: Res<RasterProg>,
    pass_descs: PassDescs<RasterProg>,
}

macro_rules! impl_desc_bind {
    () => {
        #[must_use]
        #[inline(always)]
        pub fn bind_sampled_image_view(
            mut self,
            slot: impl Into<BindSlot>,
            sampler: Res<Sampler>,
            image_view: Res<ImageView>,
        ) -> Self {
            self.pass_descs
                .bind_sampled_image_view(slot.into(), sampler, image_view);

            self
        }

        #[must_use]
        #[inline(always)]
        pub fn bind_image_view(
            mut self,
            slot: impl Into<BindSlot>,
            image_view: Res<ImageView>,
        ) -> Self {
            self.pass_descs.bind_image_view(slot.into(), image_view);

            self
        }

        #[must_use]
        #[inline(always)]
        pub fn bind_buffer(mut self, slot: impl Into<BindSlot>, buffer: Res<Buffer>) -> Self {
            self.pass_descs.bind_buffer(slot.into(), buffer);

            self
        }

        #[must_use]
        #[inline(always)]
        pub fn bind_sampled_images(
            mut self,
            slot: impl Into<BindSlot>,
            sampler: Res<Sampler>,
            image_views: &[Res<ImageView>],
        ) -> Self {
            self.pass_descs
                .bind_sampled_images(slot.into(), sampler, image_views);

            self
        }

        #[must_use]
        #[inline(always)]
        pub fn bind_image_views(
            mut self,
            slot: impl Into<BindSlot>,
            image_views: &[Res<ImageView>],
        ) -> Self {
            self.pass_descs.bind_image_views(slot.into(), image_views);

            self
        }

        #[must_use]
        #[inline(always)]
        pub fn bind_buffers(mut self, slot: impl Into<BindSlot>, buffers: &[Res<Buffer>]) -> Self {
            self.pass_descs.bind_buffers(slot.into(), buffers);

            self
        }
    };
}

impl<'a> ComputePassBuilder<'a> {
    impl_desc_bind!();
}

impl<'a> RasterPassBuilder<'a> {
    impl_desc_bind!();
}

impl<'a> ComputePassBuilder<'a> {
    #[must_use]
    pub fn new(pool: &'a ResourcePool, pipeline: Res<ComputePipeline>) -> Self {
        Self {
            pass_descs: PassDescs::new(pipeline.layout(), pipeline.prog()),
            prog: pipeline.prog(),
            pipeline,
            pool,
        }
    }

    #[must_use]
    pub fn build(self) -> Result<ComputePass, PassBuildError> {
        let (desc_sets, resource_access) = self.pass_descs.build(self.pool)?;

        Ok(ComputePass {
            accesses: resource_access,
            desc_sets,
        })
    }
}

impl<'a> RasterPassBuilder<'a> {
    #[must_use]
    pub fn new(pool: &'a ResourcePool, pipeline: Res<RasterPipeline>) -> Self {
        Self {
            pass_descs: PassDescs::new(pipeline.layout(), pipeline.prog()),
            prog: pipeline.prog(),
            pipeline,
            pool,
        }
    }

    #[must_use]
    pub fn build(self) -> Result<RasterPass, PassBuildError> {
        let (desc_sets, accesses) = self.pass_descs.build(self.pool)?;

        Ok(RasterPass {
            accesses,
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

type DescSets = Vec<Option<DescSet>>;

pub trait Pass {
    fn accesses(&self) -> Option<&Accesses>;
}

pub struct ComputePass {
    accesses: Accesses,
    desc_sets: DescSets,
}

impl Pass for ComputePass {
    fn accesses(&self) -> Option<&Accesses> {
        Some(&self.accesses)
    }
}

pub struct RasterPass {
    accesses: Accesses,
    desc_sets: DescSets,
}

impl Pass for RasterPass {
    fn accesses(&self) -> Option<&Accesses> {
        Some(&self.accesses)
    }
}

pub struct BarrierPass {
    buffers: Vec<BufferBarrier>,
    images: Vec<ImageBarrier>,
}

impl Pass for BarrierPass {
    fn accesses(&self) -> Option<&Accesses> {
        None
    }
}

pub struct BlockPassBuilder {
    bump: Bump,
    passes: Vec<Res<dyn Pass>>,
    accesses: Accesses,
}

impl BlockPassBuilder {
    #[inline]
    fn alloc_pass<T: Pass>(&mut self, pass: T) -> Res<T> {
        self.bump.alloc(pass)
    }

    #[inline]
    #[must_use]
    pub fn add<T: Pass + 'static>(mut self, pass: T) -> Self {
        let pass = self.alloc_pass(pass);
        self.add_shared(pass)
    }

    #[must_use]
    pub fn add_shared<T: Pass + 'static>(mut self, pass: Res<T>) -> Self {
        if let Some(accesses) = pass.accesses() {
            let (buffer_barriers, image_barriers) = self.accesses.join(accesses);
            let pass = self.alloc_pass(BarrierPass {
                buffers: buffer_barriers,
                images: image_barriers,
            });

            self.passes.push(pass);
        }

        self.passes.push(pass);

        self
    }

    #[must_use]
    pub fn build(self) -> Result<BlockPass, PassBuildError> {
        Ok(BlockPass {
            accesses: Accesses::default(),
            passes: self.passes,
        })
    }
}

pub struct BlockPass {
    passes: Vec<Res<dyn Pass>>,
    accesses: Accesses,
}

impl Pass for BlockPass {
    fn accesses(&self) -> Option<&Accesses> {
        Some(&self.accesses)
    }
}
