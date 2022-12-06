#![allow(dead_code)]

mod cmd;

use ahash::HashMap;
use ash::vk;
use smallvec::SmallVec;
use thiserror::Error;

use std::cell::Cell;
use std::fmt;
use std::rc::Rc;

use crate::resource::{
    Buffer, ComputePipeline, ComputeProg, DescLayout, DescPool, Image, ImageInfo, ImageKind,
    ImageLayout, ImageView, ImageViewInfo, PipelineLayout, Prog, RasterPipeline, RasterProg,
    ResourcePool, Sampler,
};
use cmd::Cmd;
use rendi_data_structs::SortedMap;
use rendi_res::{Bump, Res};
use rendi_shader::{BindSlot, DescAccess, DescCount, DescKind};

#[derive(Error, Debug, Clone, PartialEq)]
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

    #[error("vulkan error")]
    VulkanError,
}

type BoundStorage<T> = SmallVec<[T; 1]>;

enum BoundResource {
    ImageView(Option<Res<Sampler>>, BoundStorage<Res<ImageView>>),
    Buffer(BoundStorage<Res<Buffer>>),
}

impl BoundResource {
    /// The number of bound resources.
    fn count(&self) -> u32 {
        let count = match self {
            BoundResource::ImageView(_, views) => views.len(),
            BoundResource::Buffer(buffers) => buffers.len(),
        };

        count as u32
    }
}

struct Bound {
    count: DescCount,
    kind: Option<DescKind>,
    resource: BoundResource,
}

impl Bound {
    fn desc_count(&self) -> DescCount {
        self.count
    }

    fn desc_kind(&self) -> Option<DescKind> {
        self.kind
    }
}

#[derive(Debug, Clone, Copy)]
struct ImageAccess {
    input_layout: ImageLayout,
    output_layout: ImageLayout,

    access: vk::AccessFlags2,
    stage: vk::PipelineStageFlags2,
}

impl ImageAccess {
    #[inline]
    fn has_static_layout(&self) -> bool {
        self.input_layout == self.output_layout
    }
}

#[derive(Debug, Clone, Copy)]
struct BufferAccess {
    access: vk::AccessFlags2,
    stage: vk::PipelineStageFlags2,
}

#[derive(Default, Debug)]
pub struct Accesses {
    images: HashMap<Res<Image>, ImageAccess>,
    buffers: HashMap<Res<Buffer>, BufferAccess>,
}

fn is_access_dependecy(src: DescAccess, dst: DescAccess) -> bool {
    if src.writes() && dst.reads() {
        return true;
    }

    if src.reads() && dst.writes() {
        return true;
    }

    if src.writes() && dst.writes() {
        return true;
    }

    false
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

    fn join(&mut self, other: &Self) -> (Vec<cmd::BufferBarrier>, Vec<cmd::ImageBarrier>) {
        let buffer_barriers = other
            .buffers
            .iter()
            .filter_map(|(buffer, dst)| {
                let Some(src) = self.buffers.get_mut(buffer) else {
                    self.insert_buffer_access(buffer.clone(), dst.clone());
                    return None;
                };

                let barrier =
                    is_access_dependecy(src.access.into(), dst.access.into()).then(|| {
                        cmd::BufferBarrier {
                            buffer: buffer.clone(),
                            src_access: src.access,
                            dst_access: dst.access,
                            src_stage: src.stage,
                            dst_stage: dst.stage,
                        }
                    });

                src.stage = dst.stage;
                src.access = if barrier.is_none() {
                    src.access | dst.access
                } else {
                    dst.access
                };

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

                let is_dependency = is_access_dependecy(src.access.into(), dst.access.into());
                let barrier = (src.output_layout != dst.input_layout || is_dependency).then(|| {
                    cmd::ImageBarrier {
                        image: image.clone(),
                        src_layout: src.output_layout,
                        dst_layout: src.input_layout,
                        src_access: src.access,
                        dst_access: dst.access,
                        src_stage: src.stage,
                        dst_stage: dst.stage,
                    }
                });

                src.stage = dst.stage;
                src.access = if barrier.is_none() {
                    src.access | dst.access
                } else {
                    dst.access
                };

                src.output_layout = dst.output_layout;

                barrier
            })
            .collect();

        (buffer_barriers, image_barriers)
    }
}

#[derive(Default)]
pub struct DescBindings {
    bound: SortedMap<BindSlot, Bound>,
    error: Option<PassBuildError>,
}

impl DescBindings {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn bind_resource(&mut self, bind_slot: BindSlot, bound: Bound) {
        if self.bound.insert(bind_slot, bound).is_some() {
            self.error = Some(PassBuildError::DuplicateBinding { slot: bind_slot });
        }
    }

    #[inline]
    #[must_use]
    fn bind_sampled_image_view(
        mut self,
        slot: impl Into<BindSlot>,
        sampler: Res<Sampler>,
        image_view: Res<ImageView>,
    ) -> Self {
        self.bind_resource(
            slot.into(),
            Bound {
                count: DescCount::Single,
                kind: Some(DescKind::SampledImage),
                resource: BoundResource::ImageView(
                    Some(sampler),
                    BoundStorage::from_buf([image_view]),
                ),
            },
        );
        self
    }

    #[inline]
    #[must_use]
    fn bind_image_view(mut self, slot: impl Into<BindSlot>, image_view: Res<ImageView>) -> Self {
        self.bind_resource(
            slot.into(),
            Bound {
                count: DescCount::Single,
                kind: Some(DescKind::StorageImage),
                resource: BoundResource::ImageView(None, BoundStorage::from_buf([image_view])),
            },
        );
        self
    }

    #[inline]
    #[must_use]
    fn bind_buffer(mut self, slot: impl Into<BindSlot>, buffer: Res<Buffer>) -> Self {
        self.bind_resource(
            slot.into(),
            Bound {
                count: DescCount::Single,
                kind: Some(buffer.kind().into()),
                resource: BoundResource::Buffer(BoundStorage::from_buf([buffer])),
            },
        );
        self
    }

    #[inline]
    #[must_use]
    fn bind_sampled_image_views(
        mut self,
        slot: BindSlot,
        sampler: Res<Sampler>,
        image_views: &[Res<ImageView>],
    ) -> Self {
        self.bind_resource(
            slot.into(),
            Bound {
                kind: Some(DescKind::SampledImage),
                count: DescCount::Bound(image_views.len() as u32),
                resource: BoundResource::ImageView(Some(sampler), BoundStorage::from(image_views)),
            },
        );
        self
    }

    #[inline]
    #[must_use]
    fn bind_image_views(
        mut self,
        slot: impl Into<BindSlot>,
        image_views: &[Res<ImageView>],
    ) -> Self {
        self.bind_resource(
            slot.into(),
            Bound {
                kind: Some(DescKind::StorageImage),
                resource: BoundResource::ImageView(None, BoundStorage::from(image_views)),
                count: DescCount::Bound(image_views.len() as u32),
            },
        );
        self
    }

    #[inline]
    #[must_use]
    fn bind_buffers(mut self, slot: impl Into<BindSlot>, buffers: &[Res<Buffer>]) -> Self {
        let slot = slot.into();

        // Check that all buffers are the same kind.
        if !buffers.windows(2).all(|w| w[0].kind() == w[1].kind()) {
            self.error = Some(PassBuildError::MultiBufferKind { slot });
        }

        self.bind_resource(
            slot,
            Bound {
                kind: buffers.first().map(|buffer| buffer.kind().into()),
                resource: BoundResource::Buffer(BoundStorage::from(buffers)),
                count: DescCount::Bound(buffers.len() as u32),
            },
        );
        self
    }
}

fn check_bind_count(
    expected: DescCount,
    is: DescCount,
    slot: BindSlot,
) -> Result<(), PassBuildError> {
    if is == expected || (is.is_bound() && expected.is_unbound()) {
        return Ok(());
    }

    Err(PassBuildError::WrongBindCount { slot, is, expected })
}

fn check_bind_kind(expected: DescKind, is: DescKind, slot: BindSlot) -> Result<(), PassBuildError> {
    if expected != is {
        return Err(PassBuildError::WrongBindKind { expected, slot, is });
    }

    Ok(())
}

fn shader_access_flags(access: DescAccess) -> vk::AccessFlags2 {
    let mut flags = vk::AccessFlags2::empty();
    if access.reads() {
        flags |= vk::AccessFlags2::SHADER_READ
    }
    if access.writes() {
        flags |= vk::AccessFlags2::SHADER_WRITE
    }
    flags
}

fn desc_accesses<T: Prog>(
    bindings: &DescBindings,
    prog: Res<T>,
) -> Result<Accesses, PassBuildError> {
    if let Some(error) = bindings.error.clone() {
        return Err(error);
    }

    let mut accesses = Accesses::default();
    for (bind_slot, bound) in &bindings.bound {
        let Some(bind) = prog.descs().get_bind(*bind_slot) else {
            return Err(PassBuildError::NoShaderSlot(*bind_slot))
        };

        if let Some(bound_kind) = bound.desc_kind() {
            check_bind_kind(bind.kind(), bound_kind, *bind_slot)?;
        }

        check_bind_count(bind.count(), bound.desc_count(), *bind_slot)?;

        match &bound.resource {
            BoundResource::Buffer(buffers) => {
                let bind_access = shader_access_flags(bind.access());

                for buffer in buffers {
                    let access = accesses
                        .get_buffer_access(&buffer)
                        .map_or(bind_access, |buffer_access| {
                            buffer_access.access | bind_access
                        });

                    accesses.insert_buffer_access(
                        buffer.clone(),
                        BufferAccess {
                            stage: bind.stage().into(),
                            access,
                        },
                    );
                }
            }
            BoundResource::ImageView(sampler, views) => {
                let bind_access = shader_access_flags(bind.access());
                let image_layout = sampler
                    .is_some()
                    .then_some(ImageLayout::ShaderRead)
                    .unwrap_or(ImageLayout::General);

                for view in views {
                    let image = view.image();
                    let access = if let Some(access) = accesses.get_image_access(&image) {
                        debug_assert!(
                            access.has_static_layout(),
                            "image must have static layout in compute or raster pass",
                        );

                        if access.input_layout != image_layout {
                            return Err(PassBuildError::MultiImageLayout {
                                first: access.input_layout,
                                second: image_layout,
                            });
                        }

                        access.access | bind_access
                    } else {
                        bind_access
                    };

                    accesses.insert_image_access(
                        image.clone(),
                        ImageAccess {
                            stage: bind.stage().into(),
                            input_layout: image_layout,
                            output_layout: image_layout,
                            access,
                        },
                    );
                }
            }
        }
    }

    Ok(accesses)
}

fn build_desc_sets<T: Prog>(
    pool: &ResourcePool,
    bindings: &DescBindings,
    layout: &PipelineLayout,
    prog: Res<T>,
) -> Result<DescSets, PassBuildError> {
    let max_set = prog.descs().max_set().map_or(0, |s| s + 1);
    let desc_sets = (0..max_set)
        .map(|set_id| {
            let Some(set) = prog.descs().get_set(set_id) else {
                return Ok(None);
            };

            // Create a sorted list of the binding number and bound resource of the set.
            // Check at the same time that every bind slots in the shader has been bound.
            let bounds: Vec<(u32, &Bound)> = set
                .binds()
                .map(|(binding, _)| {
                    let bind_slot = BindSlot::new(set_id, *binding);
                    let Some(resource) = bindings.bound.get(&bind_slot) else {
                        return Err(PassBuildError::MissingBinding(bind_slot));
                    };

                    Ok((*binding, resource))
                })
                .collect::<Result<_, _>>()?;

            // Could collect into a `SortedMap`, but that sorts `bounds` map, which should already
            // be sorted.
            let bounds = SortedMap::from_sorted(bounds);

            // Get the number of the unbound array elements.
            let var_len = set.unbound_bind().map_or(0, |num| {
                // It is checked above that every slot in the set has been bound, so this should
                // never fail.
                bounds
                    .get(&num)
                    .map(|bound| bound.resource.count())
                    .expect("should have binding")
            });

            let layout = layout.get_desc_layout(set_id).cloned().unwrap_or_else(|| {
                panic!("pipeline doesn't have desc set {set_id}, which it should")
            });

            #[cfg(test)]
            let (handle, desc_pool) = unsafe {
                (
                    vk::DescriptorSet::null(),
                    Rc::new(DescPool::null(pool.device.clone())),
                )
            };

            #[cfg(not(test))]
            let Ok((handle, desc_pool)) = pool.desc_alloc(var_len, &layout) else {
                return Err(PassBuildError::VulkanError);
            };

            let desc_set = DescSet {
                pool: desc_pool,
                layout,
                handle,
            };

            let desc_write_infos: SmallVec<[DescWriteInfo; 12]> = bounds
                .into_iter()
                .filter_map(|(binding, bound)| {
                    let access = set
                        .get_bind(binding)
                        .expect("should have shader binding")
                        .access();

                    let write = match &bound.resource {
                        BoundResource::ImageView(sampler, views) => {
                            let ty: vk::DescriptorType = bound
                                .desc_kind()
                                .unwrap_or_else(|| {
                                    // Shouldn't get here.

                                    if sampler.is_some() {
                                        DescKind::SampledImage
                                    } else {
                                        DescKind::StorageImage
                                    }
                                })
                                .into();

                            let image_layout = if !access.contains(DescAccess::WRITE) {
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                            } else {
                                vk::ImageLayout::GENERAL
                            };

                            let sampler = sampler
                                .as_ref()
                                .map(|sampler| sampler.handle)
                                .unwrap_or(vk::Sampler::null());

                            let image_writes = views
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
                        BoundResource::Buffer(buffers) => {
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
                                image_writes: DescWrites::from(
                                    [vk::DescriptorImageInfo::default()],
                                ),
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

            #[cfg(not(test))]
            unsafe {
                pool.device.handle.update_descriptor_sets(&writes, &[]);
            }

            Ok(Some(desc_set))
        })
        .collect::<Result<_, PassBuildError>>()?;

    Ok(desc_sets)
}

/// The group count of a compute dispatch.
#[derive(Clone, PartialEq, Eq)]
pub struct GroupCount {
    x: Cell<u32>,
    y: Cell<u32>,
    z: Cell<u32>,
}

impl GroupCount {
    #[inline]
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            z: z.into(),
        }
    }

    #[inline]
    pub fn x(&self) -> u32 {
        self.x.get()
    }

    #[inline]
    pub fn y(&self) -> u32 {
        self.y.get()
    }

    #[inline]
    pub fn z(&self) -> u32 {
        self.z.get()
    }

    /// Change group count.
    #[inline]
    pub fn change(&self, x: u32, y: u32, z: u32) {
        self.x.set(x);
        self.y.set(y);
        self.z.set(z);
    }

    /// Get the total group count.
    /// Just `self.x() * self.y() * self.z()`
    #[inline]
    pub fn total_group_count(&self) -> u32 {
        self.x() * self.y() * self.z()
    }
}

impl From<(u32, u32, u32)> for GroupCount {
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Self::new(x, y, z)
    }
}

impl fmt::Debug for GroupCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("GroupCount")
            .field(&self.x())
            .field(&self.y())
            .field(&self.z())
            .finish()
    }
}

impl Default for GroupCount {
    /// Returns group count with dimensions `(1, 1, 1)`.
    fn default() -> Self {
        Self::new(1, 1, 1)
    }
}

pub struct ComputePassBuilder<'a> {
    pool: &'a ResourcePool,
    pipeline: Res<ComputePipeline>,
    prog: Res<ComputeProg>,
    /// [`Cmd`]'s that binds and handles sync for descriptor sets bound but not yet used.
    descs_bind: Vec<Cmd>,
    cmds: Vec<Cmd>,
    accesses: Accesses,
    error: Option<PassBuildError>,
}

pub struct RasterPassBuilder<'a> {
    pool: &'a ResourcePool,
    pipeline: Res<RasterPipeline>,
    prog: Res<RasterProg>,
}

impl<'a> ComputePassBuilder<'a> {
    #[must_use]
    pub fn new(pool: &'a ResourcePool, pipeline: Res<ComputePipeline>) -> Self {
        Self {
            prog: pipeline.prog(),
            accesses: Accesses::default(),
            descs_bind: Vec::new(),
            cmds: Vec::new(),
            error: None,
            pipeline,
            pool,
        }
    }

    #[must_use]
    pub fn bind_descs(mut self, bindings: &DescBindings) -> Self {
        let layout = self.pipeline.layout();
        let prog = self.pipeline.prog();

        let accesses = match desc_accesses(bindings, prog.clone()) {
            Ok(accesses) => accesses,
            Err(error) => {
                self.error = error.into();
                return self;
            }
        };

        let error = build_desc_sets(self.pool, bindings, &layout, prog.clone())
            .map(|descs| {
                let (buffer_barriers, image_barriers) = self.accesses.join(&accesses);
                self.descs_bind.clear();
                self.descs_bind.extend(
                    buffer_barriers
                        .into_iter()
                        .map(|barrier| Cmd::BufferBarrier(barrier)),
                );
                self.descs_bind.extend(
                    image_barriers
                        .into_iter()
                        .map(|barrier| Cmd::ImageBarrier(barrier)),
                );
                self.descs_bind.push(Cmd::BindDescSets(descs));
            })
            .err();

        self.error = self.error.or(error);
        self
    }

    #[must_use]
    pub fn dispatch(mut self, group_count: Res<GroupCount>) -> Self {
        self.cmds.append(&mut self.descs_bind);
        self.cmds.push(Cmd::Dispatch(group_count));
        self
    }

    #[must_use]
    pub fn build(self) -> Result<ComputePass, PassBuildError> {
        todo!()
    }
}

/*
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
*/

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

impl fmt::Debug for DescSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DescSet")
            .field("layout", &self.layout)
            .finish()
    }
}

type DescSets = Vec<Option<DescSet>>;

pub trait Pass {
    fn accesses(&self) -> Option<&Accesses>;
}

pub struct ComputePass {
    accesses: Accesses,
    desc_sets: DescSets,
    group_count: Res<GroupCount>,
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
    buffers: Vec<cmd::BufferBarrier>,
    images: Vec<cmd::ImageBarrier>,
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::core::{Device, Instance, PhysicalDevice};
    use crate::resource::BufferInfo;
    use crate::test;
    use std::rc::Rc;

    fn create_resource_pool() -> ResourcePool {
        let instance = Rc::new(Instance::new(true).unwrap());
        let physical = PhysicalDevice::select(&instance).unwrap();
        let device = Rc::new(Device::new(instance, physical, &[]).unwrap());

        ResourcePool::new(device)
    }

    fn create_compute_pipeline(pool: &ResourcePool, code: &str) -> Res<ComputePipeline> {
        let code = test::compile_comp_glsl(code).unwrap();
        let shader_module = pool
            .create_shader_module("main", bytemuck::cast_slice(&code))
            .unwrap();
        let prog = pool.create_compute_prog(shader_module).unwrap();

        pool.create_compute_pipeline(prog, &[]).unwrap()
    }

    fn buffer_barriers(cmds: &[Cmd]) -> Vec<cmd::BufferBarrier> {
        cmds.iter()
            .filter_map(|cmd| {
                if let Cmd::BufferBarrier(barrier) = cmd {
                    Some(barrier.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    #[test]
    fn wrong_buffer_kind() {
        let pool = create_resource_pool();
        let pipeline = create_compute_pipeline(
            &pool,
            r#"
                #version 450
                layout (set = 0, binding = 0) uniform Block1 { int val1; };
                void main() {}
            "#,
        );
        let buffer = unsafe {
            pool.create_invalid_buffer(&BufferInfo {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                size: 1,
            })
        };
        let group_count = pool.alloc(GroupCount::default());
        let pass = ComputePassBuilder::new(&pool, pipeline)
            .bind_descs(&DescBindings::new().bind_buffer((0, 0), buffer.clone()))
            .dispatch(group_count.clone());
        assert_eq!(
            pass.error,
            Some(PassBuildError::WrongBindKind {
                slot: (0, 0).into(),
                expected: DescKind::UniformBuffer,
                is: DescKind::StorageBuffer,
            })
        );
    }

    #[test]
    fn desc_array() {
        let pool = create_resource_pool();
        let pipeline = create_compute_pipeline(
            &pool,
            r#"
                #version 450
                #extension GL_EXT_nonuniform_qualifier: require
                layout (set = 0, binding = 0) writeonly uniform image2D a[];
                layout (set = 0, binding = 1) writeonly uniform image2D b[4];
                void main() {}
            "#,
        );
        let images: Vec<_> = (0..10)
            .map(|i| unsafe {
                let image = pool.create_invalid_image(
                    i as vk::DeviceSize,
                    &ImageInfo {
                        kind: ImageKind::Texture,
                        usage: vk::ImageUsageFlags::SAMPLED,
                        format: vk::Format::R8G8_UNORM,
                        aspect_flags: vk::ImageAspectFlags::COLOR,
                        mip_levels: 1,
                        extent: vk::Extent3D {
                            width: 10,
                            height: 10,
                            depth: 1,
                        },
                    },
                );
                pool.create_invalid_image_view(&ImageViewInfo {
                    view_type: vk::ImageViewType::TYPE_2D,
                    mips: 0..1,
                    image,
                })
            })
            .collect();
        let group_count = pool.alloc(GroupCount::default());
        let pass = ComputePassBuilder::new(&pool, pipeline)
            .bind_descs(
                &DescBindings::new()
                    .bind_image_views((0, 0), &images)
                    .bind_image_views((0, 1), &images[0..4]),
            )
            .dispatch(group_count.clone());
        assert_eq!(pass.error, None);
    }

    /// Test sync within a compute pass where a buffer is first read and then written and vice
    /// versa.
    #[test]
    fn compute_pass_buffer_sync() {
        let pool = create_resource_pool();
        let pipeline = create_compute_pipeline(
            &pool,
            r#"
                #version 450
                layout (set = 0, binding = 0) writeonly buffer Block1 { int val1; };
                layout (set = 0, binding = 1) readonly buffer Block2 { int val2; };
                void main() { val1 = val2; }
            "#,
        );
        let b1 = unsafe {
            pool.create_invalid_buffer(&BufferInfo {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                size: 1,
            })
        };
        let b2 = unsafe {
            pool.create_invalid_buffer(&BufferInfo {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                size: 2,
            })
        };
        let group_count = pool.alloc(GroupCount::default());
        let pass = ComputePassBuilder::new(&pool, pipeline)
            .bind_descs(
                &DescBindings::new()
                    .bind_buffer((0, 0), b1.clone())
                    .bind_buffer((0, 1), b2.clone()),
            )
            .dispatch(group_count.clone())
            .bind_descs(
                &DescBindings::new()
                    .bind_buffer((0, 0), b2.clone())
                    .bind_buffer((0, 1), b1.clone()),
            );
        assert_eq!(buffer_barriers(&pass.cmds).len(), 0);
        let pass = pass.dispatch(group_count);
        let barriers = buffer_barriers(&pass.cmds);
        assert_eq!(barriers.len(), 2);
        assert!(barriers.iter().any(|barrier| {
            barrier.buffer == b1
                && barrier.src_access == vk::AccessFlags2::SHADER_WRITE
                && barrier.dst_access == vk::AccessFlags2::SHADER_READ
        }));
        assert!(barriers.iter().any(|barrier| {
            barrier.buffer == b2
                && barrier.src_access == vk::AccessFlags2::SHADER_READ
                && barrier.dst_access == vk::AccessFlags2::SHADER_WRITE
        }));
    }
}
