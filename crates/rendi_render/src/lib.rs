#![warn(clippy::all)]
#![allow(clippy::zero_prefixed_literal)]

pub mod device;
pub mod reflect;

use ash::vk;
use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt;
use std::slice;

use rendi_data_structs::SortedMap;

/// The slot where a [`DescBinding`] is defined.
/// This is both the set and location within the set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindSlot {
    set: DescSetLocation,
    location: DescLocation,
}

impl BindSlot {
    #[must_use]
    pub fn new(set: DescSetLocation, location: DescLocation) -> Self {
        Self { set, location }
    }

    /// The location of the set with the binding.
    #[must_use]
    pub fn set(&self) -> DescSetLocation {
        self.set
    }

    /// The location within the set of the binding.
    #[must_use]
    pub fn location(&self) -> DescLocation {
        self.location
    }
}

impl PartialOrd for BindSlot {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BindSlot {
    fn cmp(&self, other: &Self) -> Ordering {
        let a = (self.set() as u64) << 32 | (self.location() as u64);
        let b = (other.set() as u64) << 32 | (other.location() as u64);

        a.cmp(&b)
    }
}

impl From<(u32, u32)> for BindSlot {
    fn from((set, location): (u32, u32)) -> Self {
        Self { set, location }
    }
}

impl fmt::Display for BindSlot {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "({}, {})", self.set, self.location)
    }
}

bitflags::bitflags! {
    /// The access of a resource.
    pub struct Access: u8 {
        const READ = 0b01;
        const WRITE = 0b10;
        const READ_WRITE = Self::READ.bits | Self::WRITE.bits;
    }
}

impl Access {
    /// Returns `true` if the resource is read.
    #[must_use]
    pub fn reads(self) -> bool {
        self.contains(Access::READ)
    }

    /// Returns `true` if the resource is written to.
    #[must_use]
    pub fn writes(self) -> bool {
        self.contains(Access::WRITE)
    }
}

impl From<vk::AccessFlags2> for Access {
    fn from(flags: vk::AccessFlags2) -> Self {
        const WRITE_ACCESS: u64 = vk::AccessFlags2::TRANSFER_WRITE.as_raw()
            | vk::AccessFlags2::SHADER_WRITE.as_raw()
            | vk::AccessFlags2::HOST_WRITE.as_raw()
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()
            | vk::AccessFlags2::MEMORY_WRITE.as_raw()
            | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR.as_raw();

        const READ_ACCESS: u64 = vk::AccessFlags2::TRANSFER_READ.as_raw()
            | vk::AccessFlags2::SHADER_READ.as_raw()
            | vk::AccessFlags2::HOST_READ.as_raw()
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags2::COLOR_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags2::MEMORY_READ.as_raw()
            | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR.as_raw()
            | vk::AccessFlags2::INDIRECT_COMMAND_READ.as_raw()
            | vk::AccessFlags2::INDEX_READ.as_raw()
            | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ.as_raw()
            | vk::AccessFlags2::UNIFORM_READ.as_raw()
            | vk::AccessFlags2::SHADER_SAMPLED_READ.as_raw();

        let mut access = Access::empty();

        if flags.intersects(vk::AccessFlags2::from_raw(WRITE_ACCESS)) {
            access |= Access::WRITE;
        }

        if flags.intersects(vk::AccessFlags2::from_raw(READ_ACCESS)) {
            access |= Access::READ;
        }

        access
    }
}

/// The kind of the descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescKind {
    Buffer(BufferKind),
    /// A storage image is an image that can be both read and written to from shaders, but not
    /// sampled.
    StorageImage,
    /// A sampled image is a combined image and sampler. It's immutable in shaders, but can be
    /// sampled.
    SampledImage,
}

impl Into<vk::DescriptorType> for DescKind {
    fn into(self) -> vk::DescriptorType {
        match self {
            DescKind::Buffer(BufferKind::Storage) => vk::DescriptorType::STORAGE_BUFFER,
            DescKind::Buffer(BufferKind::Uniform) => vk::DescriptorType::UNIFORM_BUFFER,
            DescKind::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
            DescKind::SampledImage => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        }
    }
}

impl fmt::Display for DescKind {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DescKind::SampledImage => write!(fmt, "sampled image"),
            DescKind::StorageImage => write!(fmt, "storage image"),
            DescKind::Buffer(buffer) => {
                write!(fmt, "{buffer} buffer")
            }
        }
    }
}

/// The kind of a buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BufferKind {
    /// A Storage buffer is slightly slower than uniform buffers, but isn't restricted in size and
    /// shader access.
    Storage,
    /// A uniform buffer is a relatively small but fast buffer. It can usually only be 65 kb and
    /// it's read-only from shaders.
    Uniform,
}

impl fmt::Display for BufferKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            BufferKind::Uniform => "uniform",
            BufferKind::Storage => "storage",
        };
        write!(f, "{name}")
    }
}

impl Into<DescKind> for BufferKind {
    fn into(self) -> DescKind {
        DescKind::Buffer(self)
    }
}

impl Into<vk::DescriptorType> for BufferKind {
    fn into(self) -> vk::DescriptorType {
        match self {
            BufferKind::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
            BufferKind::Storage => vk::DescriptorType::STORAGE_BUFFER,
        }
    }
}

impl From<vk::BufferUsageFlags> for BufferKind {
    fn from(flags: vk::BufferUsageFlags) -> Self {
        if flags.contains(vk::BufferUsageFlags::UNIFORM_BUFFER) {
            BufferKind::Uniform
        } else {
            BufferKind::Storage
        }
    }
}

impl Into<vk::BufferUsageFlags> for BufferKind {
    fn into(self) -> vk::BufferUsageFlags {
        match self {
            BufferKind::Uniform => vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferKind::Storage => vk::BufferUsageFlags::STORAGE_BUFFER,
        }
    }
}

/// This indicates the number of descriptor bound to a slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DescCount {
    /// A single descriptor.
    Single,
    /// A bound array of descriptors.
    BoundArray(u32),
    /// An unbound array of descriptors.
    UnboundArray,
}

impl DescCount {
    /// Returns true if the descriptor count is `Single`.
    #[must_use]
    pub fn is_single(&self) -> bool {
        matches!(self, DescCount::Single)
    }

    /// Returns true if the descriptor count is `Bound`.
    #[must_use]
    pub fn is_bound_array(&self) -> bool {
        matches!(self, DescCount::BoundArray(_))
    }

    /// Returns true if the descriptor count is `Unbound`.
    #[must_use]
    pub fn is_unbound_array(&self) -> bool {
        matches!(self, DescCount::UnboundArray)
    }
}

impl fmt::Display for DescCount {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DescCount::Single => write!(fmt, "single"),
            DescCount::BoundArray(count) => write!(fmt, "array length {count}"),
            DescCount::UnboundArray => write!(fmt, "unbound array"),
        }
    }
}

/// A descriptor binding.
///
/// # GLSL
///
/// ```text
/// layout(set = 0, binding = 0) writeonly buffer B1 { int v1; };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescBinding {
    pub(crate) stage: ShaderStage,
    pub(crate) access_flags: Access,
    pub(crate) kind: DescKind,
    pub(crate) count: DescCount,
}

impl DescBinding {
    /// The kind of the binding.
    #[must_use]
    pub fn kind(&self) -> DescKind {
        self.kind
    }

    /// The descriptor count of the binding.
    #[must_use]
    pub fn count(&self) -> DescCount {
        self.count
    }

    /// Get the access flags of the binding.
    #[must_use]
    pub fn access(&self) -> Access {
        self.access_flags
    }

    /// Returns the shader stage where descriptor binding is used.
    #[must_use]
    pub fn stage(&self) -> ShaderStage {
        self.stage
    }
}

#[cfg(test)]
impl Default for DescBinding {
    fn default() -> Self {
        Self {
            access_flags: Access::READ,
            count: DescCount::Single,
            kind: DescKind::Buffer(BufferKind::Uniform),
            stage: ShaderStage::Compute,
        }
    }
}

/// A push constant range of a shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PushConstRange {
    /// The start offset of the push contant. This is the offset into the struct where the first
    /// variable is stored.
    pub offset: u32,

    /// The size of the range from the offset.
    pub size: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderStage {
    /// Compute shader.
    Compute,
    /// Fragment shader.
    Fragment,
    /// Vertex shader.
    Vertex,
    /// Both vertex and fragment shader.
    Raster,
}

impl Into<vk::PipelineStageFlags2> for ShaderStage {
    fn into(self) -> vk::PipelineStageFlags2 {
        use vk::PipelineStageFlags2 as Stage;

        match self {
            ShaderStage::Compute => Stage::COMPUTE_SHADER,
            ShaderStage::Fragment => Stage::FRAGMENT_SHADER,
            ShaderStage::Vertex => Stage::VERTEX_SHADER,
            ShaderStage::Raster => Stage::FRAGMENT_SHADER | Stage::VERTEX_SHADER,
        }
    }
}

/// The base kind of a shader primitive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimKind {
    /// 32 bit signed integer.
    Int,
    /// 32 bit unsigned integer.
    Uint,
    /// 32 bit floating point.
    Float,
    /// 64 bit floating point.
    Double,
}

impl PrimKind {
    /// The byte size of a single element of the type.
    #[must_use]
    pub fn bytes(self) -> u32 {
        match self {
            PrimKind::Int | PrimKind::Uint | PrimKind::Float => 4,
            PrimKind::Double => 8,
        }
    }
}

/// The shape of an shader primitive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimShape {
    Scalar = 1,
    Vec2 = 2,
    Vec3 = 3,
    Vec4 = 4,
}

impl PrimShape {
    /// Returns the cardinality of the shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use rendi_render::PrimShape;
    /// assert_eq!(PrimShape::Vec3.cardinality(), 3);
    /// ```
    #[must_use]
    pub fn cardinality(self) -> u32 {
        self as u32
    }
}

/// A shader primitive type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimType {
    kind: PrimKind,
    shape: PrimShape,
}

impl PrimType {
    /// The byte size of the type.
    #[must_use]
    pub fn bytes(self) -> u32 {
        self.shape.cardinality() * self.kind.bytes()
    }

    /// The primitive kind of the type.
    #[must_use]
    pub fn kind(self) -> PrimKind {
        self.kind
    }

    /// The shape of the type.
    #[must_use]
    pub fn shape(self) -> PrimShape {
        self.shape
    }
}

impl Into<vk::Format> for PrimType {
    fn into(self) -> vk::Format {
        use {PrimKind::*, PrimShape::*};

        match (self.kind(), self.shape()) {
            (Int, Scalar) => vk::Format::R32_SINT,
            (Int, Vec2) => vk::Format::R32G32_SINT,
            (Int, Vec3) => vk::Format::R32G32B32_SINT,
            (Int, Vec4) => vk::Format::R32G32B32A32_SINT,

            (Uint, Scalar) => vk::Format::R32_UINT,
            (Uint, Vec2) => vk::Format::R32G32_UINT,
            (Uint, Vec3) => vk::Format::R32G32B32_UINT,
            (Uint, Vec4) => vk::Format::R32G32B32A32_UINT,

            (Float, Scalar) => vk::Format::R32_SFLOAT,
            (Float, Vec2) => vk::Format::R32G32_SFLOAT,
            (Float, Vec3) => vk::Format::R32G32B32_SFLOAT,
            (Float, Vec4) => vk::Format::R32G32B32A32_SFLOAT,

            (Double, Scalar) => vk::Format::R64_SFLOAT,
            (Double, Vec2) => vk::Format::R64G64_SFLOAT,
            (Double, Vec3) => vk::Format::R64G64B64_SFLOAT,
            (Double, Vec4) => vk::Format::R64G64B64A64_SFLOAT,
        }
    }
}

/// The location of a shader binding in a descriptor set.
pub type DescLocation = u32;

/// The shader bindings of a descriptor set.
///
/// # GLSL
///
/// ```text
/// // Bindings 1, 2 and 3.
/// layout(set = 0, binding = 0) buffer B1 { int v1; };
/// layout(set = 0, binding = 1) buffer B2 { int v2; };
/// layout(set = 0, binding = 3) buffer B3 { int v3; };
/// ```
#[derive(Debug)]
pub struct DescSetBindings {
    binds: SortedMap<DescLocation, DescBinding>,
}

impl DescSetBindings {
    /// Get binding at `location`.
    ///
    /// Returns `None` if the set doesn't have a binding at `location`.
    #[must_use]
    pub fn binding(&self, location: DescLocation) -> Option<&DescBinding> {
        self.binds.get(&location)
    }

    /// Get the maximum binding location in the set.
    ///
    /// Returns `None` if the set doesn't have any bindings.
    #[must_use]
    pub fn max_bound_location(&self) -> Option<DescLocation> {
        self.binds.last_key_value().map(|(k, _)| *k)
    }

    /// Get the minimum binding location in the set.
    ///
    /// Returns `None` if the set doesn't have any bindings.
    #[must_use]
    pub fn min_bound_location(&self) -> Option<DescLocation> {
        self.binds.first_key_value().map(|(k, _)| *k)
    }

    /// Returns iterator of all the bindings in the set.
    #[must_use]
    pub fn bindings(&self) -> impl Iterator<Item = &(DescLocation, DescBinding)> {
        self.binds.iter()
    }

    /// Returns the location of the unbound array binding in the set if there is any.
    #[must_use]
    pub fn unbound_bind(&self) -> Option<DescLocation> {
        let (num, bind) = self.binds.last_key_value()?;
        matches!(bind.count(), DescCount::UnboundArray).then_some(*num)
    }
}

/// The location of a descriptor set in a shader.

/// # GLSL
///
/// ```text
/// // Set 0.
/// layout(set = 0, binding = 0) buffer B1 { int v1; };
/// layout(set = 0, binding = 1) buffer B2 { int v2; };
///
/// // Set 1.
/// layout(set = 1, binding = 0) buffer B3 { int v3; };
/// layout(set = 1, binding = 1) buffer B4 { int v4; };
/// ```
pub type DescSetLocation = u32;

/// The descriptor bindings of a shader.
#[derive(Debug)]
pub struct ShaderBindings {
    sets: SortedMap<DescSetLocation, DescSetBindings>,
}

impl ShaderBindings {
    fn new(binds: SortedMap<BindSlot, DescBinding>) -> Self {
        let sets: Vec<_> = binds
            .group_by(|a, b| a.set() == b.set())
            .map(|binds| {
                // It will never give empty slices of bindings, so it should be OK to call `unwrap`.
                let set = binds.first().unwrap().0.set();

                let binds = binds
                    .iter()
                    .map(|(bind_slot, bind)| (bind_slot.location(), bind.clone()))
                    .collect();

                (
                    set,
                    DescSetBindings {
                        binds: SortedMap::from_sorted(binds),
                    },
                )
            })
            .collect();

        Self {
            sets: SortedMap::from_sorted(sets),
        }
    }

    /// Get the [`DescSetBindings`] at `location`.
    ///
    /// Returns `None` if there is no set at `location`.
    #[must_use]
    pub fn set(&self, location: DescSetLocation) -> Option<&DescSetBindings> {
        self.sets.get(&location)
    }

    /// Get the [`DescBinding`] at `slot`.
    ///
    /// Returns `None` if there is no binding at `slot`.
    #[must_use]
    pub fn binding(&self, slot: BindSlot) -> Option<&DescBinding> {
        self.sets
            .get(&slot.set())
            .map(|set| set.binds.get(&slot.location()))
            .flatten()
    }

    /// Get the maximum descriptor set location.
    ///
    /// Returns `None` if there are no bindings.
    #[must_use]
    pub fn max_bound_set(&self) -> Option<DescSetLocation> {
        self.sets.last_key_value().map(|(k, _)| *k)
    }

    /// Get the minimum descriptor set location.
    ///
    /// Returns `None` if there are no bindings.
    #[must_use]
    pub fn min_bound_set(&self) -> Option<DescSetLocation> {
        self.sets.first_key_value().map(|(k, _)| *k)
    }

    /// Returns an iterator of all the descriptor sets.
    #[must_use]
    pub fn sets(&self) -> impl Iterator<Item = &(DescSetLocation, DescSetBindings)> {
        self.sets.iter()
    }

    /// Returns an iterator of all the bindings.
    #[must_use]
    pub fn bindings(&self) -> impl Iterator<Item = (BindSlot, DescBinding)> + '_ {
        self.sets.iter().flat_map(|(set, binds)| {
            binds
                .bindings()
                .map(|(binding, bind)| (BindSlot::new(*set, *binding), *bind))
        })
    }
}

/// Location of a shader input.
pub type InputLocation = u32;

/// The types of the inputs of a shader.
///
/// This is for instance the vertex attributes of a vertex shader.
///
/// # GLSL
///
/// ```text
/// layout (location = 0) in vec3 in_position;
/// layout (location = 1) in vec2 in_texcoord;
/// ```
pub struct ShaderInputs {
    inputs: SortedMap<InputLocation, PrimType>,
}

impl ShaderInputs {
    #[must_use]
    fn new(inputs: Vec<(InputLocation, PrimType)>) -> Self {
        Self {
            inputs: SortedMap::from_unsorted(inputs),
        }
    }

    /// The number of inputs.
    #[must_use]
    pub fn count(&self) -> u32 {
        self.inputs.len() as u32
    }

    /// Returns true of there are any inputs.
    #[must_use]
    pub fn has_any(&self) -> bool {
        self.count() != 0
    }

    /// Returns the input at location `location`.
    ///
    /// Returns `None` if there is no input at `location`.
    #[must_use]
    pub fn get(&self, location: InputLocation) -> Option<PrimType> {
        self.inputs.get(&location).copied()
    }

    /// Iterate over all the inputs smallest to largest.
    #[must_use]
    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }
}

impl<'a> IntoIterator for &'a ShaderInputs {
    type Item = &'a (InputLocation, PrimType);
    type IntoIter = slice::Iter<'a, (InputLocation, PrimType)>;

    fn into_iter(self) -> Self::IntoIter {
        self.inputs.iter()
    }
}

/// The group count of a compute dispatch.
#[derive(Clone, PartialEq, Eq)]
pub struct GroupCount {
    x: Cell<u32>,
    y: Cell<u32>,
    z: Cell<u32>,
}

impl GroupCount {
    #[must_use]
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            z: z.into(),
        }
    }

    /// The x dimension of the group count.
    #[must_use]
    pub fn x(&self) -> u32 {
        self.x.get()
    }

    /// The y dimension of the group count.
    #[must_use]
    pub fn y(&self) -> u32 {
        self.y.get()
    }

    /// The z dimension of the group count.
    #[must_use]
    pub fn z(&self) -> u32 {
        self.z.get()
    }

    /// Change group count.
    #[must_use]
    pub fn change(&self, x: u32, y: u32, z: u32) {
        self.x.set(x);
        self.y.set(y);
        self.z.set(z);
    }

    /// Get the total group count.
    /// Returns `self.x() * self.y() * self.z()`.
    #[must_use]
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
