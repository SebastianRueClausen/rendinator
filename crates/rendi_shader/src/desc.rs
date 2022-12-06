use anyhow::{anyhow, Result};
use ash::vk;

use std::cmp::Ordering;
use std::fmt;

use crate::RwFlags;
use crate::ShaderStage;

/// The slot where a [`DescBind`] is defined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindSlot {
    set: u32,
    binding: u32,
}

impl BindSlot {
    pub fn new(set: u32, binding: u32) -> Self {
        Self { set, binding }
    }

    pub fn set(&self) -> u32 {
        self.set
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }
}

impl PartialOrd for BindSlot {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BindSlot {
    fn cmp(&self, other: &Self) -> Ordering {
        let a = (self.set as u64) << 32 | (self.binding as u64);
        let b = (other.set as u64) << 32 | (other.binding as u64);

        a.cmp(&b)
    }
}

impl From<(u32, u32)> for BindSlot {
    fn from((set, binding): (u32, u32)) -> Self {
        Self { set, binding }
    }
}

impl fmt::Display for BindSlot {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "({}, {})", self.set, self.binding)
    }
}

bitflags::bitflags! {
    pub struct DescAccess: u8 {
        const READ = 0b01;
        const WRITE = 0b10;
        const READ_WRITE = Self::READ.bits | Self::WRITE.bits;
    }
}

impl DescAccess {
    pub fn reads(self) -> bool {
        self.contains(DescAccess::READ)
    }

    pub fn writes(self) -> bool {
        self.contains(DescAccess::WRITE)
    }
}

impl From<vk::AccessFlags2> for DescAccess {
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

        let mut access = DescAccess::empty();

        if flags.intersects(vk::AccessFlags2::from_raw(WRITE_ACCESS)) {
            access |= DescAccess::WRITE;
        }

        if flags.intersects(vk::AccessFlags2::from_raw(READ_ACCESS)) {
            access |= DescAccess::READ;
        }

        access
    }
}

impl DescAccess {
    pub(crate) fn from_rw_flags(flags: RwFlags) -> Result<Self> {
        let access_flags = match (flags.non_readable, flags.non_writeable) {
            (true, false) => DescAccess::WRITE,
            (false, true) => DescAccess::READ,
            (false, false) => DescAccess::READ_WRITE,
            (true, true) => return Err(anyhow!("descriptor both non-readable and non-writeable")),
        };

        Ok(access_flags)
    }
}

/// The kind of the descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescKind {
    UniformBuffer,
    StorageBuffer,
    StorageImage,
    SampledImage,
}

impl Into<vk::DescriptorType> for DescKind {
    fn into(self) -> vk::DescriptorType {
        match self {
            DescKind::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            DescKind::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
            DescKind::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
            DescKind::SampledImage => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        }
    }
}

impl fmt::Display for DescKind {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            fmt,
            "{}",
            match self {
                DescKind::UniformBuffer => "uniform buffer",
                DescKind::StorageBuffer => "storage buffer",
                DescKind::SampledImage => "sampled image",
                DescKind::StorageImage => "storage image",
            }
        )
    }
}

/// This indicates the number of descriptor bound to a slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DescCount {
    /// A single descriptor.
    Single,
    /// A bound array of descriptors.
    Bound(u32),
    /// An unbound array of descriptors.
    Unbound,
}

impl DescCount {
    pub fn is_single(&self) -> bool {
        matches!(self, DescCount::Single)
    }

    pub fn is_bound(&self) -> bool {
        matches!(self, DescCount::Bound(_))
    }

    pub fn is_unbound(&self) -> bool {
        matches!(self, DescCount::Unbound)
    }
}

impl fmt::Display for DescCount {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DescCount::Single => write!(fmt, "single"),
            DescCount::Bound(count) => write!(fmt, "array length {count}"),
            DescCount::Unbound => write!(fmt, "unbound array"),
        }
    }
}

/// A descriptor binding as defined in a shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescBind {
    pub(crate) stage: ShaderStage,
    pub(crate) access_flags: DescAccess,
    pub(crate) kind: DescKind,
    pub(crate) count: DescCount,
}

impl DescBind {
    /// The kind of the binding.
    pub fn kind(&self) -> DescKind {
        self.kind
    }

    /// The descriptor count of the binding.
    pub fn count(&self) -> DescCount {
        self.count
    }

    /// Get the access flags of the binding.
    pub fn access(&self) -> DescAccess {
        self.access_flags
    }

    pub fn stage(&self) -> ShaderStage {
        self.stage
    }
}

#[cfg(test)]
impl Default for DescBind {
    fn default() -> Self {
        Self {
            access_flags: DescAccess::READ,
            count: DescCount::Single,
            kind: DescKind::UniformBuffer,
            stage: ShaderStage::Compute,
        }
    }
}
