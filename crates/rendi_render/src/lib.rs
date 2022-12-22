#![warn(clippy::all)]
#![allow(clippy::zero_prefixed_literal)]

#[macro_use]
extern crate log;

pub mod buffer;
pub mod device;
pub mod image;
pub mod instance;
pub mod mem;
pub mod reflect;
pub mod surface;
pub mod swapchain;

use ash::vk;
use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt;
use std::slice;

pub use crate::device::{
    Device, PhysicalDevice, Queue, QueueFamilyIndex, QueueRequest, QueueRequestKind,
};
pub use instance::{Instance, ValidationLayers};
pub use mem::{MemoryBlock, MemoryLocation, MemorySlice, MemoryType};
use rendi_data_structs::SortedMap;
pub use surface::Surface;

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("out of gpu memory")]
    Oom,
    #[error("trying to map buffer that's already mapped")]
    BufferRemap,
    #[error("missing layer: {0}")]
    MissingLayer(String),
    #[error("missing extension {0}")]
    MissingExt(String),
    #[error("no physical device found")]
    NoPhysicalDeviceFound,
    #[error("unsupported platform")]
    UnsupportedPlatform,
    #[error("invalid memory type. bits: {type_bits:0}, location: {location}")]
    InvalidMemoryType {
        type_bits: u32,
        location: MemoryLocation,
    },
    #[error("invalid image view kind {kind}, for {dimensions} with {layers} layers")]
    InvalidImageViewKind {
        dimensions: ImageDimensions,
        kind: ImageViewKind,
        layers: u32,
    },
    #[error("physical device doens't support surfaces")]
    SurfaceUnsupported,
    #[error("vulkan error: {0}")]
    VulkanFailure(vk::Result),
    #[error("failed to load vulkan functions: {0}")]
    LoadFailure(#[from] ash::LoadingError),
}

impl From<vk::Result> for RenderError {
    fn from(result: vk::Result) -> Self {
        match result {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => RenderError::Oom,
            // No real reason to handle oom here.
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => panic!("allocation failed"),
            result => RenderError::VulkanFailure(result),
        }
    }
}

/// The slot where a [`DescBinding`] is defined.
/// This is both the set and location within the set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindSlot {
    set: DescSetLocation,
    location: DescLocation,
}

impl BindSlot {
    pub fn new(set: DescSetLocation, location: DescLocation) -> Self {
        Self { set, location }
    }

    /// The location of the set with the binding.
    pub fn set(&self) -> DescSetLocation {
        self.set
    }

    /// The location within the set of the binding.
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
    pub fn reads(self) -> bool {
        self.contains(Access::READ)
    }

    /// Returns `true` if the resource is written to.
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

impl From<DescKind> for vk::DescriptorType {
    fn from(kind: DescKind) -> vk::DescriptorType {
        match kind {
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

impl From<BufferKind> for DescKind {
    fn from(kind: BufferKind) -> DescKind {
        DescKind::Buffer(kind)
    }
}

impl From<BufferKind> for vk::DescriptorType {
    fn from(kind: BufferKind) -> vk::DescriptorType {
        match kind {
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

impl From<BufferKind> for vk::BufferUsageFlags {
    fn from(kind: BufferKind) -> vk::BufferUsageFlags {
        match kind {
            BufferKind::Uniform => vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferKind::Storage => vk::BufferUsageFlags::STORAGE_BUFFER,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ImageLayout {
    #[default]
    Undefined = vk::ImageLayout::UNDEFINED.as_raw() as isize,
    General = vk::ImageLayout::GENERAL.as_raw() as isize,
    Attachment = vk::ImageLayout::ATTACHMENT_OPTIMAL.as_raw() as isize,
    ShaderRead = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL.as_raw() as isize,
    TransferRead = vk::ImageLayout::TRANSFER_SRC_OPTIMAL.as_raw() as isize,
    TransferWrite = vk::ImageLayout::TRANSFER_DST_OPTIMAL.as_raw() as isize,
}

impl From<ImageLayout> for vk::ImageLayout {
    fn from(layout: ImageLayout) -> Self {
        Self::from_raw(layout as i32)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ImageDimensions {
    D1 = vk::ImageType::TYPE_1D.as_raw() as isize,
    #[default]
    D2 = vk::ImageType::TYPE_2D.as_raw() as isize,
    D3 = vk::ImageType::TYPE_3D.as_raw() as isize,
}

impl fmt::Display for ImageDimensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ImageDimensions::D1 => "1 dimensions",
            ImageDimensions::D2 => "2 dimensions",
            ImageDimensions::D3 => "3 dimensions",
        };
        write!(f, "{name}")
    }
}

impl From<ImageDimensions> for vk::ImageType {
    fn from(dim: ImageDimensions) -> Self {
        vk::ImageType::from_raw(dim as i32)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SampleCount {
    #[default]
    S1 = vk::SampleCountFlags::TYPE_1.as_raw() as isize,
    S2 = vk::SampleCountFlags::TYPE_2.as_raw() as isize,
    S4 = vk::SampleCountFlags::TYPE_4.as_raw() as isize,
    S8 = vk::SampleCountFlags::TYPE_8.as_raw() as isize,
    S16 = vk::SampleCountFlags::TYPE_16.as_raw() as isize,
    S32 = vk::SampleCountFlags::TYPE_32.as_raw() as isize,
    S64 = vk::SampleCountFlags::TYPE_64.as_raw() as isize,
}

impl From<SampleCount> for vk::SampleCountFlags {
    fn from(sample_count: SampleCount) -> Self {
        vk::SampleCountFlags::from_raw(sample_count as u32)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImageExtent {
    pub width: u32,
    pub height: u32,
    pub depth_or_layers: u32,
}

impl ImageExtent {
    /// Returns the total amount of texels in the image.
    #[must_use]
    pub fn texel_count(&self) -> u32 {
        self.width * self.height * self.depth_or_layers
    }

    /// Returns the aspect ratio of width to height of the image.
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }

    /// Returns the aspect ratio of width to height of the image.
    #[must_use]
    pub fn max_mip_levels(&self, dim: ImageDimensions) -> u32 {
        match dim {
            ImageDimensions::D1 => 1,
            ImageDimensions::D2 => 32 - self.width.max(self.height).leading_zeros(),
            ImageDimensions::D3 => {
                let max_axis = self.width.max(self.height.max(self.depth_or_layers));
                32 - max_axis.leading_zeros()
            }
        }
    }

    #[must_use]
    pub fn mip_level_size(&self, level: u32, dim: ImageDimensions) -> ImageExtent {
        ImageExtent {
            width: 1_u32.max(self.width >> level),
            height: 1_u32.max(self.width >> level),
            depth_or_layers: match dim {
                ImageDimensions::D1 | ImageDimensions::D2 => self.depth_or_layers,
                ImageDimensions::D3 => u32::max(1, self.depth_or_layers >> level),
            },
        }
    }
}

impl From<ImageExtent> for vk::Extent3D {
    fn from(extent: ImageExtent) -> Self {
        vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: extent.depth_or_layers,
        }
    }
}

macro_rules! raw_format {
    ($name:tt) => {
        vk::Format::$name.as_raw() as isize
    };
}

/// Block compression type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BcFormat {
    Bc1AlphaSrgb = raw_format!(BC1_RGBA_SRGB_BLOCK),
    Bc1Srgb = raw_format!(BC1_RGB_SRGB_BLOCK),
    Bc1AlphaUnorm = raw_format!(BC1_RGBA_UNORM_BLOCK),
    Bc1Unorm = raw_format!(BC1_RGB_UNORM_BLOCK),
    Bc2Srgb = raw_format!(BC2_SRGB_BLOCK),
    Bc2Unorm = raw_format!(BC2_UNORM_BLOCK),
    Bc3Srgb = raw_format!(BC3_SRGB_BLOCK),
    Bc3Unorm = raw_format!(BC3_UNORM_BLOCK),
    Bc4Snorm = raw_format!(BC4_SNORM_BLOCK),
    Bc4Unorm = raw_format!(BC4_UNORM_BLOCK),
    Bc5Snorm = raw_format!(BC5_SNORM_BLOCK),
    Bc5Unorm = raw_format!(BC5_UNORM_BLOCK),
    Bc7Srgb = raw_format!(BC7_SRGB_BLOCK),
    Bc7Unorm = raw_format!(BC7_UNORM_BLOCK),
}

impl BcFormat {
    pub fn block_size(&self) -> vk::DeviceSize {
        match self {
            BcFormat::Bc1AlphaSrgb
            | BcFormat::Bc1Srgb
            | BcFormat::Bc1AlphaUnorm
            | BcFormat::Bc1Unorm => 8,
            _ => 16,
        }
    }
}

impl From<BcFormat> for vk::Format {
    fn from(format: BcFormat) -> Self {
        vk::Format::from_raw(format as i32)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RawFormat {
    R8Unorm = raw_format!(R8_UNORM),
    R8Snorm = raw_format!(R8_SNORM),
    R8Uint = raw_format!(R8_UINT),
    R8Sint = raw_format!(R8_SINT),
    R8Srgb = raw_format!(R8_SRGB),
    Rg8Unorm = raw_format!(R8G8_UNORM),
    Rg8Snorm = raw_format!(R8G8_SNORM),
    Rg8Uint = raw_format!(R8G8_UINT),
    Rg8Sint = raw_format!(R8G8_SINT),
    Rg8Srgb = raw_format!(R8G8_SRGB),
    Rgb8Unorm = raw_format!(R8G8B8_UNORM),
    Rgb8Snorm = raw_format!(R8G8B8_SNORM),
    Rgb8Uint = raw_format!(R8G8B8_UINT),
    Rgb8Sint = raw_format!(R8G8B8_SINT),
    Rgb8Srgb = raw_format!(R8G8B8_SRGB),
    Rgba8Unorm = raw_format!(R8G8B8A8_UNORM),
    Rgba8Snorm = raw_format!(R8G8B8A8_SNORM),
    Rgba8Uint = raw_format!(R8G8B8A8_UINT),
    Rgba8Sint = raw_format!(R8G8B8A8_SINT),
    Rgba8Srgb = raw_format!(R8G8B8A8_SRGB),

    R16Unorm = raw_format!(R16_UNORM),
    R16Snorm = raw_format!(R16_SNORM),
    R16Uint = raw_format!(R16_UINT),
    R16Sint = raw_format!(R16_SINT),
    R16Sfloat = raw_format!(R16_SFLOAT),
    Rg16Unorm = raw_format!(R16G16_UNORM),
    Rg16Snorm = raw_format!(R16G16_SNORM),
    Rg16Uint = raw_format!(R16G16_UINT),
    Rg16Sint = raw_format!(R16G16_SINT),
    Rg16Sfloat = raw_format!(R16G16_SFLOAT),
    Rgb16Unorm = raw_format!(R16G16B16_UNORM),
    Rgb16Snorm = raw_format!(R16G16B16_SNORM),
    Rgb16Uint = raw_format!(R16G16B16_UINT),
    Rgb16Sint = raw_format!(R16G16B16_SINT),
    Rgb16Sfloat = raw_format!(R16G16B16_SFLOAT),
    Rgba16Unorm = raw_format!(R16G16B16A16_UNORM),
    Rgba16Snorm = raw_format!(R16G16B16A16_SNORM),
    Rgba16Uint = raw_format!(R16G16B16A16_UINT),
    Rgba16Sint = raw_format!(R16G16B16A16_SINT),
    Rgba16Sfloat = raw_format!(R16G16B16A16_SFLOAT),

    R32Uint = raw_format!(R32_UINT),
    R32Sint = raw_format!(R32_SINT),
    R32Sfloat = raw_format!(R32_SFLOAT),
    Rg32Uint = raw_format!(R32G32_UINT),
    Rg32Sint = raw_format!(R32G32_SINT),
    Rg32Sfloat = raw_format!(R32G32_SFLOAT),
    Rgb32Uint = raw_format!(R32G32B32_UINT),
    Rgb32Sint = raw_format!(R32G32B32_SINT),
    Rgb32Sfloat = raw_format!(R32G32B32_SFLOAT),
    Rgba32Uint = raw_format!(R32G32B32A32_UINT),
    Rgba32Sint = raw_format!(R32G32B32A32_SINT),
    Rgba32Sfloat = raw_format!(R32G32B32A32_SFLOAT),

    R64Uint = raw_format!(R64_UINT),
    R64Sint = raw_format!(R64_SINT),
    R64Sfloat = raw_format!(R64_SFLOAT),
    Rg64Uint = raw_format!(R64G64_UINT),
    Rg64Sint = raw_format!(R64G64_SINT),
    Rg64Sfloat = raw_format!(R64G64_SFLOAT),
    Rgb64Uint = raw_format!(R64G64B64_UINT),
    Rgb64Sint = raw_format!(R64G64B64_SINT),
    Rgb64Sfloat = raw_format!(R64G64B64_SFLOAT),
    Rgba64Uint = raw_format!(R64G64B64A64_UINT),
    Rgba64Sint = raw_format!(R64G64B64A64_SINT),
    Rgba64Sfloat = raw_format!(R64G64B64A64_SFLOAT),
}

impl From<RawFormat> for vk::Format {
    fn from(format: RawFormat) -> Self {
        vk::Format::from_raw(format as i32)
    }
}

impl RawFormat {
    pub fn element_count(&self) -> vk::DeviceSize {
        use RawFormat::*;
        match self {
            R8Unorm | R8Snorm | R8Uint | R8Sint | R8Srgb | R16Unorm | R16Snorm | R16Uint
            | R16Sint | R16Sfloat | R32Uint | R32Sint | R32Sfloat | R64Uint | R64Sint
            | R64Sfloat => 1,
            Rg8Unorm | Rg8Snorm | Rg8Uint | Rg8Sint | Rg8Srgb | Rg16Unorm | Rg16Snorm
            | Rg16Uint | Rg16Sint | Rg16Sfloat | Rg32Uint | Rg32Sint | Rg32Sfloat | Rg64Uint
            | Rg64Sint | Rg64Sfloat => 2,
            Rgb8Unorm | Rgb8Snorm | Rgb8Uint | Rgb8Sint | Rgb8Srgb | Rgb16Unorm | Rgb16Snorm
            | Rgb16Uint | Rgb16Sint | Rgb16Sfloat | Rgb32Uint | Rgb32Sint | Rgb32Sfloat
            | Rgb64Uint | Rgb64Sint | Rgb64Sfloat => 3,
            _ => 4,
        }
    }

    pub fn element_size(&self) -> vk::DeviceSize {
        use RawFormat::*;
        match self {
            R8Unorm | R8Snorm | R8Uint | R8Sint | R8Srgb | Rg8Unorm | Rg8Snorm | Rg8Uint
            | Rg8Sint | Rg8Srgb | Rgb8Unorm | Rgb8Snorm | Rgb8Uint | Rgb8Sint | Rgb8Srgb
            | Rgba8Unorm | Rgba8Snorm | Rgba8Uint | Rgba8Sint | Rgba8Srgb => 1,
            R16Unorm | R16Snorm | R16Uint | R16Sint | R16Sfloat | Rg16Unorm | Rg16Snorm
            | Rg16Uint | Rg16Sint | Rg16Sfloat | Rgb16Unorm | Rgb16Snorm | Rgb16Uint
            | Rgb16Sint | Rgb16Sfloat | Rgba16Unorm | Rgba16Snorm | Rgba16Uint | Rgba16Sint
            | Rgba16Sfloat => 2,
            R32Uint | R32Sint | Rg32Uint | R32Sfloat | Rg32Sint | Rg32Sfloat | Rgb32Uint
            | Rgb32Sint | Rgb32Sfloat | Rgba32Uint | Rgba32Sint | Rgba32Sfloat => 3,
            R64Uint | R64Sint | R64Sfloat | Rg64Uint | Rg64Sint | Rg64Sfloat | Rgb64Uint
            | Rgb64Sint | Rgb64Sfloat | Rgba64Uint | Rgba64Sint | Rgba64Sfloat => 4,
        }
    }

    pub fn texel_size(&self) -> vk::DeviceSize {
        self.element_count() * self.element_size()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ImageFormat {
    Bc(BcFormat),
    Raw(RawFormat),
}

impl ImageFormat {
    pub fn is_compressed(&self) -> bool {
        matches!(self, ImageFormat::Bc(_))
    }
}

impl From<ImageFormat> for vk::Format {
    fn from(format: ImageFormat) -> Self {
        match format {
            ImageFormat::Bc(format) => format.into(),
            ImageFormat::Raw(format) => format.into(),
        }
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum ImageViewKind {
    #[default]
    View,
    Cube,
}

impl fmt::Display for ImageViewKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ImageViewKind::View => "view",
            ImageViewKind::Cube => "cube",
        };
        write!(f, "{name}")
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentMode {
    #[default]
    Fifo = vk::PresentModeKHR::FIFO.as_raw() as isize,
    Immediate = vk::PresentModeKHR::IMMEDIATE.as_raw() as isize,
    Mailbox = vk::PresentModeKHR::MAILBOX.as_raw() as isize,
}

impl From<PresentMode> for vk::PresentModeKHR {
    fn from(mode: PresentMode) -> Self {
        vk::PresentModeKHR::from_raw(mode as i32)
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct SurfaceExtent {
    width: u32,
    height: u32,
}

impl From<SurfaceExtent> for vk::Extent2D {
    fn from(extent: SurfaceExtent) -> Self {
        vk::Extent2D {
            width: extent.width,
            height: extent.height,
        }
    }
}

impl From<vk::Extent2D> for SurfaceExtent {
    fn from(extent: vk::Extent2D) -> Self {
        SurfaceExtent {
            width: extent.width,
            height: extent.height,
        }
    }
}

impl From<SurfaceExtent> for ImageExtent {
    fn from(extent: SurfaceExtent) -> Self {
        ImageExtent {
            width: extent.width,
            height: extent.height,
            depth_or_layers: 1,
        }
    }
}

impl SurfaceExtent {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn pixel_count(&self) -> u32 {
        self.width * self.height
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
    pub fn is_single(&self) -> bool {
        matches!(self, DescCount::Single)
    }

    /// Returns true if the descriptor count is `Bound`.
    pub fn is_bound_array(&self) -> bool {
        matches!(self, DescCount::BoundArray(_))
    }

    /// Returns true if the descriptor count is `Unbound`.
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
    pub fn kind(&self) -> DescKind {
        self.kind
    }

    /// The descriptor count of the binding.
    pub fn count(&self) -> DescCount {
        self.count
    }

    /// Get the access flags of the binding.
    pub fn access(&self) -> Access {
        self.access_flags
    }

    /// Returns the shader stage where descriptor binding is used.
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

impl From<ShaderStage> for vk::PipelineStageFlags2 {
    fn from(stage: ShaderStage) -> vk::PipelineStageFlags2 {
        use vk::PipelineStageFlags2 as Stage;
        match stage {
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
    pub fn bytes(self) -> u32 {
        self.shape.cardinality() * self.kind.bytes()
    }

    /// The primitive kind of the type.
    pub fn kind(self) -> PrimKind {
        self.kind
    }

    /// The shape of the type.
    pub fn shape(self) -> PrimShape {
        self.shape
    }
}

impl From<PrimType> for vk::Format {
    fn from(prim: PrimType) -> vk::Format {
        use {PrimKind::*, PrimShape::*};
        match (prim.kind(), prim.shape()) {
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
    pub fn binding(&self, location: DescLocation) -> Option<&DescBinding> {
        self.binds.get(&location)
    }

    /// Get the maximum binding location in the set.
    ///
    /// Returns `None` if the set doesn't have any bindings.
    pub fn max_bound_location(&self) -> Option<DescLocation> {
        self.binds.last_key_value().map(|(k, _)| *k)
    }

    /// Get the minimum binding location in the set.
    ///
    /// Returns `None` if the set doesn't have any bindings.
    pub fn min_bound_location(&self) -> Option<DescLocation> {
        self.binds.first_key_value().map(|(k, _)| *k)
    }

    /// Returns iterator of all the bindings in the set.
    pub fn bindings(&self) -> impl Iterator<Item = &(DescLocation, DescBinding)> {
        self.binds.iter()
    }

    /// Returns the location of the unbound array binding in the set if there is any.
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
                    .map(|(bind_slot, bind)| (bind_slot.location(), *bind))
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
    pub fn set(&self, location: DescSetLocation) -> Option<&DescSetBindings> {
        self.sets.get(&location)
    }

    /// Get the [`DescBinding`] at `slot`.
    ///
    /// Returns `None` if there is no binding at `slot`.
    pub fn binding(&self, slot: BindSlot) -> Option<&DescBinding> {
        self.sets
            .get(&slot.set())
            .and_then(|set| set.binds.get(&slot.location()))
    }

    /// Get the maximum descriptor set location.
    ///
    /// Returns `None` if there are no bindings.
    pub fn max_bound_set(&self) -> Option<DescSetLocation> {
        self.sets.last_key_value().map(|(k, _)| *k)
    }

    /// Get the minimum descriptor set location.
    ///
    /// Returns `None` if there are no bindings.
    pub fn min_bound_set(&self) -> Option<DescSetLocation> {
        self.sets.first_key_value().map(|(k, _)| *k)
    }

    /// Returns an iterator of all the descriptor sets.
    pub fn sets(&self) -> impl Iterator<Item = &(DescSetLocation, DescSetBindings)> {
        self.sets.iter()
    }

    /// Returns an iterator of all the bindings.
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
    fn new(inputs: Vec<(InputLocation, PrimType)>) -> Self {
        Self {
            inputs: SortedMap::from_unsorted(inputs),
        }
    }

    /// The number of inputs.
    pub fn count(&self) -> u32 {
        self.inputs.len() as u32
    }

    /// Returns true of there are any inputs.
    pub fn has_any(&self) -> bool {
        self.count() != 0
    }

    /// Returns the input at location `location`.
    ///
    /// Returns `None` if there is no input at `location`.
    pub fn get(&self, location: InputLocation) -> Option<PrimType> {
        self.inputs.get(&location).copied()
    }

    /// Iterate over all the inputs smallest to largest.
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
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            z: z.into(),
        }
    }

    /// The x dimension of the group count.
    pub fn x(&self) -> u32 {
        self.x.get()
    }

    /// The y dimension of the group count.
    pub fn y(&self) -> u32 {
        self.y.get()
    }

    /// The z dimension of the group count.
    pub fn z(&self) -> u32 {
        self.z.get()
    }

    /// Change group count.
    pub fn change(&self, x: u32, y: u32, z: u32) {
        self.x.set(x);
        self.y.set(y);
        self.z.set(z);
    }

    /// Get the total group count.
    /// Returns `self.x() * self.y() * self.z()`.
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
