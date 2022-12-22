use crate::{
    mem, Device, ImageDimensions, ImageExtent, ImageFormat, ImageLayout, ImageViewKind,
    MemoryLocation, MemorySlice, Queue, RenderError, SampleCount,
};
use ash::vk::{self, DeviceSize};
use rendi_res::Res;
use smallvec::SmallVec;
use std::cell::Cell;
use std::hash::{Hash, Hasher};

type Queues = SmallVec<[Res<Queue>; 4]>;

pub enum ImageUsage {
    ColorTarget(Res<Queue>),
    DepthTarget(Res<Queue>),
    Sample(Res<Queue>),
    Storage(Res<Queue>),
}

fn parse_usages(usages: &[ImageUsage]) -> (vk::ImageUsageFlags, vk::ImageAspectFlags, Queues) {
    let mut usage_flags = vk::ImageUsageFlags::empty();
    let mut aspect_flags = vk::ImageAspectFlags::empty();
    let mut queues: Queues = usages
        .iter()
        .map(|usage| match usage {
            ImageUsage::ColorTarget(queue) => {
                usage_flags |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
                aspect_flags |= vk::ImageAspectFlags::COLOR;
                queue.clone()
            }
            ImageUsage::DepthTarget(queue) => {
                usage_flags |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
                aspect_flags |= vk::ImageAspectFlags::DEPTH;
                queue.clone()
            }
            ImageUsage::Sample(queue) => {
                usage_flags |= vk::ImageUsageFlags::SAMPLED;
                queue.clone()
            }
            ImageUsage::Storage(queue) => {
                usage_flags |= vk::ImageUsageFlags::STORAGE;
                queue.clone()
            }
        })
        .collect();

    queues.sort();
    queues.dedup();

    (usage_flags, aspect_flags, queues)
}

enum Storage {
    Slice(MemorySlice),
    Swapchain,
}

pub struct ImageInfo<'a> {
    pub device: Res<Device>,
    pub usages: &'a [ImageUsage],
    pub format: ImageFormat,
    pub extent: ImageExtent,
    pub sample_count: SampleCount,
    pub dimensions: ImageDimensions,
    pub mip_levels: u32,
}

/// TODO: Make images memory-mappable.
pub struct Image {
    pub(crate) handle: vk::Image,
    aspect_flags: vk::ImageAspectFlags,
    layout: Cell<ImageLayout>,
    dimensions: ImageDimensions,
    queues: Queues,
    size: DeviceSize,
    extent: ImageExtent,
    format: ImageFormat,
    storage: Storage,
    mip_levels: u32,
    device: Res<Device>,
}

impl Image {
    pub fn new<A: mem::Allocator>(allocator: &mut A, info: ImageInfo) -> Result<Self, RenderError> {
        let device = info.device;
        let (usage_flags, aspect_flags, queues) = parse_usages(info.usages);
        let array_layers = match info.dimensions {
            ImageDimensions::D1 | ImageDimensions::D2 => info.extent.depth_or_layers,
            ImageDimensions::D3 => 1,
        };
        let tiling = vk::ImageTiling::OPTIMAL;
        let flags = (array_layers == 6)
            .then_some(vk::ImageCreateFlags::CUBE_COMPATIBLE)
            .unwrap_or_default();
        let handle = if queues.len() > 1 {
            let queue_indices: SmallVec<[_; 4]> =
                queues.iter().map(|queue| queue.family_index()).collect();
            let info = vk::ImageCreateInfo::builder()
                .queue_family_indices(&queue_indices)
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .image_type(info.dimensions.into())
                .format(info.format.into())
                .mip_levels(info.mip_levels)
                .array_layers(array_layers)
                .extent(info.extent.into())
                .samples(info.sample_count.into())
                .usage(usage_flags)
                .tiling(tiling)
                .flags(flags)
                .build();
            unsafe { device.handle().create_image(&info, None)? }
        } else {
            let info = vk::ImageCreateInfo::builder()
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .image_type(info.dimensions.into())
                .format(info.format.into())
                .mip_levels(info.mip_levels)
                .array_layers(array_layers)
                .extent(info.extent.into())
                .samples(info.sample_count.into())
                .usage(usage_flags)
                .tiling(tiling)
                .flags(flags)
                .build();
            unsafe { device.handle().create_image(&info, None)? }
        };
        let reqs = unsafe { device.handle().get_image_memory_requirements(handle) };
        let layout = unsafe { mem::MemoryLayout::new_unchecked(reqs.size, reqs.alignment) };
        let memory_type = device
            .physical()
            .get_memory_type(reqs.memory_type_bits, MemoryLocation::Gpu)?;
        let slice = allocator.alloc(layout, memory_type)?;
        unsafe {
            device
                .handle()
                .bind_image_memory2(&[vk::BindImageMemoryInfo::builder()
                    .memory(slice.block().handle)
                    .memory_offset(slice.start())
                    .image(handle)
                    .build()])?;
        }
        let storage = Storage::Slice(slice);
        Ok(Self {
            layout: ImageLayout::Undefined.into(),
            dimensions: info.dimensions,
            mip_levels: info.mip_levels,
            format: info.format,
            extent: info.extent,
            size: reqs.size,
            aspect_flags,
            device,
            handle,
            storage,
            queues,
        })
    }

    pub fn dimensions(&self) -> ImageDimensions {
        self.dimensions
    }

    pub fn is_owned_by_swapchain(&self) -> bool {
        matches!(self.storage, Storage::Swapchain)
    }

    pub fn layout(&self) -> ImageLayout {
        self.layout.get()
    }

    pub fn queues(&self) -> &[Res<Queue>] {
        &self.queues
    }

    pub fn size(&self) -> DeviceSize {
        self.size
    }

    pub fn extent(&self) -> ImageExtent {
        self.extent
    }

    pub fn format(&self) -> ImageFormat {
        self.format
    }

    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    pub fn layer_count(&self) -> u32 {
        match self.dimensions() {
            ImageDimensions::D1 | ImageDimensions::D2 => self.extent().depth_or_layers,
            ImageDimensions::D3 => 1,
        }
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.device.handle().destroy_image(self.handle, None);
        }
    }
}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl Eq for Image {}

impl Hash for Image {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LevelRange {
    pub start: u32,
    pub count: u32,
}

pub struct ImageViewInfo {
    pub image: Res<Image>,
    pub kind: ImageViewKind,
    pub mip_range: LevelRange,
    pub layer_range: LevelRange,
}

pub struct ImageView {
    pub(crate) handle: vk::ImageView,
    mip_range: LevelRange,
    layer_range: LevelRange,
    image: Res<Image>,
}

fn image_view_type(
    dim: ImageDimensions,
    kind: ImageViewKind,
    layers: u32,
) -> Result<vk::ImageViewType, RenderError> {
    use {ImageDimensions::*, ImageViewKind::*};
    let view_type = match (dim, kind) {
        (D1, View) => vk::ImageViewType::TYPE_1D,
        (D2, View) => vk::ImageViewType::TYPE_2D,
        (D3, View) => vk::ImageViewType::TYPE_3D,
        (D2, Cube) if layers == 6 => vk::ImageViewType::CUBE,
        _ => {
            return Err(RenderError::InvalidImageViewKind {
                dimensions: dim,
                layers,
                kind,
            })
        }
    };
    Ok(view_type)
}

impl ImageView {
    pub fn new(device: Res<Device>, info: ImageViewInfo) -> Result<Self, RenderError> {
        let layers = info.image.extent().depth_or_layers;
        let dimensions = info.image.dimensions();
        let view_type = image_view_type(dimensions, info.kind, layers)?;
        if info.layer_range.start + info.layer_range.count >= info.image.layer_count() {
            panic!(
                "layer range outside range of image layers {}.",
                info.image.layer_count(),
            );
        }
        if info.mip_range.start + info.mip_range.count >= info.image.mip_levels() {
            panic!(
                "mip range outside range of image mip_levels {}.",
                info.image.mip_levels(),
            );
        }
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(info.image.aspect_flags)
            .base_mip_level(info.mip_range.start)
            .level_count(info.mip_range.count)
            .base_array_layer(info.layer_range.start)
            .layer_count(info.layer_range.count)
            .build();
        let view_info = vk::ImageViewCreateInfo::builder()
            .subresource_range(subresource_range)
            .format(info.image.format().into())
            .view_type(view_type)
            .image(info.image.handle)
            .build();
        let handle = unsafe { device.handle().create_image_view(&view_info, None)? };
        Ok(Self {
            layer_range: info.layer_range,
            mip_range: info.mip_range,
            image: info.image,
            handle,
        })
    }

    pub fn mip_range(&self) -> &LevelRange {
        &self.mip_range
    }

    pub fn level_range(&self) -> &LevelRange {
        &self.layer_range
    }

    pub fn image(&self) -> &Res<Image> {
        &self.image
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            self.image
                .device
                .handle()
                .destroy_image_view(self.handle, None);
        }
    }
}

impl PartialEq for ImageView {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl Eq for ImageView {}

impl Hash for ImageView {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
    }
}
