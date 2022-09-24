use anyhow::Result;
use ash::extensions::{ext, khr};
use ash::vk;
use smallvec::{SmallVec, smallvec};

use std::ffi::{self, CStr, CString};
use std::{mem, ops, slice, env};
use std::cell::{UnsafeCell, RefCell, Cell};
use std::rc::Rc;
use std::str::FromStr;
use crate::resource::*;

pub struct Renderer {
    pub device: Res<Device>,
    pub swapchain: Res<Swapchain>,

    frame_queue: FrameQueue,

    graphics_queue: Res<Queue>,
    transfer_queue: Res<Queue>,
    compute_queue: Res<Queue>,

    /// Pool of resources expected to live as long the renderer.
    pub static_pool: ResourcePool,

    /// Pool of resources which may be deleted / reallocated.
    ///
    /// TODO: Implement a resource pool which uses a buddy allocation strategy instead of bump.
    pub pool: ResourcePool,
}

impl Renderer {
    pub fn new(window: &winit::window::Window) -> Result<Self> {
        let static_pool = ResourcePool::with_block_size(50 * 1024 * 1024, 1024 * 1024);
        let pool = ResourcePool::with_block_size(10 * 1024 * 1024, 1024 * 1024);

        let validate = env::var("RENDINATOR_VALIDATE")
            .as_ref()
            .map(|var| bool::from_str(var).unwrap_or(false))
            .unwrap_or(false);

        let instance = pool.alloc(Instance::new(validate)?);
        let physical = PhysicalDevice::select(&instance)?;
        let surface = pool.alloc(Surface::new(instance.clone(), window)?);

        let graphics_queue_req = physical
            .get_queue_req(QueueReqKind::Graphics(surface.clone()))
            .ok_or_else(|| anyhow!("can't find valid graphics queue"))?;
        let transfer_queue_req = physical
            .get_queue_req(QueueReqKind::Transfer)
            .ok_or_else(|| anyhow!("can't find valid transfer queue"))?;
        let compute_queue_req = physical
            .get_queue_req(QueueReqKind::Compute)
            .ok_or_else(|| anyhow!("can't find valid compute queue"))?;

        let device = pool.alloc(Device::new(instance, physical, &[
            &graphics_queue_req,
            &transfer_queue_req,
            &compute_queue_req,
        ])?);

        let graphics_queue = pool.alloc(Queue::new(device.clone(), &graphics_queue_req)?);
        let transfer_queue = pool.alloc(Queue::new(device.clone(), &transfer_queue_req)?);
        let compute_queue = pool.alloc(Queue::new(device.clone(), &compute_queue_req)?);

        let window_extent = vk::Extent2D {
            width: window.inner_size().width,
            height: window.inner_size().height,
        };

        let swapchain = pool.alloc(Swapchain::new(
            device.clone(),
            &pool,
            surface.clone(),
            graphics_queue.clone(),
            window_extent,
        )?);

        let frame_queue = FrameQueue::new(device.clone(), graphics_queue.clone())?;

        device.wait_until_idle();

        Ok(Self {
            device,
            swapchain,
            frame_queue,
            graphics_queue,
            transfer_queue,
            compute_queue,
            static_pool,
            pool,
        })
    }

    pub fn draw<R>(&self, render: R) -> Result<()>
    where
        R: FnOnce(&CommandRecorder, FrameIndex, u32),
    {
        self.frame_queue.next_frame();

        let frame = self.frame_queue.current_frame();

        unsafe {
            self.device.handle.wait_for_fences(&[frame.ready_to_draw], true, u64::MAX)?;
            frame.command_buffer.reset()?;
        }

        loop {
            use NextSwapchainImage::*;

            let UpToDate { image_index } = self.swapchain.get_next_image(&frame)? else {
                // TODO: Do something here.
                panic!("out of date swapchain");
            };

            unsafe { self.device.handle.reset_fences(&[frame.ready_to_draw])?; }

            frame.command_buffer.record(SubmitCount::OneTime, |recorder| {
                let swapchain_image = self.swapchain.image(image_index);

                render(&recorder, frame.index, image_index);

                // Transition swapchain image to present layout.
                recorder.image_barrier(&ImageBarrierReq {
                    flags: vk::DependencyFlags::BY_REGION,
                    src_stage: vk::PipelineStageFlags2::RESOLVE,
                    dst_stage: vk::PipelineStageFlags2::empty(),
                    src_mask: vk::AccessFlags2::TRANSFER_WRITE,
                    dst_mask: vk::AccessFlags2::empty(),
                    new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    image: swapchain_image.image().clone(),
                    mips: 0..1,
                });

            })?;

            // Submit command buffer to be rendered. Wait for semaphore `frame.presented` first and
            // signals `frame.rendered´ and `frame.ready_to_draw` when all commands have been
            // executed.
            frame.command_buffer.submit_wait(SubmitWaitReq {
                wait_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                fence: frame.ready_to_draw,
                signal: frame.rendered,
                wait: frame.presented,
            })?;

            // Wait for the frame to be rendered before presenting it to the surface.
            unsafe {
                let wait = [frame.rendered];
                let swapchains = [self.swapchain.handle.get()];
                let indices = [image_index];

                let present_info = vk::PresentInfoKHR::builder()
                    .wait_semaphores(&wait)
                    .swapchains(&swapchains)
                    .image_indices(&indices);

                let res = self.swapchain.loader
                    .queue_present(self.transfer_queue.handle, &present_info)
                    .unwrap_or_else(|err| {
                        if let vk::Result::ERROR_OUT_OF_DATE_KHR = err {
                            warn!("out of data swapchain during present");
                        } else {
                            panic!("failed to present to swapchain");
                        }

                        false
                    });

                if res {
                    warn!("swapchain suboptimal for surface");
                }
            }

            break;
        }

        Ok(())
    }

    pub fn exec<F, R>(&self, queue: Res<Queue>, func: F) -> Result<R>
    where
        F: FnOnce(&CommandRecorder) -> R
    {
        let buffer = CommandBuffer::new(self.device.clone(), queue)?;
        let ret = buffer.record(SubmitCount::OneTime, func)?;
        buffer.submit_wait_idle()?;
        Ok(ret)
    }

    /// Record and submut a command seqeunce to `Self::transfer_queue`.
    pub fn transfer_with<F, R>(&self, func: F) -> Result<R>
    where
        F: FnOnce(&CommandRecorder) -> R
    {
        self.exec(self.transfer_queue.clone(), func)
    }

    /// Record and submut a command seqeunce to `Self::compute_queue`.
    pub fn compute_with<F, R>(&self, func: F) -> Result<R>
    where
        F: FnOnce(&CommandRecorder) -> R
    {
        self.exec(self.compute_queue.clone(), func)
    }

    pub fn transfer_queue(&self) -> Res<Queue> {
        self.transfer_queue.clone()
    }

    #[allow(dead_code)]
    pub fn graphics_queue(&self) -> Res<Queue> {
        self.graphics_queue.clone()
    }
    
    /// Handle window resize. The extent of the swapchain and framebuffers will we match that of
    /// `window`.
    pub fn resize(&mut self, window: &winit::window::Window) -> Result<()> {
        trace!("resizing");

        self.device.wait_until_idle();

        let extent = vk::Extent2D {
            width: window.inner_size().width,
            height: window.inner_size().height,
        };

        self.swapchain.recreate(&self.pool, extent)?;
        self.device.wait_until_idle();

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.device.wait_until_idle();
    }
}

pub struct Instance {
    entry: ash::Entry,
    handle: ash::Instance,
    messenger: DebugMessenger,
    layers: Vec<CString>,
}

impl Instance {
    pub fn new(enable_validation_layers: bool) -> Result<Self> {
        let entry = unsafe { ash::Entry::load()? };

        let mut debug_info = {
            use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;
            use vk::DebugUtilsMessageTypeFlagsEXT as Type;

            vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    Severity::ERROR | Severity::WARNING | Severity::INFO | Severity::VERBOSE,
                )
                .message_type(Type::GENERAL | Type::PERFORMANCE | Type::VALIDATION)
                .pfn_user_callback(Some(debug_callback))
        };

        let layers = if enable_validation_layers {
            vec![CString::new("VK_LAYER_KHRONOS_validation").unwrap()]
        } else {
            vec![]
        };

        let layer_names: Vec<_> = layers.iter().map(|layer| layer.as_ptr()).collect();
        let version = vk::make_api_version(0, 1, 3, 0);

        let handle = unsafe {
            let engine_name = CString::new("spillemotor").unwrap();
            let app_name = CString::new("spillemotor").unwrap();

            let app_info = vk::ApplicationInfo::builder()
                .application_name(&app_name)
                .application_version(vk::make_api_version(0, 0, 0, 1))
                .engine_name(&engine_name)
                .engine_version(vk::make_api_version(0, 0, 0, 1))
                .api_version(version);

            let ext_names = [
                ext::DebugUtils::name().as_ptr(),
                khr::Surface::name().as_ptr(),

                #[cfg(target_os = "windows")]
                khr::Win32Surface::name().as_ptr(),

                #[cfg(target_os = "linux")]
                khr::WaylandSurface::name().as_ptr(),

                #[cfg(target_os = "linux")]
                khr::XlibSurface::name().as_ptr(),

                #[cfg(target_os = "linux")]
                khr::XcbSurface::name().as_ptr(),

                #[cfg(target_os = "macos")]
                ext::MetalSurface::name().as_ptr(),

                #[cfg(target_os = "macos")]
                vk::KhrPortabilityEnumerationFn::name().as_ptr(),

                #[cfg(target_os = "macos")]
                vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
            ];

            let flags = if cfg!(target_os = "macos") {
                vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
            } else {
                vk::InstanceCreateFlags::default()
            };

            let info = vk::InstanceCreateInfo::builder()
                .flags(flags)
                .push_next(&mut debug_info)
                .application_info(&app_info)
                .enabled_layer_names(&layer_names)
                .enabled_extension_names(&ext_names);

            entry.create_instance(&info, None)?
        };

        let messenger = DebugMessenger::new(&entry, &handle, &debug_info)?;
        
        Ok(Self { entry, handle, messenger, layers })
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.messenger.loader.destroy_debug_utils_messenger(self.messenger.handle, None);
            self.handle.destroy_instance(None);
        }
    }
}

pub struct PhysicalDevice {
    handle: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_properties: Vec<vk::QueueFamilyProperties>,
}

impl PhysicalDevice {
    pub fn select(instance: &Instance) -> Result<Self> {
        let handle = unsafe {
            instance.handle
                .enumerate_physical_devices()?
                .into_iter()
                .max_by_key(|dev| {
                    let properties = instance.handle.get_physical_device_properties(*dev);

                    let name = CStr::from_ptr(properties.device_name.as_ptr())
                        .to_str()
                        .unwrap_or("invalid")
                        .to_string();

                    trace!("device candicate: {name}");

                    let mut score = properties.limits.max_image_dimension2_d;

                    if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                        score += 1000;
                    }

                    if properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
                        score += 500;
                    }

                    if properties.device_type == vk::PhysicalDeviceType::CPU {
                        score = 0;
                    }

                    score
                })
                .ok_or_else(|| anyhow!("no physical devices presented"))?
        };

        let memory_properties = unsafe { instance.handle.get_physical_device_memory_properties(handle) };
        let properties = unsafe { instance.handle.get_physical_device_properties(handle) };
        let queue_properties = unsafe { instance.handle.get_physical_device_queue_family_properties(handle) };

        Ok(Self { handle, memory_properties, properties, queue_properties })
    }

    pub fn get_queue_req(&self, kind: QueueReqKind) -> Option<QueueReq> {
        let index = match &kind {
            QueueReqKind::Graphics(surface) => {
                // This is technically not required to be exist, but it doesn't seem to be a
                // problem in reallity.
                let flags = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;

                self.queue_properties
                    .iter()
                    .enumerate()
                    .position(|(i, p)| {
                        p.queue_flags.contains(flags)
                            && unsafe {
                                surface
                                    .loader
                                    .get_physical_device_surface_support(
                                        self.handle, i as u32, surface.handle,
                                    )
                                    .unwrap_or(false)
                            }
                    })
            }
            QueueReqKind::Transfer => {
                self.queue_properties
                    .iter()
                    .position(|p| p.queue_flags.contains(vk::QueueFlags::TRANSFER))
            }
            QueueReqKind::Compute => {
                self.queue_properties
                    .iter()
                    .position(|p| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
            }
        };

        index.map(|family_index| {
            let flags = self.queue_properties[family_index].queue_flags;
            QueueReq { flags, family_index: family_index as u32 }
        })
    }
    
    pub fn get_memory_type_index(
        &self,
        type_bits: u32,
        flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let props = &self.memory_properties;

        props.memory_types[0..(props.memory_type_count as usize)]
            .iter()
            .enumerate()
            .map(|(i, memory_type)| (i as u32, memory_type))
            .position(|(i, memory_type)| {
                type_bits & (1 << i) != 0 && (memory_type.property_flags & flags) == flags
            })
            .map(|i| i as u32)
    }
}


/// The device and data connected to the device used for rendering. This struct data is static
/// after creation and doesn't depend on external factors such as display size.
pub struct Device {
    pub handle: ash::Device,
    pub physical: PhysicalDevice,

    instance: Res<Instance>,
}

impl Device {
    pub fn new(
        instance: Res<Instance>,
        physical: PhysicalDevice,
        queue_reqs: &[&QueueReq],
    ) -> Result<Self> {
        let priorities = [1.0_f32];

        let mut queue_infos: Vec<_> = queue_reqs
            .iter()
            .map(|req| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(req.family_index)
                    .queue_priorities(&priorities)
                    .build()
            })
            .collect();

        queue_infos.dedup_by_key(|info| info.queue_family_index);

        let extensions = [
            khr::Swapchain::name().as_ptr(),

            #[cfg(target_os = "macos")]
            vk::KhrPortabilitySubsetFn::name().as_ptr(),
        ];

        let enabled_features = vk::PhysicalDeviceFeatures::builder()
            .shader_storage_image_multisample(true)
            .sampler_anisotropy(true)
            .multi_draw_indirect(true)
            .build();

        let mut vk11_features = vk::PhysicalDeviceVulkan11Features::builder()
            .storage_buffer16_bit_access(true)
            .shader_draw_parameters(true)
            .build();

        let mut vk12_features = vk::PhysicalDeviceVulkan12Features::builder()
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_input_attachment_array_dynamic_indexing(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true)
            .sampler_filter_minmax(true)
            .scalar_block_layout(true)
            .draw_indirect_count(true)
            .descriptor_indexing(true)
            .shader_float16(true)
            .build();

        let mut vk13_features = vk::PhysicalDeviceVulkan13Features::builder()
            .dynamic_rendering(true)
            .synchronization2(true)
            .build();

        let layer_names: Vec<_> = instance.layers
            .iter()
            .map(|layer| layer.as_ptr())
            .collect();

        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layer_names)
            .enabled_features(&enabled_features)
            .push_next(&mut vk11_features)
            .push_next(&mut vk12_features)
            .push_next(&mut vk13_features);

        let handle = unsafe { instance.handle.create_device(physical.handle, &device_info, None)? };

        Ok(Self { instance, physical, handle })
    }

    /// Get the sample count for msaa. For now it just the highest sample count the device
    /// supports, but below 8 samples.
    pub fn sample_count(&self) -> vk::SampleCountFlags {
        let counts = self
            .physical
            .properties
            .limits
            .framebuffer_depth_sample_counts;

        // We don't wan't more than 8.
        let types = [
            vk::SampleCountFlags::TYPE_8,
            vk::SampleCountFlags::TYPE_4,
            vk::SampleCountFlags::TYPE_2,
        ];

        for t in types {
            if counts.contains(t) {
                return t;
            }
        }

        return vk::SampleCountFlags::TYPE_1;
    }

    pub fn wait_until_idle(&self) {
        unsafe {
            self.handle.device_wait_idle().expect("failed waiting for idle device");
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.handle.destroy_device(None); }
    }
}

/// State relating to the printing vulkan debug info.
struct DebugMessenger {
    loader: ext::DebugUtils,
    handle: vk::DebugUtilsMessengerEXT,
}

impl DebugMessenger {
    fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        info: &vk::DebugUtilsMessengerCreateInfoEXT,
    ) -> Result<Self> {
        let loader = ext::DebugUtils::new(&entry, &instance);
        let handle = unsafe { loader.create_debug_utils_messenger(&info, None)? };

        Ok(Self { loader, handle })
    }
}

#[derive(Clone)]
pub enum QueueReqKind {
    Graphics(Res<Surface>),
    Transfer,
    Compute,
}

#[derive(Clone, Copy)]
pub struct QueueReq {
    flags: vk::QueueFlags,
    family_index: u32,
}

/// A vulkan queue and it's family index.
pub struct Queue {
    handle: vk::Queue,
    pool: vk::CommandPool,

    #[allow(dead_code)]
    flags: vk::QueueFlags,

    family_index: u32,

    device: Res<Device>,
}

impl Queue {
    pub fn new(device: Res<Device>, req: &QueueReq) -> Result<Self> {
        let (pool, handle) = unsafe {
            let info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(req.family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            let pool = device.handle.create_command_pool(&info, None)?;
            let handle = device.handle.get_device_queue(req.family_index, 0);

            (pool, handle)
        };

        Ok(Self { handle, flags: req.flags, family_index: req.family_index, device, pool })
    }

    pub fn family_index(&self) -> u32 {
        self.family_index
    }

    pub fn wait_idle(&self) -> Result<()> {
        Ok(unsafe { self.device.handle.queue_wait_idle(self.handle)? })
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_command_pool(self.pool, None); }
    }
}

struct Frame {
    /// This get's signaled when the frame has been presented and is then available to draw to
    /// again.
    presented: vk::Semaphore,

    /// This get's signaled when the GPU is done drawing to the frame and the frame is then ready
    /// to be presented.
    rendered: vk::Semaphore,

    /// This is essentially the same as `presented`, but used to sync with the CPU.
    ready_to_draw: vk::Fence,

    /// Command buffer to for drawing and transfers. This has to be re-recorded before each frame,
    /// since the amount of frames (most likely) doesn't match the number if swapchain images.
    pub command_buffer: CommandBuffer,

    pub index: FrameIndex,
}

impl Frame {
    fn new(device: &Device, index: FrameIndex, command_buffer: CommandBuffer) -> Result<Self> {
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let presented = unsafe { device.handle.create_semaphore(&semaphore_info, None)? };
        let rendered = unsafe { device.handle.create_semaphore(&semaphore_info, None)? };

        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let ready_to_draw = unsafe { device.handle.create_fence(&fence_info, None)? };

        Ok(Self { presented, rendered, ready_to_draw, command_buffer, index })
    }
}

struct FrameQueue {
    frames: PerFrame<Frame>,

    /// The index of the frame currently being rendered or presented. It changes just before
    /// rendering of the next image begins.
    frame_index: Cell<FrameIndex>,

    device: Res<Device>,

    #[allow(dead_code)]
    graphics_queue: Res<Queue>,
}

impl FrameQueue {
    pub fn new(device: Res<Device>, graphics_queue: Res<Queue>) -> Result<Self> {
        let frames = PerFrame::try_from_fn(|index| {
            let buffer = CommandBuffer::new(device.clone(), graphics_queue.clone())?;
            Frame::new(&device, index, buffer)
        })?;

        let frame_index = Cell::new(FrameIndex::Uno);

        Ok(Self { frames, frame_index, device: device.clone(), graphics_queue })
    }

    pub fn next_frame(&self) {
        self.frame_index.set(match self.index() {
            FrameIndex::Uno => FrameIndex::Dos,
            FrameIndex::Dos => FrameIndex::Uno,
        });
    }

    pub fn current_frame(&self) -> &Frame {
        &self.frames[self.index()]
    }

    pub fn index(&self) -> FrameIndex {
        self.frame_index.get()
    }
}

impl Drop for FrameQueue {
    fn drop(&mut self) {
        unsafe {
            for frame in &self.frames {
                self.device.handle.destroy_semaphore(frame.rendered, None);
                self.device.handle.destroy_semaphore(frame.presented, None);
                self.device.handle.destroy_fence(frame.ready_to_draw, None);
            }
        }
    }
}

/// TODO: Make this handle more display servers besides Wayland.
pub struct Surface {
    handle: vk::SurfaceKHR,
    loader: khr::Surface,

    #[allow(dead_code)]
    instance: Res<Instance>,
}

impl Surface {
    fn new(instance: Res<Instance>, window: &winit::window::Window) -> Result<Self> {
        let loader = khr::Surface::new(&instance.entry, &instance.handle);

        use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
        let handle = match window.raw_window_handle() {
            #[cfg(target_os = "windows")]
            RawWindowHandle::Win32(handle) => {
                let info = vk::Win32SurfaceCreateInfoKHR::default()
                    .hinstance(handle.hinstance)
                    .hwnd(handle.hwnd);
                let loader = khr::Win32Surface::new(&instance.entry, &instance.handle);
                unsafe { loader.create_win32_surface(&info, None) }
            }

            #[cfg(target_os = "linux")]
            RawWindowHandle::Wayland(handle) => {
                let info = vk::WaylandSurfaceCreateInfoKHR::builder()
                    .display(handle.display)
                    .surface(handle.surface);
                let loader = khr::WaylandSurface::new(&instance.entry, &instance.handle);
                unsafe { loader.create_wayland_surface(&info, None) }
            }

            #[cfg(target_os = "linux")]
            RawWindowHandle::Xlib(handle) => {
                let info = vk::XlibSurfaceCreateInfoKHR::builder()
                    .dpy(handle.display as *mut _)
                    .window(handle.window);
                let loader = khr::XlibSurface::new(&instance.entry, &instance.handle);
                unsafe { loader.create_xlib_surface(&info, None) }
            }

            #[cfg(target_os = "linux")]
            RawWindowHandle::Xcb(handle) => {
                let info = vk::XcbSurfaceCreateInfoKHR::builder()
                    .connection(handle.connection)
                    .window(handle.window);
                let loader = khr::XcbSurface::new(&instance.entry, &instance.handle);
                unsafe { loader.create_xcb_surface(&info, None) }
            }

            #[cfg(target_os = "macos")]
            RawWindowHandle::AppKit(handle) => unsafe {
                use raw_window_metal::{appkit, Layer};

                let layer = appkit::metal_layer_from_handle(handle);
                let layer = match layer {
                    Layer::Existing(layer) | Layer::Allocated(layer) => layer as *mut _,
                    Layer::None => {
                        return Err(anyhow!("failed to load metal layer"));
                    }
                };

                let info = vk::MetalSurfaceCreateInfoEXT::builder().layer(&*layer);
                let loader = ext::MetalSurface::new(&instance.entry, &instance.handle);

                loader.create_metal_surface(&info, None)
            },
            _ => {
                return Err(anyhow!("unsupported platform"));
            }
        };
        Ok(Self { handle: handle?, loader, instance })
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_surface(self.handle, None) }
    }
}

pub struct Swapchain {
    loader: khr::Swapchain,

    present_mode: vk::PresentModeKHR,
    surface_format: vk::SurfaceFormatKHR,

    handle: Cell<vk::SwapchainKHR>,
    extent: Cell<vk::Extent2D>,

    images: RefCell<Vec<Res<ImageView>>>,

    device: Res<Device>,
    surface: Res<Surface>,
    graphics_queue: Res<Queue>,
}

enum NextSwapchainImage {
    UpToDate { image_index: u32 },
    OutOfDate,
}

impl Swapchain {
    /// Create a new swapchain. `extent` is used to determine the size of the swapchain images only
    /// if it aren't able to determine it from `surface`.
    pub fn new(
        device: Res<Device>,
        pool: &ResourcePool,
        surface: Res<Surface>,
        graphics_queue: Res<Queue>,
        extent: vk::Extent2D,
    ) -> Result<Self> {
        let (surface_formats, _present_modes, surface_caps) = unsafe {
            let format = surface
                .loader
                .get_physical_device_surface_formats(
                    device.physical.handle,
                    surface.handle,
                )?;
            let modes = surface
                .loader
                .get_physical_device_surface_present_modes(
                    device.physical.handle,
                    surface.handle,
                )?;
            let caps = surface
                .loader
                .get_physical_device_surface_capabilities(
                    device.physical.handle,
                    surface.handle,
                )?;
            (format, modes, caps)
        };

        let queue_families = [graphics_queue.family_index];
        let min_image_count = 2.max(surface_caps.min_image_count);

        let surface_format = surface_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| surface_formats.first())
            .ok_or_else(|| anyhow!("can't find valid surface format"))?
            .clone();

        let extent = if surface_caps.current_extent.width != u32::MAX {
            surface_caps.current_extent
        } else {
            vk::Extent2D {
                width: extent.width.clamp(
                    surface_caps.min_image_extent.width,
                    surface_caps.max_image_extent.width,
                ),
                height: extent.height.clamp(
                    surface_caps.min_image_extent.height,
                    surface_caps.max_image_extent.height,
                ),
            }
        };

        let preferred_present_mode = vk::PresentModeKHR::FIFO;

        let present_mode = _present_modes
            .iter()
            .any(|mode| *mode == preferred_present_mode)
            .then_some(preferred_present_mode)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let swapchain_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface.handle)
            .min_image_count(min_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .image_extent(extent)
            .image_array_layers(1);
        let loader = khr::Swapchain::new(&device.instance.handle, &device.handle);
        
        let handle = unsafe { loader.create_swapchain(&swapchain_info, None)? };
        let images = unsafe { loader.get_swapchain_images(handle)? };

        trace!("using {} swap chain images", images.len());

        let images: Result<Vec<_>> = images
            .into_iter()
            .map(|handle| {
                let memory_flags = vk::MemoryPropertyFlags::empty();
                let image = Image::from_device(device.clone(), pool, memory_flags, &ImageReq {
                    usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    aspect_flags: vk::ImageAspectFlags::COLOR,
                    kind: ImageKind::Swapchain { handle },
                    format: surface_format.format,
                    mip_levels: 1,
                    extent: vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    },
                })?;

                ImageView::from_device(device.clone(), pool, &ImageViewReq {
                    view_type: vk::ImageViewType::TYPE_2D,
                    image: image.clone(),
                    mips: 0..1,
                })
            })
            .collect();

        Ok(Self {
            handle: Cell::new(handle),
            extent: Cell::new(extent),
            images: RefCell::new(images?),
            device: device.clone(),
            graphics_queue,
            surface_format,
            present_mode,
            surface,
            loader,
        })
    }

    /// Recreate swapchain from `self` to a new `extent`.
    ///
    /// `extent` must be valid here unlike in `Self::new`, otherwise it could end in and endless
    /// cycle if recreating the swapchain, if for some reason the surface continues to give us and
    /// invalid extent.
    pub fn recreate(&self, pool: &ResourcePool, extent: vk::Extent2D) -> Result<()> {
        if extent.width == u32::MAX {
            return Err(anyhow!("`extent` must be valid when recreating swapchain"));
        }

        let surface_caps = unsafe {
            self.surface.loader.get_physical_device_surface_capabilities(
                self.device.physical.handle,
                self.surface.handle,
            )?
        };

        let queue_families = [self.graphics_queue.family_index];
        let min_image_count = (FRAMES_IN_FLIGHT as u32).max(surface_caps.min_image_count);

        let swapchain_info = vk::SwapchainCreateInfoKHR::builder()
            .old_swapchain(self.handle.get())
            .surface(self.surface.handle)
            .min_image_count(min_image_count)
            .image_format(self.surface_format.format)
            .image_color_space(self.surface_format.color_space)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.present_mode)
            .image_extent(extent)
            .image_array_layers(1);

        let new = unsafe { self.loader.create_swapchain(&swapchain_info, None)? };

        unsafe {
            self.loader.destroy_swapchain(self.handle.get(), None);
            self.images.borrow_mut().clear();
        }

        self.handle.set(new);
        self.extent.set(extent);

        let images = unsafe { self.loader.get_swapchain_images(self.handle.get())? };
        let images: Result<Vec<_>> = images
            .into_iter()
            .map(|handle| {
                let memory_flags = vk::MemoryPropertyFlags::empty();
                let image = Image::from_device(self.device.clone(), pool, memory_flags, &ImageReq {
                    usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    aspect_flags: vk::ImageAspectFlags::COLOR,
                    kind: ImageKind::Swapchain { handle },
                    format: self.surface_format.format,
                    mip_levels: 1,
                    extent: vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    },
                })?;

                ImageView::from_device(self.device.clone(), pool, &ImageViewReq {
                    view_type: vk::ImageViewType::TYPE_2D,
                    image: image.clone(),
                    mips: 0..1,
                })
            })
            .collect();

        *self.images.borrow_mut() = images?;

        Ok(())
    }

    pub fn image(&self, index: u32) -> Res<ImageView> {
        self.images.borrow()[index as usize].clone()
    }

    fn get_next_image(&self, frame: &Frame) -> Result<NextSwapchainImage> {
        let next_image = unsafe {
            self.loader.acquire_next_image(
                self.handle.get(),
                u64::MAX,
                frame.presented,
                vk::Fence::null(),
            )
        };
        match next_image {
            Ok((image_index, false)) => Ok(NextSwapchainImage::UpToDate { image_index }),
            Ok((_, true)) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                Ok(NextSwapchainImage::OutOfDate)
            }
            Err(result) => Err(result.into()),
        }
    }

    pub fn viewports(&self) -> [vk::Viewport; 1] {
        [vk::Viewport {
            width: self.extent().width as f32,
            height: self.extent().height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
            x: 0.0,
            y: 0.0,
        }]
    }

    pub fn scissors(&self) -> [vk::Rect2D; 1] {
        [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent(),
        }]
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.extent().width as f32 / self.extent().height as f32
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent.get()
    }

    pub fn extent_3d(&self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.extent().width,
            height: self.extent().height,
            depth: 1,
        }
    }

    pub fn format(&self) -> vk::Format {
        self.surface_format.format
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_swapchain(self.handle.get(), None);
        }
    }
}

pub struct ShaderModule {
    handle: vk::ShaderModule,
    entry: CString,

    device: Res<Device>,
}

impl ShaderModule {
    pub fn new(
        renderer: &Renderer,
        entry: &str,
        code: &[u8],
    ) -> Result<Self> {
        let device = renderer.device.clone();

        if code.len() % mem::size_of::<u32>() != 0 {
            return Err(anyhow!("shader code size must be a multiple of 4"));
        }

        if code.as_ptr().align_offset(mem::align_of::<u32>()) != 0 {
            return Err(anyhow!("shader code must be aligned to `u32`"));
        }

        let code = unsafe { slice::from_raw_parts(code.as_ptr() as *const u32, code.len() / 4) };
        let info = vk::ShaderModuleCreateInfo::builder()
            .code(code);
        let handle = unsafe { device.handle.create_shader_module(&info, None)? };

        let Ok(entry) = CString::new(entry) else {
            return Err(anyhow!("invalid entry name `entry`"));
        };

        Ok(ShaderModule { device, handle, entry })
    }

    fn stage_create_info(
        &self,
        stage: vk::ShaderStageFlags,
    ) -> impl ops::Deref<Target = vk::PipelineShaderStageCreateInfo> + '_ {
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(stage)
            .module(self.handle)
            .name(&self.entry)
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_shader_module(self.handle, None);
        }
    }
}

pub struct LayoutBinding {
    pub stage: vk::ShaderStageFlags,
    pub ty: vk::DescriptorType,
    pub array_count: Option<u32>,
}

pub struct LayoutBindings {
    pub bindings: SmallVec<[vk::DescriptorSetLayoutBinding; 6]>,
    pub flags: SmallVec<[vk::DescriptorBindingFlags; 6]>,
    pub variable_set_count: u32,
}

impl LayoutBindings {
    fn new(bindings: &[LayoutBinding]) -> Self  {
        let flags = bindings
            .iter()
            .map(|binding| {
                let mut flags = vk::DescriptorBindingFlags::empty(); 

                if binding.array_count.is_some() {
                    flags |= vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
                }

                flags
            })
            .collect();
        let variable_set_count = bindings
            .last()
            .map(|b| b.array_count)
            .flatten()
            .unwrap_or(1);
        let bindings = bindings
            .iter()
            .enumerate()
            .map(|(i, binding)| {
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(i as u32)
                    .descriptor_type(binding.ty)
                    .descriptor_count(binding.array_count.unwrap_or(1))
                    .stage_flags(binding.stage)
                    .build()
            })
            .collect();
        Self { bindings, flags, variable_set_count }
    }

    fn iter(&self) -> impl Iterator<Item = &vk::DescriptorSetLayoutBinding> {
        self.bindings.iter()
    }
}

pub enum DescriptorBinding<'a> {
    Buffer(Res<Buffer>),
    Image(Res<TextureSampler>, vk::ImageLayout, Res<ImageView>),
    VariableImageArray(
        Res<TextureSampler>,
        vk::ImageLayout,
        &'a [Res<ImageView>],
    ),
}

pub struct DescriptorSetLayout {
    pub handle: vk::DescriptorSetLayout,
    pub bindings: LayoutBindings,
    device: Res<Device>, 
}

impl DescriptorSetLayout {
    pub fn new(renderer: &Renderer, bindings: &[LayoutBinding]) -> Result<Self> {
        Self::create(renderer, LayoutBindings::new(bindings))
    }

    fn create(renderer: &Renderer, bindings: LayoutBindings) -> Result<Self> {
        let device = renderer.device.clone();
        let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
            .binding_flags(&bindings.flags)
            .build();
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings.bindings)
            .push_next(&mut binding_flags)
            .build();
        let handle = unsafe { device.handle.create_descriptor_set_layout(&layout_info, None)? };
        Ok(Self { handle, bindings, device })
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_descriptor_set_layout(self.handle, None); }
    }
}

pub struct DescriptorPool {
    pub handle: vk::DescriptorPool,

    device: Res<Device>,
}

impl DescriptorPool {
    pub fn new(
        renderer: &Renderer,
        max_sets: u32,
        sizes: &[vk::DescriptorPoolSize],
    ) -> Result<Self> {
        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&sizes)
            .max_sets(max_sets);

        let handle = unsafe {
            renderer.device.handle.create_descriptor_pool(&info, None)?
        };

        Ok(Self { handle, device: renderer.device.clone() })
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_descriptor_pool(self.handle, None); }
    }
}

pub struct DescriptorSet {
    pub handle: vk::DescriptorSet,

    pub layout: Res<DescriptorSetLayout>,
    pub pool: Rc<DescriptorPool>,

    #[allow(dead_code)]
    resources: SmallVec<[DummyRes; 32]>,

    device: Res<Device>,
}

impl DescriptorSet {
    pub fn new(
        renderer: &Renderer,
        resource_pool: &ResourcePool,
        layout: Res<DescriptorSetLayout>,
        bindings: &[DescriptorBinding],
    ) -> Result<Res<Self>> {
        let device = renderer.device.clone();
        let (handle, pool) = resource_pool.descriptor_alloc(renderer, &layout)?;

        struct Info {
            ty: vk::DescriptorType,
            buffers: SmallVec<[vk::DescriptorBufferInfo; 1]>,
            images: SmallVec<[vk::DescriptorImageInfo; 1]>,
        }

        let infos: SmallVec<[Info; 12]> = bindings.iter()
            .zip(layout.bindings.iter())
            .map(|(binding, layout_binding)| match &binding {
                DescriptorBinding::Buffer(buffer) => Info {
                    ty: layout_binding.descriptor_type,
                    images: smallvec![vk::DescriptorImageInfo::default()],
                    buffers: smallvec![vk::DescriptorBufferInfo {
                        buffer: buffer.handle,
                        range: buffer.size(),
                        offset: 0,
                    }],
                },
                DescriptorBinding::Image(sampler, layout, image) => Info {
                    ty: layout_binding.descriptor_type,
                    buffers: smallvec![vk::DescriptorBufferInfo::default()],
                    images: smallvec![vk::DescriptorImageInfo {
                        image_view: image.handle,
                        sampler: sampler.handle,
                        image_layout: *layout,
                    }],
                },
                DescriptorBinding::VariableImageArray(sampler, layout, array) => Info {
                    ty: layout_binding.descriptor_type,
                    buffers: smallvec![vk::DescriptorBufferInfo::default()],
                    images: array
                        .iter()
                        .map(|image| vk::DescriptorImageInfo {
                            image_layout: *layout,
                            image_view: image.handle,
                            sampler: sampler.handle,
                        })
                        .collect(),
                }
            })
            .collect();

        let writes: SmallVec<[vk::WriteDescriptorSet; 12]> = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::builder()
                    .dst_binding(binding as u32)
                    .descriptor_type(info.ty)
                    .buffer_info(&info.buffers)
                    .image_info(&info.images)
                    .dst_set(handle)
                    .build()
            })
            .collect();

        unsafe {
            device.handle.update_descriptor_sets(&writes, &[])
        }

        let mut resources: SmallVec<[_; 32]> = SmallVec::default();
        for binding in bindings {
            match &binding {
                DescriptorBinding::Buffer(buffer) => {
                    resources.push(buffer.create_dummy());
                }
                DescriptorBinding::Image(sampler, _, image) => {
                    resources.push(sampler.create_dummy());
                    resources.push(image.create_dummy());
                }
                DescriptorBinding::VariableImageArray(sampler, _, array) => {
                    resources.push(sampler.create_dummy());

                    for image in *array {
                        resources.push(image.create_dummy()); 
                    }
                }
            }
        }

        Ok(resource_pool.alloc(Self { device, layout, pool, handle, resources }))
    }

    pub fn layout(&self) -> Res<DescriptorSetLayout> {
        self.layout.clone()
    }
}

#[derive(Clone)]
pub struct PipelineLayout {
    pub handle: vk::PipelineLayout,

    #[allow(dead_code)]
    descriptor_layouts: SmallVec<[Res<DescriptorSetLayout>; 2]>,

    device: Res<Device>,
}

impl PipelineLayout {
    pub fn new(
        renderer: &Renderer,
        consts: &[vk::PushConstantRange],
        layouts: &[Res<DescriptorSetLayout>],
    ) -> Result<Self> {
        let device = renderer.device.clone();

        let handle = unsafe {
            let layouts: SmallVec<[_; 12]> = layouts
                .iter()
                .map(|layout| layout.handle)
                .collect();
            let info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&layouts)
                .push_constant_ranges(&consts);
            device.handle.create_pipeline_layout(&info, None)?
        };

        let mut descriptor_layouts = SmallVec::default();
        for layout in layouts {
            descriptor_layouts.push(layout.clone());
        }

        Ok(Self { handle, descriptor_layouts, device })
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_pipeline_layout(self.handle, None); }
    }
}

pub struct ComputePipeline {
    handle: vk::Pipeline,

    layout: Res<PipelineLayout>,
    device: Res<Device>,
}

impl ComputePipeline {
    pub fn new(
        renderer: &Renderer,
        layout: Res<PipelineLayout>,
        shader: &ShaderModule,
    ) -> Result<Self> {
        let device = renderer.device.clone();
        let stage = shader.stage_create_info(vk::ShaderStageFlags::COMPUTE);
        let create_infos = [vk::ComputePipelineCreateInfo::builder()
            .layout(layout.handle)
            .stage(*stage)
            .build()];
        let handle = unsafe {
            device.handle
                .create_compute_pipelines(vk::PipelineCache::null(), &create_infos, None)
                .map_err(|(_, err)| err)?
                .first()
                .unwrap()
                .clone()
        };
        Ok(Self { device, handle, layout })
    }

    pub fn layout(&self) -> Res<PipelineLayout> {
        self.layout.clone()
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_pipeline(self.handle, None); }
    }
}

pub struct GraphicsPipelineReq<'a> {
    pub layout: Res<PipelineLayout>,

    pub vertex_attributes: &'a [vk::VertexInputAttributeDescription],
    pub vertex_bindings: &'a [vk::VertexInputBindingDescription],
    pub depth_stencil_info: &'a vk::PipelineDepthStencilStateCreateInfo,

    pub depth_format: vk::Format,
    pub color_format: vk::Format,
    pub sample_count: vk::SampleCountFlags,

    pub vertex_shader: &'a ShaderModule,
    pub fragment_shader: &'a ShaderModule,

    pub cull_mode: vk::CullModeFlags,
}

pub struct GraphicsPipeline {
    pub handle: vk::Pipeline,

    layout: Res<PipelineLayout>,
    device: Res<Device>,
}

impl GraphicsPipeline {
    pub fn new(renderer: &Renderer, req: GraphicsPipelineReq) -> Result<Self> {
        let device = renderer.device.clone();

        let shader_stages = [
            *req.vertex_shader.stage_create_info(vk::ShaderStageFlags::VERTEX),
            *req.fragment_shader.stage_create_info(vk::ShaderStageFlags::FRAGMENT),
        ];

        let vert_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&req.vertex_attributes)
            .vertex_binding_descriptions(&req.vertex_bindings);

        let vert_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewports = renderer.swapchain.viewports();
        let scissors = renderer.swapchain.scissors();

        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterize_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .front_face(vk::FrontFace::CLOCKWISE)
            .cull_mode(req.cull_mode)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0);

        let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(req.sample_count);

        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_blend_op(vk::BlendOp::ADD)
            .blend_enable(true)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build()];

        let color_blend_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(&color_blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        let color_formats = [req.color_format];

        let mut rendering_info = vk::PipelineRenderingCreateInfo::builder()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(req.depth_format);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .dynamic_state(&dynamic_state)
            .stages(&shader_stages)
            .vertex_input_state(&vert_input_info)
            .input_assembly_state(&vert_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterize_info)
            .multisample_state(&multisample_info)
            .depth_stencil_state(&req.depth_stencil_info)
            .color_blend_state(&color_blend_info)
            .layout(req.layout.handle)
            .push_next(&mut rendering_info);

        let handle = unsafe {
            *device.handle
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_info.build()],
                    None,
                )
                .map_err(|(_, error)| anyhow!("failed to create pipeline: {error}"))?
                .first()
                .unwrap()
        };

        Ok(Self { device, handle, layout: req.layout })
    }

    pub fn layout(&self) -> Res<PipelineLayout> {
        self.layout.clone()
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_pipeline(self.handle, None); }
    }
}

pub struct CommandBuffer {
    handle: vk::CommandBuffer,

    /// This keeps track of all the items used by the command buffer.
    ///
    /// It's not free since clearing this means jumping through a pointer for each item, but it
    /// makes sure the items live as long as they are used by the buffer.
    bound_resources: UnsafeCell<Vec<DummyRes>>,

    queue: Res<Queue>,
    device: Res<Device>,
}

pub enum SubmitCount {
    OneTime,
    Multiple,
}

impl CommandBuffer {
    pub fn new(device: Res<Device>, queue: Res<Queue>) -> Result<Self> {
        let info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(queue.pool)
            .command_buffer_count(1);

        let handles = unsafe { device.handle.allocate_command_buffers(&info)? };
        let bound_resources = UnsafeCell::new(Vec::new());

        Ok(Self { handle: *handles.first().unwrap(), queue, device, bound_resources })
    }

    pub fn reset(&self) -> Result<()> {
        unsafe {
            let flags = vk::CommandBufferResetFlags::empty();
            self.device.handle.reset_command_buffer(self.handle, flags)?;

            // We can clear all the bound items when the buffer is reset.
            (*self.bound_resources.get()).clear();
        }

        Ok(())
    }

    fn bind_resource<T>(&self, res: Res<T>) {
        unsafe { (*self.bound_resources.get()).push(res.to_dummy()); }
    }

    pub fn record<F, R>(&self, submit_count: SubmitCount, func: F) -> Result<R>
    where
        F: FnOnce(&CommandRecorder) -> R
    {
        let flags = match submit_count {
            SubmitCount::OneTime => vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            SubmitCount::Multiple => vk::CommandBufferUsageFlags::empty(),
        };

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::builder().flags(flags);
            self.device.handle.begin_command_buffer(self.handle, &begin_info)?;
        }

        let recorder = CommandRecorder { buffer: &self };
        let ret = func(&recorder);

        unsafe { self.device.handle.end_command_buffer(self.handle)?; }

        Ok(ret)
    }
}

pub struct SubmitWaitReq {
    signal: vk::Semaphore,
    wait: vk::Semaphore,
    wait_stage: vk::PipelineStageFlags,
    fence: vk::Fence,
}

impl CommandBuffer {
    pub fn submit_wait(&self, req: SubmitWaitReq) -> Result<()> {
        let wait = [req.wait];
        let signals = [req.signal];
        let command_buffers = [self.handle];
        let stages = [req.wait_stage];

        let submit_info = [vk::SubmitInfo::builder()
            .wait_dst_stage_mask(&stages)
            .wait_semaphores(&wait)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signals)
            .build()];

        unsafe {
            self.device.handle.queue_submit(
                self.queue.handle,
                &submit_info,
                req.fence,
            )?;
        }

        Ok(())
    }

    pub fn submit_wait_idle(&self) -> Result<()> {
        let buffers = [self.handle];
        let submit_infos = [vk::SubmitInfo::builder()
            .command_buffers(&buffers)
            .build()];

        unsafe {
            self.device.handle.queue_submit(
                self.queue.handle,
                &submit_infos,
                vk::Fence::null(),
            )?;
        }

        self.queue.wait_idle()
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        self.queue.wait_idle().expect("failed to wait for idle queue");
        unsafe { self.device.handle.free_command_buffers(self.queue.pool, &[self.handle]) }
    }
}

pub struct IndexedDrawReq {
    pub instance: u32,
    pub index_count: u32,
    pub index_start: u32,
    pub vertex_offset: i32,
}

pub struct IndexedIndirectDrawReq {
    pub draw_buffer: Res<Buffer>,
    pub count_buffer: Res<Buffer>,

    pub draw_command_size: vk::DeviceSize,
    pub draw_offset: vk::DeviceSize,
    pub count_offset: vk::DeviceSize,

    pub max_draw_count: u32,
}

pub struct BufferBarrierReq {
    pub buffer: Res<Buffer>,

    pub src_mask: vk::AccessFlags2,
    pub dst_mask: vk::AccessFlags2,
    pub src_stage: vk::PipelineStageFlags2,
    pub dst_stage: vk::PipelineStageFlags2,
}

pub struct ImageBarrierReq {
    pub image: Res<Image>,

    pub flags: vk::DependencyFlags,

    pub src_mask: vk::AccessFlags2,
    pub dst_mask: vk::AccessFlags2,
    pub src_stage: vk::PipelineStageFlags2,
    pub dst_stage: vk::PipelineStageFlags2,

    pub new_layout: vk::ImageLayout,

    pub mips: ops::Range<u32>,
}

pub struct ImageResolveReq {
    pub src: Res<Image>,
    pub dst: Res<Image>,
    pub src_mip: u32,
    pub dst_mip: u32,
}

pub struct ImageBlitReq {
    pub src: Res<Image>,
    pub dst: Res<Image>,
    pub filter: vk::Filter,
    pub src_mip: u32,
    pub dst_mip: u32,
}

pub struct DescriptorBindReq<'a> {
    pub bind_point: vk::PipelineBindPoint,
    pub layout: Res<PipelineLayout>,
    pub descriptors: &'a [Res<DescriptorSet>],
}

pub struct BeginRenderingReq {
    pub color_target: Res<ImageView>,
    pub depth_target: Res<ImageView>,
    pub swapchain: Res<Swapchain>,
}

pub struct CommandRecorder<'a> {
    buffer: &'a CommandBuffer,
}

impl<'a> CommandRecorder<'a> {
    fn device(&self) -> &Device {
        &self.buffer.device
    }

    pub fn copy_buffers(&self, src: Res<Buffer>, dst: Res<Buffer>) {
        let size = src.size().min(dst.size());
        let regions = [vk::BufferCopy2::builder()
            .src_offset(0)
            .dst_offset(0)
            .size(size)
            .build()];

        let info = vk::CopyBufferInfo2::builder()
            .src_buffer(src.handle)
            .dst_buffer(dst.handle)
            .regions(&regions);

        unsafe { 
            self.device().handle.cmd_copy_buffer2(self.buffer.handle, &info);
        }

        self.buffer.bind_resource(src);
        self.buffer.bind_resource(dst);
    }

    pub fn copy_buffer_to_image(&self, src: Res<Buffer>, dst: Res<Image>, mip_level: u32) {
        let subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(dst.aspect_flags())
            .mip_level(mip_level)
            .base_array_layer(0)
            .layer_count(dst.layer_count())
            .build();
        
        let extent = dst.extent(mip_level);

        let regions = [vk::BufferImageCopy2::builder()
            .buffer_offset(0)
            .buffer_row_length(extent.width)
            .buffer_image_height(0)
            .image_extent(extent)
            .image_subresource(subresource)
            .build()];

        let info = vk::CopyBufferToImageInfo2::builder()
            .src_buffer(src.handle)
            .dst_image(dst.handle)
            .dst_image_layout(dst.layout())
            .regions(&regions);

        unsafe {
            self.device().handle.cmd_copy_buffer_to_image2(self.buffer.handle, &info);
        }

        self.buffer.bind_resource(src);
        self.buffer.bind_resource(dst);
    }

    pub fn image_barrier(&self, req: &ImageBarrierReq) {
        let subresource = vk::ImageSubresourceRange::builder()
            .aspect_mask(req.image.aspect_flags())
            .base_mip_level(req.mips.start)
            .level_count(req.mips.end - req.mips.start)
            .base_array_layer(0)
            .layer_count(req.image.layer_count())
            .build();
        let barriers = [vk::ImageMemoryBarrier2::builder()
            .image(req.image.handle)
            .old_layout(req.image.layout())
            .new_layout(req.new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(req.src_mask)
            .dst_access_mask(req.dst_mask)
            .src_stage_mask(req.src_stage)
            .dst_stage_mask(req.dst_stage)
            .subresource_range(subresource)
            .build()];

        req.image.layout.set(req.new_layout);

        let dependency_info = vk::DependencyInfo::builder()
            .dependency_flags(req.flags)
            .image_memory_barriers(&barriers);

        unsafe {
            self.device().handle.cmd_pipeline_barrier2(self.buffer.handle, &dependency_info);
        }

        self.buffer.bind_resource(req.image.clone());

    }

    pub fn resolve_image(&self, req: &ImageResolveReq) {
        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(req.src.aspect_flags())
            .base_array_layer(0)
            .layer_count(1)
            .mip_level(req.src_mip)
            .build();
        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(req.dst.aspect_flags())
            .base_array_layer(0)
            .layer_count(1)
            .mip_level(req.dst_mip)
            .build();
        let regions = [vk::ImageResolve2::builder()
            .src_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .dst_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .src_subresource(src_subresource)
            .dst_subresource(dst_subresource)
            .extent(req.src.extent(req.src_mip))
            .build()];
        let resolve_info = vk::ResolveImageInfo2::builder()
           .src_image(req.src.handle)
           .dst_image(req.dst.handle)
           .src_image_layout(req.src.layout())
           .dst_image_layout(req.dst.layout())
           .regions(&regions);

        unsafe {
            self.device().handle.cmd_resolve_image2(self.buffer.handle, &resolve_info);
        }

        self.buffer.bind_resource(req.src.clone());
        self.buffer.bind_resource(req.dst.clone());
    }

    pub fn blit_image(&self, req: &ImageBlitReq) {
        let src_extent = req.src.extent(req.src_mip);
        let dst_extent = req.dst.extent(req.dst_mip);

        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(req.src.aspect_flags())
            .base_array_layer(0)
            .layer_count(1)
            .mip_level(req.src_mip)
            .build();
        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(req.dst.aspect_flags())
            .base_array_layer(0)
            .layer_count(1)
            .mip_level(req.dst_mip)
            .build();
        let regions = [vk::ImageBlit2::builder()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: src_extent.width as i32,
                    y: src_extent.height as i32,
                    z: 1,
                },
            ])
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: dst_extent.width as i32,
                    y: dst_extent.height as i32,
                    z: 1,
                },
            ])
            .src_subresource(src_subresource)
            .dst_subresource(dst_subresource)
            .build()];
        let blit_info = vk::BlitImageInfo2::builder()
           .src_image(req.src.handle)
           .dst_image(req.dst.handle)
           .src_image_layout(req.src.layout())
           .dst_image_layout(req.dst.layout())
           .filter(req.filter)
           .regions(&regions);

        unsafe {
            self.device().handle.cmd_blit_image2(self.buffer.handle, &blit_info);
        }

        self.buffer.bind_resource(req.src.clone());
        self.buffer.bind_resource(req.dst.clone());
    }

    /// Transition the layout of `image` to `new`.
    pub fn transition_image_layout(&self, image: Res<Image>, new: vk::ImageLayout) {
        let (src_stage, dst_stage, src_mask, dst_mask) = match (image.layout(), new) {
            (_, vk::ImageLayout::GENERAL) => (
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::empty(),
            ),
            (_, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::empty(),
            ),
            (_, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::PipelineStageFlags2::TRANSFER,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::empty(),
            ),
            _ => {
                todo!()
            }
        };

        self.image_barrier(&ImageBarrierReq {
            flags: vk::DependencyFlags::BY_REGION,
            mips: 0..image.mip_level_count(),
            new_layout: new,
            src_stage,
            dst_stage,
            src_mask,
            dst_mask,
            image,
        });
    }

    pub fn begin_rendering(&self, req: &BeginRenderingReq) {
        if req.color_target.image().layout() != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
            self.image_barrier(&ImageBarrierReq {
                flags: vk::DependencyFlags::BY_REGION,
                src_stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                dst_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                src_mask: vk::AccessFlags2::NONE,
                dst_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                image: req.color_target.image().clone(),
                mips: 0..1,
            });
        }

        let color_attachments = [
            vk::RenderingAttachmentInfo::builder()
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image_view(req.color_target.handle)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                })
                .build(),
        ];

        let depth_resolve_attachment = vk::RenderingAttachmentInfo::builder()
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .image_view(req.depth_target.handle)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            });

        let rendering_info = vk::RenderingInfo::builder()
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_resolve_attachment)
            .layer_count(1)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: req.swapchain.extent(),
            });

        let viewports = req.swapchain.viewports();
        let scissors = req.swapchain.scissors();

        unsafe {
            self.device().handle.cmd_begin_rendering(self.buffer.handle, &rendering_info);
            self.device().handle.cmd_set_viewport(self.buffer.handle, 0, &viewports);
            self.device().handle.cmd_set_scissor(self.buffer.handle, 0, &scissors);
        }
    }

    pub fn end_rendering(&self) {
        unsafe {
            self.device().handle.cmd_end_rendering(self.buffer.handle);
        };
    }

    pub fn dispatch(&self, pipeline: Res<ComputePipeline>, group_count: [u32; 3]) {
        let bind_point = vk::PipelineBindPoint::COMPUTE;
        unsafe {
            self.device().handle.cmd_bind_pipeline(
                self.buffer.handle,
                bind_point,
                pipeline.handle,
            );
            self.device().handle.cmd_dispatch(
                self.buffer.handle,
                group_count[0],
                group_count[1],
                group_count[2],
            );
        }

        self.buffer.bind_resource(pipeline);
    }

    pub fn bind_descriptor_sets(&self, req: &DescriptorBindReq) {
        let descs: SmallVec<[_; 12]> = req.descriptors
            .iter()
            .map(|desc| desc.handle)
            .collect();

        unsafe {
            self.device().handle.cmd_bind_descriptor_sets(
                self.buffer.handle,
                req.bind_point,
                req.layout.handle,
                0,
                &descs,
                &[],
            );
        }

        self.buffer.bind_resource(req.layout.clone());

        for desc in req.descriptors {
            self.buffer.bind_resource(desc.clone());
        }

    }

    pub fn bind_vertex_buffer(&self, buffer: Res<Buffer>) {
        unsafe {
            self.device().handle.cmd_bind_vertex_buffers(
                self.buffer.handle,
                0,
                &[buffer.handle],
                &[0],
            );
        }

        self.buffer.bind_resource(buffer);
    }

    pub fn bind_index_buffer(&self, buffer: Res<Buffer>, index_type: vk::IndexType) {
        unsafe {
            self.device().handle.cmd_bind_index_buffer(
                self.buffer.handle,
                buffer.handle,
                0,
                index_type,
            );
        }

        self.buffer.bind_resource(buffer);
    }

    pub fn push_constants<T: bytemuck::NoUninit>(
        &self,
        layout: Res<PipelineLayout>,
        stage: vk::ShaderStageFlags,
        offset: u32,
        val: &T,
    ) {
        let bytes = bytemuck::bytes_of(val);
        unsafe {
            self.device().handle.cmd_push_constants(
                self.buffer.handle,
                layout.handle,
                stage,
                offset,
                bytes,
            );
        }

        self.buffer.bind_resource(layout);
    }

    pub fn bind_graphics_pipeline(&self, pipeline: Res<GraphicsPipeline>) {
        let bind_point = vk::PipelineBindPoint::GRAPHICS;
        unsafe {
            self.device().handle.cmd_bind_pipeline(
                self.buffer.handle,
                bind_point,
                pipeline.handle,
            );
        }

        self.buffer.bind_resource(pipeline);
    }
    
    pub fn draw(&self, vertex_count: u32, vertex_start: u32) {
        unsafe {
            self.device().handle.cmd_draw(
                self.buffer.handle,
                vertex_count,
                1,
                vertex_start,
                0,
            );
        }
    }

    pub fn draw_indexed(&self, req: IndexedDrawReq) {
        unsafe {
            self.device().handle.cmd_draw_indexed(
                self.buffer.handle,
                req.index_count,
                1,
                req.index_start,
                req.vertex_offset,
                req.instance,
            );
        }
    }

    pub fn update_buffer<T: bytemuck::NoUninit>(&self, buffer: Res<Buffer>, val: &T) {
        assert_eq!(
            buffer.size(),
            mem::size_of::<T>() as vk::DeviceSize,
            "size of buffer doesn't match length of data"
        );

        unsafe {
            self.device().handle.cmd_update_buffer(
                self.buffer.handle,
                buffer.handle,
                0,
                bytemuck::bytes_of(val),
            );
        }

        self.buffer.bind_resource(buffer);
    }

    pub fn draw_indexed_indirect_count(&self, req: &IndexedIndirectDrawReq) {
        unsafe {
            self.device().handle.cmd_draw_indexed_indirect_count(
                self.buffer.handle,
                req.draw_buffer.handle,
                req.draw_offset,
                req.count_buffer.handle,
                req.count_offset,
                req.max_draw_count,
                req.draw_command_size as u32,
            );
        }

        self.buffer.bind_resource(req.draw_buffer.clone());
        self.buffer.bind_resource(req.count_buffer.clone());
    }

    pub fn buffer_barrier(&self, req: &BufferBarrierReq) {
        let barriers = [vk::BufferMemoryBarrier2::builder()
            .src_access_mask(req.src_mask)
            .dst_access_mask(req.dst_mask)
            .src_stage_mask(req.src_stage)
            .dst_stage_mask(req.dst_stage)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(req.buffer.handle)
            .offset(0)
            .size(req.buffer.size())
            .build()];

        let dependency_info = vk::DependencyInfo::builder()
            .dependency_flags(vk::DependencyFlags::empty())
            .buffer_memory_barriers(&barriers);

        unsafe {
            self.device().handle.cmd_pipeline_barrier2(self.buffer.handle, &dependency_info);
        }

        self.buffer.bind_resource(req.buffer.clone());
    }
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    cb_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut ffi::c_void,
) -> vk::Bool32 {
    let message = CStr::from_ptr((*cb_data).p_message);

    use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;

    if severity.contains(Severity::ERROR) {
        error!("vulkan({ty:?}): {message:?}");
    } else if severity.contains(Severity::WARNING) {
        warn!("vulkan({ty:?}): {message:?}");
    } else if severity.contains(Severity::INFO) {
        info!("vulkan({ty:?}): {message:?}");
    } else if severity.contains(Severity::VERBOSE) {
        trace!("vulkan({ty:?}): {message:?}");
    }

    vk::FALSE
}

/// The number of frames being worked on concurrently. This could easily be higher, you could for
/// instance work on all swapchain images concurrently, but you risk having the CPU run ahead,
/// which add latency and delays. You also save some memory by having less depth buffers and such.
pub const FRAMES_IN_FLIGHT: usize = 2;

#[derive(Clone, Copy)]
pub enum FrameIndex {
    Uno = 0,
    Dos = 1,
}

impl FrameIndex {
    pub const ALL: [Self; FRAMES_IN_FLIGHT] = [
        FrameIndex::Uno,
        FrameIndex::Dos,
    ];

    pub fn enumerate() -> impl Iterator<Item = Self> {
        Self::ALL.into_iter()
    }

    pub fn last(self) -> Self {
        match self {
            FrameIndex::Uno => FrameIndex::Dos,
            FrameIndex::Dos => FrameIndex::Uno,
        }
    }
}
