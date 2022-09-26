use anyhow::Result;
use ash::extensions::{ext, khr};
use ash::vk;

use std::ffi::{self, CStr, CString};
use std::env;
use std::cell::{RefCell, Cell};
use std::str::FromStr;
use crate::resource::*;
use crate::command::*;

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
            .get_queue_req(QueueRequestKind::Graphics(surface.clone()))
            .ok_or_else(|| anyhow!("can't find valid graphics queue"))?;
        let transfer_queue_req = physical
            .get_queue_req(QueueRequestKind::Transfer)
            .ok_or_else(|| anyhow!("can't find valid transfer queue"))?;
        let compute_queue_req = physical
            .get_queue_req(QueueRequestKind::Compute)
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
                recorder.image_barrier(&ImageBarrierInfo {
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
            frame.command_buffer.submit_wait(SubmitWaitInfo {
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

    pub fn get_queue_req(&self, kind: QueueRequestKind) -> Option<QueueRequest> {
        let index = match &kind {
            QueueRequestKind::Graphics(surface) => {
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
            QueueRequestKind::Transfer => {
                self.queue_properties
                    .iter()
                    .position(|p| p.queue_flags.contains(vk::QueueFlags::TRANSFER))
            }
            QueueRequestKind::Compute => {
                self.queue_properties
                    .iter()
                    .position(|p| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
            }
        };

        index.map(|family_index| {
            let flags = self.queue_properties[family_index].queue_flags;
            QueueRequest { flags, family_index: family_index as u32 }
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
        queue_reqs: &[&QueueRequest],
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
pub enum QueueRequestKind {
    Graphics(Res<Surface>),
    Transfer,
    Compute,
}

#[derive(Clone, Copy)]
pub struct QueueRequest {
    flags: vk::QueueFlags,
    family_index: u32,
}

/// A vulkan queue and it's family index.
pub struct Queue {
    pub handle: vk::Queue,
    pub pool: vk::CommandPool,

    #[allow(dead_code)]
    flags: vk::QueueFlags,

    family_index: u32,

    device: Res<Device>,
}

impl Queue {
    pub fn new(device: Res<Device>, req: &QueueRequest) -> Result<Self> {
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

        let presented = unsafe {
            device.handle.create_semaphore(&semaphore_info, None)?
        };

        let rendered = unsafe {
            device.handle.create_semaphore(&semaphore_info, None)?
        };

        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let ready_to_draw = unsafe {
            device.handle.create_fence(&fence_info, None)?
        };

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
                let image = pool.create_image_from_device(device.clone(), memory_flags,
                    &ImageInfo {
                        usage: vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                        aspect_flags: vk::ImageAspectFlags::COLOR,
                        kind: ImageKind::Swapchain { handle },
                        format: surface_format.format,
                        mip_levels: 1,
                        extent: vk::Extent3D {
                            width: extent.width,
                            height: extent.height,
                            depth: 1,
                        },
                    },
                )?;

                pool.create_image_view_from_device(device.clone(), &ImageViewInfo {
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
                let image = pool.create_image_from_device(self.device.clone(), memory_flags,
                    &ImageInfo {
                        usage: vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                        aspect_flags: vk::ImageAspectFlags::COLOR,
                        kind: ImageKind::Swapchain { handle },
                        format: self.surface_format.format,
                        mip_levels: 1,
                        extent: vk::Extent3D {
                            width: extent.width,
                            height: extent.height,
                            depth: 1,
                        },
                    },
                )?;

                pool.create_image_view_from_device(self.device.clone(), &ImageViewInfo {
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
