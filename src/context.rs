use std::rc::Rc;

use winit::{dpi::PhysicalSize, window::Window};

pub struct Context {
    pub surface_size: wgpu::Extent3d,
    pub surface_format: wgpu::TextureFormat,
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub limits: wgpu::Limits,
    pub window: Rc<Window>,
    pub frame_index: usize,
    pub backend: wgpu::Backend,
    pub present_mode: wgpu::PresentMode,
    pub shader_composer: naga_oil::compose::Composer,
}

impl Context {
    pub fn new(window: Rc<Window>) -> Self {
        let surface_size = physical_size_to_texture_size(window.inner_size());

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
            backends: wgpu::Backends::all(),
        });

        let surface = unsafe {
            instance
                .create_surface(window.as_ref())
                .expect("failed creating surface")
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("failed request of adapter");

        let backend = adapter.get_info().backend;

        let features = wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
            | wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::CLEAR_TEXTURE
            | wgpu::Features::PUSH_CONSTANTS
            | wgpu::Features::VERTEX_WRITABLE_STORAGE;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                limits: wgpu::Limits {
                    max_push_constant_size: 128,
                    max_sampled_textures_per_shader_stage: 128,
                    ..Default::default()
                },
                label: Some("device"),
                features,
            },
            None,
        ))
        .expect("failed request of device and queue");

        let format = surface
            .get_capabilities(&adapter)
            .formats
            .iter()
            .find(|format| format.is_srgb())
            .expect("no supported surface formats")
            .clone();

        let present_mode = wgpu::PresentMode::Fifo;
        let alpha_mode = wgpu::CompositeAlphaMode::Auto;

        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                width: surface_size.width,
                height: surface_size.height,
                view_formats: vec![],
                present_mode,
                alpha_mode,
                format,
            },
        );

        let shader_composer = create_shader_composer();

        Self {
            limits: adapter.limits(),
            surface_format: format,
            surface_size,
            present_mode,
            window,
            surface,
            device,
            queue,
            backend,
            frame_index: 0,
            shader_composer,
        }
    }

    pub fn create_shader_module(&mut self, source: &str, path: &str) -> naga::Module {
        self.shader_composer
            .make_naga_module(naga_oil::compose::NagaModuleDescriptor {
                source,
                file_path: &path,
                ..Default::default()
            })
            .unwrap_or_else(|err| {
                let err = err.emit_to_string(&self.shader_composer);
                panic!("failed to create shader module {path}: {err}")
            })
    }

    pub fn resize_surface(&mut self, size: PhysicalSize<u32>) {
        let is_minimized = size.width == 0 || size.height == 0;

        let has_changed =
            size.width != self.surface_size.width || size.height != self.surface_size.height;
        self.surface_size = physical_size_to_texture_size(size);

        if !is_minimized && has_changed {
            self.surface.configure(
                &self.device,
                &wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    present_mode: self.present_mode,
                    format: self.surface_format,
                    width: size.width,
                    height: size.height,
                    view_formats: vec![],
                },
            );
        }
    }
}

fn create_shader_composer() -> naga_oil::compose::Composer {
    let mut composer = naga_oil::compose::Composer::default();
    composer.validate = false;

    macro_rules! add_include {
        ($file_path:literal) => {
            composer
                .add_composable_module(naga_oil::compose::ComposableModuleDescriptor {
                    source: include_str!($file_path),
                    file_path: $file_path,
                    ..Default::default()
                })
                .unwrap_or_else(|err| panic!("failed to include shader {}: {err}", $file_path));
        };
    }

    add_include!("include_shaders/util.wgsl");
    add_include!("include_shaders/consts.wgsl");
    add_include!("include_shaders/mesh.wgsl");
    add_include!("include_shaders/pbr.wgsl");
    add_include!("include_shaders/light.wgsl");

    composer
}

fn physical_size_to_texture_size(size: PhysicalSize<u32>) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: size.width,
        height: size.height,
        depth_or_array_layers: 1,
    }
}
