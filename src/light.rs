use ash::vk;
use anyhow::Result;

use std::mem;

use crate::scene::*;
use crate::camera::*;
use crate::resource::*;
use crate::command::*;
use crate::core::*;

use rendi_math::prelude::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::NoUninit)]
pub struct DirLight {
    pub direction: Vec4,
    pub irradiance: Vec4,
}

impl Default for DirLight {
    fn default() -> Self {
        Self {
            direction: Vec4::new(0.0, 0.8, -1.0, 1.0).normalize(),
            irradiance: Vec4::splat(2.0),
        }
    }
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, bytemuck::NoUninit)]
pub struct PointLight {
    world_position: Vec4,
    lum_radius: Vec4,
}

impl PointLight {
    pub fn new(pos: Vec3, lum: Vec3) -> Self {
        const POW_CUTOFF: f32 = 0.6;

        // Calculate the radius effected by the light.
        //
        // More specifically calculate distance from the light where the irradiance hits
        // `POW_CUTOFF`.
        let radius = (2.82095 * lum.max_element().sqrt()) / POW_CUTOFF.sqrt();
        
        let world_position = pos.extend(1.0);
        let lum_radius = lum.extend(radius);

        Self { world_position, lum_radius }
    }
}

#[repr(C)]
struct LightPos {
    pos_radius: Vec4,
}

#[repr(C)]
struct Aabb {
    min: Vec4,
    max: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
pub struct LightInfo {
    dir_light: DirLight,

    /// The number of subdivisions in each axis.
    ///
    /// w is the total number of clusters:
    subdivisions: UVec4,

    /// The size of the clusters in screen space in the x and y dimensions.
    ///
    /// The size on the z-axis is not constant but scales logarithmic as nears the z_far plane.
    cluster_size: UVec2,

    /// Constants used in shaders.
    depth_factors: Vec2,
    point_light_count: u32,

    padding: [u32; 3],
}

impl LightInfo {
    fn new(dir_light: DirLight, point_light_count: u32, camera: &Camera) -> Self {
        let width = camera.surface_size.x;
        let height = camera.surface_size.y;

        let subdivisions = UVec4::new(12, 12, 24, 12 * 12 * 24);
        let cluster_size = UVec2::new(width as u32 / subdivisions.x, height as u32 / subdivisions.y);

        let depth_factors = Vec2::new(
            subdivisions.z as f32 / (camera.z_far / camera.z_near).ln(),
            subdivisions.z as f32 * camera.z_near.ln() / (camera.z_far / camera.z_near).ln(),
        );

        let padding = [0x0; 3];

        Self { subdivisions, cluster_size, depth_factors, dir_light, point_light_count, padding }
    }

    pub fn cluster_subdivisions(&self) -> UVec3 {
        self.subdivisions.truncate()
    }

    pub fn cluster_count(&self) -> u32 {
        self.subdivisions.w
    }
}

#[repr(C)]
struct LightMask {
    bits: [u32; MAX_LIGHT_COUNT.div_ceil(32)]
}

/// Pipelines and buffers used to implement clustered shading as described in the paper.
/// "clustered deferred and forward shading - Ola Olsson, Markus Billeter and Ulf Assarsson".
///
/// # The pipeline stages
///
/// Three compute shaders are used to cluster and cull the lights, two of which runs every frame.
///
/// ## Cluster build
/// 
/// This only runs once on startup and every time the window is resized. It's calculates an AABB
/// for each cluster in view space. 
///
/// ## Light update
///
/// This is the first shader to run every frame. It runs in `O(n)` time, where n is the number of
/// lights in the scene. It simply updates the light position buffer by transforming the world
/// space coordinate of each light into view space.
///
/// ## Cluster Update
///
/// This runs after the light update stage is done. It's job is to find which lights effects which
/// clusters. For now it's very naiv. It works by simply iterating through each cluster and
/// checking which lights sphere intersects with the clusters AABB. This means that it has a time
/// complexity of `O(n * k)` time where n is the number of lights and `k` the number of clusters.
///
/// This could be improved by an accelerated structure such as an BVH as descriped in the paper
/// "clustered deferred and forward shading - Ola Olsson, Markus Billeter and Ulf Assarsson".
///
/// # The different buffers
///
/// ## Light buffer
///
/// This is the data of [`LightBufferData`] and contains all the lights in the scene. For now it's
/// static meaning that all the lights are uploaded up front and can't be changed at runtime
/// without temperarely stopping rendering and uploading the new data via a staging buffer.
///
/// ## AABB buffer
///
/// This holds an [`Aabb`] for each cluster in the view fustrum. This one is created at startup and
/// recreated if the resolution changes.
///
/// ## Light mask buffer
///
/// A list of [`LightMask`] for each cluster. This is updated each frame before starting the
/// shading pass. It has one copy per frame in flight.
///
/// This buffer has a copy for each frame in flight and is updated before drawing each frame by
/// the `cluster_update` compute shader.
///
/// This is one of two main ways of doing this. Another way would be to have some kind of light
/// list where each cluster get's it's own slice of light indices. This would work as well, but
/// would take up more a lot more memory unless you start doing heuristics about how many lights
/// each cluster can have, which will show artifacts it that limit is reached.
///
/// ## Light position buffer
///
/// A list of the positions of all lights in view space. This is updated before doing light culling
/// each frame. This is simply an optimization as we could just as easily transform the positions
/// during light culling, but since light culling runs in `O(n * k)` time where n is the number
/// of lights and `k` the number of clusters, this will hopefully speed things up.
///
pub struct Lights {
    pub info: LightInfo,
    pub info_buffer: Res<Buffer>,

    pub light_buffer: Res<Buffer>,
    pub cluster_aabb_buffer: Res<Buffer>,
    pub light_pos_buffers: PerFrame<Res<Buffer>>,
    pub light_mask_buffers: PerFrame<Res<Buffer>>,

    pub light_count: u32,
    pub dir_light: DirLight,

    pub descs: PerFrame<Res<DescSet>>,

    cluster_update: Res<ComputePipeline>,
    light_update: Res<ComputePipeline>,

    build_clusters: CommandBuffer, 
}

impl Lights {
    pub fn new(
        renderer: &Renderer,
        camera_descs: &CameraDescs,
        camera: &Camera,
        dir_light: DirLight,
        lights: &[PointLight],
    ) -> Result<Self> {
        let pool = &renderer.static_pool;

        let point_light_count = lights.len() as u32;

        let info = LightInfo::new(dir_light, point_light_count, camera);
        let cluster_count = info.cluster_count() as usize;

        let info_buffer = pool.create_buffer(MemoryLocation::Gpu, &BufferInfo {
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            size: mem::size_of::<LightInfo>() as vk::DeviceSize,
        })?;

        let light_data = bytemuck::cast_slice(&lights);
        let light_buffer = pool.create_buffer(MemoryLocation::Gpu, &BufferInfo {
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            size: light_data.len() as vk::DeviceSize,
        })?;

        let cluster_aabb_buffer = pool.create_buffer(MemoryLocation::Gpu, &BufferInfo {
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            size: (cluster_count * mem::size_of::<Aabb>()) as vk::DeviceSize,
        })?;

        let light_mask_buffers = PerFrame::try_from_fn(|_| {
            pool.create_buffer(MemoryLocation::Gpu, &BufferInfo {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                size: (cluster_count * mem::size_of::<LightMask>()) as vk::DeviceSize,
            })
        })?;

        let light_pos_buffers = PerFrame::try_from_fn(|_| {
            pool.create_buffer(MemoryLocation::Gpu, &BufferInfo {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                size: mem::size_of::<[LightPos; MAX_LIGHT_COUNT]>() as vk::DeviceSize,
            })
        })?;

        let staging_pool = ResourcePool::with_block_size(renderer.device.clone(), 128, 1024 * 24);

        let light_staging = staging_pool.create_buffer(MemoryLocation::Cpu, &BufferInfo {
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            size: light_data.len() as vk::DeviceSize,
        })?;

        light_staging.get_mapped()?.fill(light_data); 

        renderer.transfer_with(|recorder| {
            recorder.update_buffer(info_buffer.clone(), &info);
            recorder.copy_buffers(light_staging.clone(), light_buffer.clone())
        })?;

        let layout = pool.create_desc_layout(&[
            DescLayoutSlot {
                binding: 0,
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                count: rendi_shader::DescCount::Single,
            },
            DescLayoutSlot {
                binding: 1,
                ty: vk::DescriptorType::STORAGE_BUFFER,
                count: rendi_shader::DescCount::Single,
            },
            DescLayoutSlot {
                binding: 2,
                ty: vk::DescriptorType::STORAGE_BUFFER,
                count: rendi_shader::DescCount::Single,
            },
            DescLayoutSlot {
                binding: 3,
                ty: vk::DescriptorType::STORAGE_BUFFER,
                count: rendi_shader::DescCount::Single,
            },
            DescLayoutSlot {
                binding: 4,
                ty: vk::DescriptorType::STORAGE_BUFFER,
                count: rendi_shader::DescCount::Single,
            },
        ])?;

        let descs = PerFrame::try_from_fn(|frame_index| {
            pool.create_desc_set(layout.clone(), &[
                DescBinding::Buffer(info_buffer.clone()),
                DescBinding::Buffer(cluster_aabb_buffer.clone()),
                DescBinding::Buffer(light_buffer.clone()),
                DescBinding::Buffer(light_pos_buffers[frame_index].clone()),
                DescBinding::Buffer(light_mask_buffers[frame_index].clone()),
            ])
        })?;

        let cluster_build = {
            let code = include_bytes_aligned_as!(u32, "../assets/shaders/cluster_build.comp.spv");

            let shader = pool.create_shader_module("main", code)?;
            let prog = pool.create_compute_prog(shader)?;

            pool.create_compute_pipeline(prog, &[])?
        };

        let light_update = {
            let code = include_bytes_aligned_as!(u32, "../assets/shaders/light_update.comp.spv");

            let shader = pool.create_shader_module("main", code)?;
            let prog = pool.create_compute_prog(shader)?;

            pool.create_compute_pipeline(prog, &[])?
        };

        let cluster_update = {
            let code = include_bytes_aligned_as!(u32, "../assets/shaders/cluster_update.comp.spv");

            let shader = pool.create_shader_module("main", code)?;
            let prog = pool.create_compute_prog(shader)?;

            pool.create_compute_pipeline(prog, &[])?
        };

        let light_count = lights.len() as u32;
        let build_clusters = CommandBuffer::new(renderer.device.clone(), renderer.transfer_queue())?;

        build_clusters.record(SubmitCount::Multiple, |recorder| {
            recorder.bind_descs(&DescBindInfo {
                bind_point: vk::PipelineBindPoint::COMPUTE,
                layout: cluster_build.layout(),
                descs: &[
                    camera_descs.descs.any().clone(),
                    descs.any().clone()
                ],
            });

            let subdivisions = info.cluster_subdivisions();
            recorder.dispatch(cluster_build.clone(), subdivisions.into());
        })?;

        let lights = Self {
            info,
            info_buffer,
            build_clusters,
            light_buffer,
            cluster_aabb_buffer,
            light_pos_buffers,
            light_mask_buffers,
            light_count,
            light_update,
            cluster_update,
            dir_light,
            descs,
        };

        lights.build_clusters()?;

        Ok(lights)
    }

    fn build_clusters(&self) -> Result<()> {
        self.build_clusters.submit_wait_idle()
    }

    pub fn handle_resize(&mut self, renderer: &Renderer, camera: &Camera) -> Result<()> {
        self.info = LightInfo::new(self.dir_light, self.light_count, camera);
        renderer.transfer_with(|recorder| {
            recorder.update_buffer(self.info_buffer.clone(), &self.info);
        })?;

        self.build_clusters()
    }
}

pub fn prepare_lights(
    lights: &Lights,
    camera_descs: &CameraDescs,
    index: FrameIndex,
    recorder: &CommandRecorder,
) {
    recorder.bind_descs(&DescBindInfo {
        bind_point: vk::PipelineBindPoint::COMPUTE,
        layout: lights.light_update.layout(),
        descs: &[
            camera_descs.descs[index].clone(),
            lights.descs[index].clone(),
        ],
    });

    recorder.dispatch(lights.light_update.clone(), [
        lights.light_count.div_ceil(64), 1, 1,
    ]);

    recorder.buffer_barrier(&BufferBarrierInfo {
        buffer: lights.light_pos_buffers[index].clone(),
        src_mask: vk::AccessFlags2::SHADER_WRITE,
        dst_mask: vk::AccessFlags2::SHADER_READ,
        src_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
    });

    let group_count = lights.info.cluster_subdivisions();
    recorder.dispatch(lights.cluster_update.clone(), group_count.into());

    recorder.buffer_barrier(&BufferBarrierInfo {
        buffer: lights.light_mask_buffers[index].clone(),
        src_mask: vk::AccessFlags2::SHADER_WRITE,
        dst_mask: vk::AccessFlags2::SHADER_READ,
        src_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        dst_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
    });
}

const MAX_LIGHT_COUNT: usize = 256;
