use std::{array, borrow::Cow, iter, mem};

use glam::{Mat4, Vec4, Vec4Swizzles};

use crate::{
    camera::Camera,
    context::Context,
    resources::{
        Consts, SceneState, ShadowCascade, ShadowCascades, SHADOW_CASCADE_COUNT,
        SHADOW_CASCADE_FORMAT,
    },
};

pub struct ShadowPhase {
    render_cascade: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl ShadowPhase {
    pub fn new(
        context: &mut Context,
        scene_state: &SceneState,
        shadow_cascades: &ShadowCascades,
    ) -> Self {
        let cascade_render_module = context.create_shader_module(
            include_str!("shaders/shadow_render.wgsl"),
            "shaders/shadow_render.wgsl",
            &[],
        );

        let cascade_render_shader =
            context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("cascade render"),
                    source: wgpu::ShaderSource::Naga(Cow::Owned(cascade_render_module)),
                });

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("shadow"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let bind_group = create_bind_group(context, &shadow_cascades, &bind_group_layout);

        let render_cascade_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("render cascade"),
                    bind_group_layouts: &[&bind_group_layout, &scene_state.bind_group_layout],
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX,
                        range: 0..mem::size_of::<u32>() as u32,
                    }],
                });

        let render_cascade =
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("render cascade"),
                    layout: Some(&render_cascade_layout),
                    vertex: wgpu::VertexState {
                        module: &cascade_render_shader,
                        entry_point: "vertex",
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: SHADOW_CASCADE_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    fragment: None,
                    multiview: None,
                });

        Self {
            bind_group,
            bind_group_layout,
            render_cascade,
        }
    }

    pub fn resize_surface(&mut self, context: &Context, shadow_cascades: &ShadowCascades) {
        self.bind_group = create_bind_group(context, &shadow_cascades, &self.bind_group_layout);
    }

    pub fn record(
        &self,
        shadow_cascades: &ShadowCascades,
        scene_state: &SceneState,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        for (index, cascade) in shadow_cascades.cascades.iter().enumerate() {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render cascade"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: cascade,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(&self.render_cascade);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_bind_group(1, &scene_state.bind_group, &[]);

            let index = index as u32;
            render_pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, &index.to_le_bytes());

            for (index, draw) in scene_state.primitive_draw_infos.iter().enumerate() {
                let index = index as u32;
                render_pass.draw(draw.indices.clone(), index..index + 1);
            }
        }
    }
}

fn create_bind_group(
    context: &Context,
    shadow_cascades: &ShadowCascades,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &shadow_cascades.cascade_info,
                    offset: 0,
                    size: None,
                }),
            }],
        })
}

pub fn create_cascades(camera: &Camera, consts: &Consts) -> Vec<ShadowCascade> {
    let lambda = 0.6;

    let mut cascades: Vec<_> = iter::repeat(ShadowCascade::default())
        .take(SHADOW_CASCADE_COUNT)
        .collect();

    let frustrum_corners: [_; 8] = array::from_fn(|index| {
        let z = index / 4;

        let index = index - z * 4;
        let y = index / 2;
        let x = index % 2;

        let value = consts.inverse_proj_view
            * Vec4::new(
                2.0 * x as f32 - 1.0,
                2.0 * y as f32 - 1.0,
                2.0 * z as f32 - 1.0,
                1.0,
            );

        value / value.w
    });

    let mut center: Vec4 = frustrum_corners.iter().sum();
    center /= 8.0;

    let radius = frustrum_corners
        .iter()
        .fold(0.0, |radius, corner| corner.distance(center).max(radius));

    for cascade in &mut cascades {
        let light_pos = center.xyz() + consts.sun.direction.xyz() * radius;
        let focal_point = center.xyz();

        let view = Mat4::look_at_rh(light_pos, focal_point, Camera::UP);

        let (min, max) =
            frustrum_corners
                .into_iter()
                .fold((Vec4::MAX, Vec4::MIN), |(min, max), corner| {
                    let corner = view * corner;
                    (corner.min(min), corner.max(max))
                });

        let proj = Mat4::orthographic_rh(min.x, max.x, min.y, max.y, 0.0, camera.z_far - camera.z_near);
        cascade.proj_view = proj * view;
    }

    /*
    let depth_range = camera.z_far - camera.z_near;

    let min_depth = camera.z_near;
    let max_depth = min_depth + depth_range;

    let depth_ratio = max_depth / min_depth;

    for (index, cascade) in cascades.iter_mut().enumerate() {
        let p = (index as f32 + 1.0) / SHADOW_CASCADE_COUNT as f32;
        let log_depth = min_depth * depth_ratio.powf(p);

        let uniform = min_depth + depth_range * p;
        let depth = lambda * (log_depth - uniform) + uniform;

        cascade.split = (depth - camera.z_near) / depth_range;
    }

    let frustrum_corners = frustrum_corners(consts.inverse_proj_view);
    let mut center: Vec4 = frustrum_corners
        .iter()
        .sum();
    center /= 8.0;

    for cascade in &mut cascades {
        let mut frustrum_corners = frustrum_corners;

        for index in 0..4 {
            let dist = frustrum_corners[i + 4] - frustrum_corners[i]
        }

        let eye = center - consts.sun.direction;
        let view = Mat4::look_at_lh(eye.xyz(), center.xyz(), Camera::UP);

        let (min, max) = frustrum_corners
            .iter()
            .fold((Vec4::MAX, Vec4::MIN), |(min, max), corner| {
                let corner = view * *corner;
                (corner.max(max), corner.min(min))
            });

        let proj = Mat4::orthographic_lh(min.x, max.x, min.y, max.y, min.z, max.z);

        cascade.split_depth = camera.z_near + cascade.split * depth_range;
        cascade.proj_view = proj * view;
    }
    */

    cascades
}
