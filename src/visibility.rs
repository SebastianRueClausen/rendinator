use std::borrow::Cow;

use glam::Vec4;

use crate::{
    asset,
    camera::{self, Camera},
    context::Context,
    resources::{self, ConstState, RenderState, SceneState},
};

pub struct VisiblityPhase {
    visibility: wgpu::RenderPipeline,
}

impl VisiblityPhase {
    pub fn new(context: &mut Context, scene_state: &SceneState) -> Self {
        let visiblity_module = context.create_shader_module(
            include_str!("shaders/visibility.wgsl"),
            "shaders/visibility.wgsl",
            &[],
        );

        let visibility_shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("visibility"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(visiblity_module)),
            });

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("render"),
                    push_constant_ranges: &[],
                    bind_group_layouts: &[
                        ConstState::bind_group_layout(&context),
                        &scene_state.bind_group_layout,
                    ],
                });

        let visibility = context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("render"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    entry_point: "vertex",
                    module: &visibility_shader,
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    entry_point: "fragment",
                    module: &visibility_shader,
                    targets: &[Some(wgpu::ColorTargetState {
                        format: resources::VISIBILITY_BUFFER_FORMAT,
                        write_mask: wgpu::ColorWrites::ALL,
                        blend: None,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: resources::DEPTH_BUFFER_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        Self { visibility }
    }

    pub fn record(
        &self,
        const_state: &ConstState,
        render_state: &RenderState,
        scene_state: &SceneState,
        camera: &Camera,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        encoder.clear_texture(
            &render_state.visibility.texture,
            &wgpu::ImageSubresourceRange {
                aspect: wgpu::TextureAspect::All,
                ..Default::default()
            },
        );

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("visibility pass"),
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &render_state.depth.view,
                stencil_ops: None,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
            }),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &render_state.visibility.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            })],
        });

        render_pass.set_pipeline(&self.visibility);
        render_pass.set_bind_group(0, &const_state.bind_group, &[]);
        render_pass.set_bind_group(1, &scene_state.bind_group, &[]);

        let frustrum = camera.frustrum();
        for (index, draw) in scene_state.primitive_draw_infos.iter().enumerate() {
            if sphere_inside_frustrum(&draw.bounding_sphere, &frustrum) {
                let index = index as u32;
                render_pass.draw(draw.indices.clone(), index..index + 1);
            }
        }
    }
}

fn sphere_inside_frustrum(sphere: &asset::BoundingSphere, frustrum: &camera::Frustrum) -> bool {
    let signed_distance = |plane: Vec4| -> f32 {
        plane.x * sphere.center.x + plane.y * sphere.center.y + plane.z * sphere.center.z + plane.w
            - sphere.radius
    };

    let outside_frustrum = signed_distance(frustrum.left) > 0.0
        || signed_distance(frustrum.right) > 0.0
        || signed_distance(frustrum.bottom) > 0.0
        || signed_distance(frustrum.top) > 0.0
        || signed_distance(frustrum.far) > 0.0
        || signed_distance(frustrum.near) > 0.0;

    !outside_frustrum
}
