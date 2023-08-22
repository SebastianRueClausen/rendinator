use std::iter;
use std::rc::Rc;

use winit::{dpi::PhysicalSize, window::Window};

use crate::asset;
use crate::bloom::BloomPhase;
use crate::camera::Camera;
use crate::context::Context;
use crate::display::BlendPhase;
use crate::render::RenderPhase;
use crate::resources::{ConstState, Consts, RenderState, SceneState};
use crate::temporal_resolve::TemporalResolvePhase;

pub struct Renderer {
    context: Context,
    render_phase: RenderPhase,
    display_phase: BlendPhase,
    bloom_phase: BloomPhase,
    temporal_resolve_phase: TemporalResolvePhase,
    const_state: ConstState,
    render_state: RenderState,
    scene_state: SceneState,
    consts: Option<Consts>,
}

impl Renderer {
    pub fn new(window: Rc<Window>, scene: &asset::Scene) -> Self {
        let mut context = Context::new(window);

        let const_state = ConstState::new(&context);
        let render_state = RenderState::new(&context);
        let scene_state = SceneState::new(&context, scene);

        let render_phase = RenderPhase::new(&mut context, &scene_state, &render_state);
        let temporal_resolve_phase = TemporalResolvePhase::new(&mut context, &render_state);
        let bloom_phase = BloomPhase::new(&mut context, &render_state);

        let display_format = context.surface_format;
        let display_phase = BlendPhase::new(&mut context, display_format, &render_state.post.view);

        Self {
            context,
            const_state,
            render_state,
            temporal_resolve_phase,
            bloom_phase,
            scene_state,
            render_phase,
            display_phase,
            consts: None,
        }
    }

    pub fn draw(&mut self, camera: &Camera) -> Result<(), wgpu::SurfaceError> {
        let consts = Consts::new(camera, &self.context, self.consts.take());
        self.consts = Some(consts);

        let bytes = bytemuck::bytes_of(&consts);
        self.context
            .queue
            .write_buffer(&self.const_state.const_buffer, 0, bytes);

        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("main encoder"),
                });

        self.render_phase.record(
            &self.context,
            camera,
            &self.const_state,
            &self.render_state,
            &self.scene_state,
            &mut encoder,
        );

        self.temporal_resolve_phase.record(
            &self.context,
            &consts,
            &self.const_state,
            &self.render_state,
            &mut encoder,
        );

        self.bloom_phase
            .record(&self.const_state, &self.render_state, &mut encoder);

        let surface_texture = self.context.surface.get_current_texture()?;
        let frame_buffer = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("frame buffer"),
                ..Default::default()
            });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("final"),
            depth_stencil_attachment: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame_buffer,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            })],
        });

        self.display_phase
            .record(wgpu::Color::WHITE, &mut render_pass);
        drop(render_pass);

        self.context.queue.submit(iter::once(encoder.finish()));
        surface_texture.present();

        Ok(())
    }

    pub fn resize_surface(&mut self, size: PhysicalSize<u32>) {
        self.context.resize_surface(size);
        self.render_state = RenderState::new(&self.context);

        self.render_phase
            .resize_surface(&self.context, &self.render_state);
        self.temporal_resolve_phase
            .resize_surface(&self.context, &self.render_state);
        self.bloom_phase
            .resize_surface(&self.context, &self.render_state);
        self.display_phase
            .change_input(&self.context, &self.render_state.post.view);
    }
}
