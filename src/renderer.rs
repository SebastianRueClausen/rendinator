use std::iter;
use std::rc::Rc;
use std::time::Duration;

use winit::{dpi::PhysicalSize, window::Window};

use crate::asset;
use crate::atmosphere::AtmospherePhase;
use crate::bloom::BloomPhase;
use crate::camera::Camera;
use crate::context::Context;
use crate::depth_reduce::DepthReducePhase;
use crate::display::DisplayPhase;
use crate::resources::{
    ConstState, Consts, DepthPyramid, RenderState, SceneState, ShadowCascades, Skybox,
};
use crate::shade::ShadePhase;
use crate::shadow::ShadowPhase;
use crate::temporal_resolve::TemporalResolvePhase;
use crate::visibility::VisiblityPhase;

pub struct Renderer {
    context: Context,
    atmosphere_phase: AtmospherePhase,
    depth_reduce_phase: DepthReducePhase,
    shadow_phase: ShadowPhase,
    visibility_phase: VisiblityPhase,
    render_phase: ShadePhase,
    display_phase: DisplayPhase,
    bloom_phase: BloomPhase,
    temporal_resolve_phase: TemporalResolvePhase,
    const_state: ConstState,
    shadow_cascades: ShadowCascades,
    render_state: RenderState,
    scene_state: SceneState,
    depth_pyramid: DepthPyramid,
    skybox: Skybox,
    consts: Option<Consts>,
}

impl Renderer {
    pub fn new(window: Rc<Window>, scene: &asset::Scene) -> Self {
        let mut context = Context::new(window);

        let const_state = ConstState::new(&context);
        let render_state = RenderState::new(&context);
        let scene_state = SceneState::new(&context, scene);
        let shadow_cascades = ShadowCascades::new(&context);
        let depth_pyramid = DepthPyramid::new(&context);
        let skybox = Skybox::new(&context);

        let atmosphere_phase = AtmospherePhase::new(&mut context, &skybox);
        let depth_reduce_phase = DepthReducePhase::new(&mut context, &render_state, &depth_pyramid);
        let shadow_phase =
            ShadowPhase::new(&mut context, &scene_state, &shadow_cascades, &depth_pyramid);
        let visibility_phase = VisiblityPhase::new(&mut context, &scene_state);
        let render_phase = ShadePhase::new(
            &mut context,
            &scene_state,
            &render_state,
            &shadow_cascades,
            &skybox,
        );
        let temporal_resolve_phase = TemporalResolvePhase::new(&mut context, &render_state);
        let bloom_phase = BloomPhase::new(&mut context, &render_state);

        let display_format = context.surface_format;
        let display_phase =
            DisplayPhase::new(&mut context, display_format, &render_state.post.view);

        Self {
            context,
            const_state,
            atmosphere_phase,
            depth_reduce_phase,
            shadow_phase,
            visibility_phase,
            render_state,
            depth_pyramid,
            skybox,
            temporal_resolve_phase,
            bloom_phase,
            scene_state,
            shadow_cascades,
            render_phase,
            display_phase,
            consts: None,
        }
    }

    pub fn draw(
        &mut self,
        delta_time: Duration,
        camera: &Camera,
    ) -> Result<(), wgpu::SurfaceError> {
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

        if consts.frame_index == 0 {
            self.atmosphere_phase
                .record(&self.const_state, &mut encoder);
        }

        self.visibility_phase.record(
            &self.const_state,
            &self.render_state,
            &self.scene_state,
            camera,
            &mut encoder,
        );

        self.depth_reduce_phase
            .record(&self.depth_pyramid, &self.const_state, &mut encoder);

        self.shadow_phase.record(
            &self.const_state,
            &self.shadow_cascades,
            &self.scene_state,
            &mut encoder,
        );

        self.render_phase.record(
            &self.context,
            camera,
            &self.const_state,
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

        self.display_phase
            .record(&self.context, delta_time, &frame_buffer, &mut encoder);

        self.context.queue.submit(iter::once(encoder.finish()));
        surface_texture.present();

        Ok(())
    }

    pub fn resize_surface(&mut self, size: PhysicalSize<u32>) {
        self.context.resize_surface(size);
        self.render_state = RenderState::new(&self.context);
        self.depth_pyramid = DepthPyramid::new(&self.context);

        self.depth_reduce_phase.rezize_surface(
            &self.context,
            &self.render_state,
            &self.depth_pyramid,
        );
        self.shadow_phase
            .resize_surface(&self.context, &self.shadow_cascades, &self.depth_pyramid);
        self.render_phase.resize_surface(
            &self.context,
            &self.render_state,
            &self.shadow_cascades,
            &self.skybox,
        );
        self.temporal_resolve_phase
            .resize_surface(&self.context, &self.render_state);
        self.bloom_phase
            .resize_surface(&self.context, &self.render_state);
        self.display_phase
            .resize_surface(&self.context, &self.render_state.post.view);
    }
}
