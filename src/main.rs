#![feature(let_else, int_roundings, array_try_from_fn)]

#[macro_use]
extern crate log;

#[macro_use]
extern crate anyhow;

#[macro_use]
mod macros;

mod resource;
mod core;
mod scene;
mod skybox;
mod light;
mod text;
mod camera;

use anyhow::Result;
use glam::Vec3;
use winit::event::VirtualKeyCode;

use std::time::Instant;
use std::path::Path;

use crate::core::*;
use crate::text::TextPass;
use crate::scene::Scene;
use crate::light::{Lights, PointLight};
use crate::camera::{Camera, CameraUniforms};
use crate::resource::ResourcePool;
use crate::skybox::Skybox;

fn main() -> Result<()> {
    env_logger::init();

    use winit::event::{Event, WindowEvent, ElementState};
    use winit::event_loop::ControlFlow;

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop)?;

    window.set_cursor_grab(false)?;
    window.set_cursor_visible(false);

    let mut minimized = false;

    let mut last_update = Instant::now();
    let mut last_draw = Instant::now();

    let mut renderer = Renderer::new(&window)?;
    let mut input_state = InputState::default();

    let resource_pool = ResourcePool::new();

    let font = asset::Font::load(Path::new("assets/fonts/source_code_pro.font"))?;
    let mut text_pass = TextPass::new(&renderer, &resource_pool, &font)?;

    let mut camera = Camera::new(renderer.swapchain.aspect_ratio());
    let camera_uniforms = CameraUniforms::new(&renderer, &resource_pool, &camera)?;

    let lights = debug_lights();
    let mut lights = Lights::new(&renderer, &resource_pool, &camera_uniforms, &camera, &lights)?;

    let skybox = Skybox::new(&renderer, &lights, &resource_pool)?;

    let scene = asset::Scene::load(Path::new("assets/scenes/sponza.scene"))?;
    let scene = Scene::from_scene_asset(
        &renderer,
        &resource_pool,
        &camera_uniforms,
        &skybox,
        &lights,
        &scene,
    )?;

    event_loop.run(move |event, _, controlflow| match event {
        Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
            *controlflow = ControlFlow::Exit;
        }
        Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
            if let Some(key) = input.virtual_keycode {
                match input.state {
                    ElementState::Pressed => input_state.key_pressed(key),
                    ElementState::Released => input_state.key_released(key),
                }
            }
        }
        Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
            input_state.mouse_moved((position.x, position.y));
        }
        Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
            if size.width == 0 && size.height == 0 {
                minimized = true
            } else {
                minimized = false;
                renderer.resize(&window).expect("failed to resize window");

                text_pass.handle_resize(&renderer);
                camera.update_proj(renderer.swapchain.aspect_ratio());

                camera_uniforms
                    .update_proj(&renderer, &camera)
                    .expect("failed to update projection");

                lights
                    .handle_resize(&renderer, &camera_uniforms, &camera)
                    .expect("failed to resize lights");
            }
        }
        Event::MainEventsCleared => {
            camera.update(&mut input_state, last_update.elapsed());
            last_update = Instant::now();

            if !minimized {
                let elapsed = last_draw.elapsed();
                last_draw = Instant::now();

                let res = renderer.draw(
                    |recorder, frame_index| {
                        camera_uniforms
                            .update_view(frame_index, &camera)
                            .expect("failed to update view");

                        lights.prepare_lights(frame_index, &camera_uniforms, recorder);
                        scene.prepare_draw_buffers(frame_index, &camera, recorder);
                    },
                    |recorder, frame_index| {
                        scene.draw(frame_index, &camera_uniforms, &lights, recorder);
                        skybox.draw(&camera, frame_index, recorder); 

                        text_pass.draw_text(recorder, frame_index, |texts| {
                            let fps = format!("fps: {}", 1.0 / elapsed.as_secs_f64());
                            let pos = format!(
                                "position: ({}, {}, {})",
                                camera.pos.x,
                                camera.pos.y,
                                camera.pos.z,
                            );

                            texts.add_label(30.0, Vec3::new(20.0, 20.0, 0.5), &fps); 
                            texts.add_label(30.0, Vec3::new(20.0, 60.0, 0.5), &pos);
                        })
                        .expect("failed do draw text");
                    },
                );

                res.expect("failed rendering");
            }
        }
        _ => {}
    });
}

#[derive(Default)]
pub struct InputState {
    /// Keeps track of if each `VirtualKeyCode` is pressed or not. Each key code represents a
    /// single bit.
    key_pressed: [u64; 3],

    /// The current position of the mouse. `None` if no `mouse_moved` event has been received.
    mouse_pos: Option<(f64, f64)>,

    /// Contains the mouse position delta since last time `mouse_delta`.
    mouse_delta: Option<(f64, f64)>,
}

impl InputState {
    pub fn mouse_moved(&mut self, pos: (f64, f64)) {
        let mouse_pos = self.mouse_pos.unwrap_or(pos);
        let mouse_delta = self.mouse_delta.unwrap_or((0.0, 0.0));

        self.mouse_delta = Some((
            mouse_delta.0 + (mouse_pos.0 - pos.0),
            mouse_delta.1 + (mouse_pos.1 - pos.1),
        ));

        self.mouse_pos = Some(pos);   
    }

    pub fn key_pressed(&mut self, key: VirtualKeyCode) {
        let major = key as usize / 64;
        let minor = key as usize % 64;

        self.key_pressed[major] |= 1 << minor;
    }

    pub fn key_released(&mut self, key: VirtualKeyCode) {
        let major = key as usize / 64;
        let minor = key as usize % 64;
    
        self.key_pressed[major] &= !(1 << minor);
    }

    pub fn is_key_pressed(&self, key: VirtualKeyCode) -> bool {
        let major = key as usize / 64;
        let minor = key as usize % 64;
   
        self.key_pressed[major] & (1 << minor) != 0
    }

    pub fn mouse_delta(&mut self) -> (f64, f64) {
        self.mouse_delta.take().unwrap_or((0.0, 0.0))
    }
}

fn debug_lights() -> Vec<PointLight> {
    let mut lights = Vec::default();

    for i in 0..20 {
        let red = (i % 2) as f32;
        let blue = ((i + 1) % 2) as f32;

        let start = Vec3::new(-16.0, -3.0, -8.0);
        let end = Vec3::new(15.0, 13.0, 8.0);

        let position = start.lerp(end, i as f32 / 20.0);

        lights.push(PointLight::new(
            position,
            Vec3::new(red, 1.0, blue) * 6.0,
            8.0,
        ));
    }

    lights 
}
