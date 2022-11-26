#![feature(int_roundings, array_try_from_fn)]

#[macro_use]
extern crate log;

#[macro_use]
extern crate anyhow;

#[macro_use]
mod macros;

mod resource;
mod command;
mod core;
mod scene;
mod skybox;
mod light;
mod text;
mod camera;
mod frame;
mod pass;

use ash::vk;
use anyhow::Result;
use winit::event::VirtualKeyCode;

use std::time::{Duration, Instant};
use std::path::Path;

use crate::command::*;
use crate::core::*;
use crate::text::TextPass;
use crate::scene::{ForwardPass, Scene};
use crate::light::{DirLight, PointLight};
use crate::camera::Camera;
use crate::skybox::Skybox;

use rendi_math::prelude::*;

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

    let mut camera = Camera::new(renderer.swapchain.size());

    let lights = debug_lights();
    let dir_light = DirLight::default();

    let scene: rendi_asset::Scene = rendi_asset::load(Path::new("assets/scenes/sponza.scene"))?;
    let scene = Scene::from_scene_asset(&renderer, &scene, dir_light, &lights)?;

    let mut forward_pass = ForwardPass::new(&renderer, &camera, &scene)?;

    let render_target_info = forward_pass.render_target_info();

    let skybox = Skybox::new(&renderer, render_target_info, &forward_pass.lights)?;

    let font: rendi_sdf::Atlas =
        rendi_asset::load(Path::new("assets/fonts/font.font"))?;

    let mut text_pass = TextPass::new(&renderer, render_target_info, font)?;

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

                camera.handle_resize(renderer.swapchain.size());
                text_pass.handle_resize(&renderer);

                forward_pass
                    .handle_resize(&mut renderer, &camera)
                    .expect("failed to resize forward pass");
            }
        }
        Event::MainEventsCleared => {
            update_camera(&mut camera, &mut input_state, last_update.elapsed());
            last_update = Instant::now();

            if !minimized {
                let elapsed = last_draw.elapsed();
                last_draw = Instant::now();

                let swapchain = renderer.swapchain.clone();
                let res = renderer.draw(|recorder, frame_index, swapchain_image| {
                    scene::draw(
                        &renderer,
                        &forward_pass,
                        &scene,
                        &camera,
                        frame_index,
                        recorder,
                    );

                    let render_info = RenderInfo {
                        color_target: Some(forward_pass.color_images[frame_index].clone()),
                        depth_target: forward_pass.depth_images[frame_index].clone(),
    
                        color_load_op:  vk::AttachmentLoadOp::LOAD,
                        depth_load_op:  vk::AttachmentLoadOp::LOAD,

                        swapchain: swapchain.clone(),
                    };

                    recorder.render(&render_info, |recorder| {
                        skybox::draw(&skybox, &camera, recorder);

                        text_pass.draw_text(recorder, frame_index, |texts| {
                            let fps = format!("fps: {}", 1.0 / elapsed.as_secs_f64());
                            let pos = format!(
                                "position: ({}, {}, {})",
                                camera.pos.x,
                                camera.pos.y,
                                camera.pos.z,
                            );

                            let primitives_drawn = forward_pass.draw_descs[frame_index]
                                .primitives_drawn()
                                .expect("failed to get amount of primitives drawn");

                            let primitives_drawn = format!("primitives: {}", primitives_drawn);

                            texts.add_label(0.02, Vec3::new(10.0, 50.0, 0.5), &pos);
                            texts.add_label(0.02, Vec3::new(10.0, 100.0, 0.5), &fps); 
                            texts.add_label(0.02, Vec3::new(10.0, 150.0, 0.5), &primitives_drawn);
                        })
                        .expect("failed do draw text");
                    });

                    let color_image = forward_pass.color_images[frame_index].image().clone();
                    let swapchain_image = swapchain_image.image().clone();

                    recorder.image_barrier(&ImageBarrierInfo {
                        flags: vk::DependencyFlags::BY_REGION,
                        src_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        dst_stage: vk::PipelineStageFlags2::RESOLVE,
                        src_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                        dst_mask: vk::AccessFlags2::TRANSFER_WRITE,
                        new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        image: color_image.clone(),
                        mips: color_image.mip_levels(),
                    });

                    recorder.image_barrier(&ImageBarrierInfo {
                        flags: vk::DependencyFlags::BY_REGION,
                        src_stage: vk::PipelineStageFlags2::empty(),
                        dst_stage: vk::PipelineStageFlags2::RESOLVE,
                        src_mask: vk::AccessFlags2::empty(),
                        dst_mask: vk::AccessFlags2::TRANSFER_WRITE,
                        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        image: swapchain_image.clone(),
                        mips: swapchain_image.mip_levels(),
                    });

                    recorder.resolve_image(&ImageResolveInfo {
                        src: color_image.clone(),
                        dst: swapchain_image.clone(),
                        src_mip: 0,
                        dst_mip: 0,
                    });

                    if false {
                        forward_pass.pyramid_debug(frame_index, swapchain_image.clone(), recorder, 4);
                    }

                    // Transition swapchain image to present layout.
                    recorder.image_barrier(&ImageBarrierInfo {
                        flags: vk::DependencyFlags::BY_REGION,
                        src_stage: vk::PipelineStageFlags2::RESOLVE,
                        dst_stage: vk::PipelineStageFlags2::empty(),
                        src_mask: vk::AccessFlags2::TRANSFER_WRITE,
                        dst_mask: vk::AccessFlags2::empty(),
                        new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                        image: swapchain_image.clone(),
                        mips: swapchain_image.mip_levels(),
                    });
                });

                res.expect("failed rendering");
            }
        }
        _ => {}
    });
}

fn update_camera(camera: &mut Camera, input_state: &mut InputState, dt: Duration) {
    let movement_speed = if input_state.is_key_pressed(VirtualKeyCode::LShift) {
        30.0
    } else {
        15.0
    };

    let rotation_speed = 1.0;

    let speed = movement_speed * dt.as_secs_f32();

    if input_state.is_key_pressed(VirtualKeyCode::W) {
        camera.pos += camera.front * speed;
    }

    if input_state.is_key_pressed(VirtualKeyCode::S) {
        camera.pos -= camera.front * speed;
    }

    if input_state.is_key_pressed(VirtualKeyCode::A) {
        camera.pos -= camera.front.cross(Camera::UP).normalize() * speed;
    }

    if input_state.is_key_pressed(VirtualKeyCode::D) {
        camera.pos += camera.front.cross(Camera::UP).normalize() * speed;
    }

    let (x_delta, y_delta) = input_state.mouse_delta();
   
    camera.yaw += (x_delta as f32 * rotation_speed) % 365.0;
    camera.pitch -= y_delta as f32 * rotation_speed;
    camera.pitch = camera.pitch.clamp(-89.0, 89.0);

    camera.front = -Vec3::new(
        f32::cos(camera.yaw.to_radians()) * f32::cos(camera.pitch.to_radians()),
        f32::sin(camera.pitch.to_radians()),
        f32::sin(camera.yaw.to_radians()) * f32::cos(camera.pitch.to_radians()),
    )
    .normalize();

    camera.update_view_matrix();
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

#[allow(unused)]
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
        ));
    }

    lights 
}
