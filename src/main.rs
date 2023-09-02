mod asset;
mod atmosphere;
mod bloom;
mod camera;
mod context;
mod depth_reduce;
mod display;
mod renderer;
mod resources;
mod shade;
mod shadow;
mod temporal_resolve;
mod util;
mod visibility;

use asset::{AssetPath, Scene};
use bit_set::BitSet;
use glam::Vec2;
use winit::dpi::PhysicalSize;
use winit::event::{
    ElementState, Event, KeyboardInput, ModifiersState, VirtualKeyCode, WindowEvent,
};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use std::rc::Rc;
use std::time::{Duration, Instant};

use camera::{Camera, CameraDelta};
use renderer::Renderer;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = Rc::new(
        WindowBuilder::new()
            .with_inner_size(PhysicalSize {
                width: 800,
                height: 800,
            })
            .build(&event_loop)
            .expect("failed to create window"),
    );

    let mut state = State::new(aspect_ratio(window.inner_size()));

    let mut renderer = {
        let scene = Scene::from_gltf(&AssetPath::new("sponza/Sponza.gltf", "sponza/Sponza.scene"))
            .expect("failed to load scene");

        Renderer::new(window.clone(), &scene)
    };

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            WindowEvent::Resized(size) => {
                state.camera.resize_proj(aspect_ratio(*size));
                renderer.resize_surface(*size);
            }
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                state.camera.resize_proj(aspect_ratio(**new_inner_size));
                renderer.resize_surface(**new_inner_size);
            }
            WindowEvent::KeyboardInput { input, .. } => {
                let Some(key) = input.virtual_keycode else {
                    return;
                };

                match input.state {
                    ElementState::Pressed => state.inputs.key_pressed(key),
                    ElementState::Released => state.inputs.key_released(key),
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.inputs.mouse_moved(Vec2 {
                    x: position.x as f32,
                    y: position.y as f32,
                });
            }
            WindowEvent::ModifiersChanged(modifier_state) => {
                state.inputs.modifier_change(*modifier_state);
            }
            _ => (),
        },
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            let delta_time = state.last_update.elapsed();
            state.last_update = Instant::now();

            state.handle_inputs(delta_time);

            if let Err(err) = renderer.draw(delta_time, &state.camera) {
                match err {
                    wgpu::SurfaceError::Lost => {
                        state.camera.resize_proj(aspect_ratio(window.inner_size()));
                    }
                    wgpu::SurfaceError::OutOfMemory => *control_flow = ControlFlow::Exit,
                    err => eprintln!("{err:?}"),
                }
            }
        }
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        _ => (),
    })
}

fn aspect_ratio(size: PhysicalSize<u32>) -> f32 {
    size.width as f32 / size.height as f32
}

struct State {
    inputs: Inputs,
    camera: Camera,
    last_update: Instant,
}

impl State {
    fn new(aspect_ratio: f32) -> Self {
        Self {
            inputs: Inputs::default(),
            camera: Camera::new(aspect_ratio),
            last_update: Instant::now(),
        }
    }

    fn handle_inputs(&mut self, delta_time: Duration) {
        let mut camera_delta = CameraDelta::default();
        let delta_time = delta_time.as_secs_f32();

        let move_speed = 20.0;
        let mouse_sensitivity = 0.5;

        if self.inputs.is_key_pressed(VirtualKeyCode::W) {
            camera_delta.forward += move_speed * delta_time;
        }

        if self.inputs.is_key_pressed(VirtualKeyCode::S) {
            camera_delta.backward += move_speed * delta_time;
        }

        if self.inputs.is_key_pressed(VirtualKeyCode::A) {
            camera_delta.left += move_speed * delta_time;
        }

        if self.inputs.is_key_pressed(VirtualKeyCode::D) {
            camera_delta.right += move_speed * delta_time;
        }

        let mouse_delta = self.inputs.take_mouse_delta();

        if self.inputs.is_shift_pressed() {
            camera_delta.yaw += mouse_sensitivity * mouse_delta.x;
            camera_delta.pitch += mouse_sensitivity * mouse_delta.y;

            self.camera.move_by_delta(camera_delta);
        }
    }
}

#[derive(Default)]
struct Inputs {
    keys_pressed: BitSet,
    modifier_state: ModifiersState,
    mouse_position: Option<Vec2>,
    mouse_delta: Option<Vec2>,
}

impl Inputs {
    fn mouse_moved(&mut self, to: Vec2) {
        let position = self.mouse_position.unwrap_or(to);
        let delta = self.mouse_delta.unwrap_or_default();

        self.mouse_delta = Some(delta + (position - to));
        self.mouse_position = Some(to);
    }

    fn key_pressed(&mut self, key: VirtualKeyCode) {
        self.keys_pressed.insert(key as usize);
    }

    fn key_released(&mut self, key: VirtualKeyCode) {
        self.keys_pressed.remove(key as usize);
    }

    fn modifier_change(&mut self, modifier_state: ModifiersState) {
        self.modifier_state = modifier_state;
    }

    fn is_key_pressed(&self, key: VirtualKeyCode) -> bool {
        self.keys_pressed.contains(key as usize)
    }

    fn take_mouse_delta(&mut self) -> Vec2 {
        self.mouse_delta.take().unwrap_or_default()
    }

    fn is_shift_pressed(&self) -> bool {
        self.modifier_state.shift()
    }
}
