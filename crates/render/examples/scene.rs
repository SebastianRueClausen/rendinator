use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{mem, thread};

use bit_set::BitSet;
use eyre::Result;
use glam::Vec2;
use raw_window_handle::HasRawDisplayHandle;
use render::{CameraMove, FrameRequest, GuiRequest, Renderer};
use winit::event::{
    ElementState, Event, ModifiersState, VirtualKeyCode, WindowEvent,
};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;

#[derive(Debug, Clone)]
struct ScenePath {
    gltf: PathBuf,
    cached: PathBuf,
}

impl ScenePath {
    fn new(gltf: impl AsRef<Path>) -> Self {
        let gltf = gltf.as_ref();
        Self { gltf: gltf.to_path_buf(), cached: gltf.with_extension("scene") }
    }
}

#[derive(Debug, Default)]
enum SceneState {
    #[default]
    Empty,
    Loaded {
        path: ScenePath,
    },
    Loading {
        path: ScenePath,
        progress: Arc<Mutex<asset_import::Progress>>,
        thread: thread::JoinHandle<Result<asset::Scene>>,
    },
}

impl SceneState {
    fn start_loading(&mut self, path: ScenePath) {
        let progress = Arc::new(Mutex::new(asset_import::Progress::default()));
        *self = SceneState::Loading {
            path: path.clone(),
            progress: progress.clone(),
            thread: thread::spawn(move || {
                if let Ok(scene) = asset::Scene::deserialize(&path.cached) {
                    Ok(scene)
                } else {
                    let scene =
                        asset_import::load_gltf(&path.gltf, Some(progress))?;
                    if let Err(error) = scene.serialize(&path.cached) {
                        eprintln!(
                            "failed to store cached scene to {:?}: {}",
                            path.cached, error
                        );
                    } else {
                        println!("cached scene to {:?}", path.cached);
                    }
                    Ok(scene)
                }
            }),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();

    let mut scene = asset::Scene::default();
    let mut renderer = Some(create_renderer(&window, &scene));
    let mut gui = Gui::new(&window);
    let mut scene_state = SceneState::default();
    let mut inputs = Inputs::default();
    let mut last_update = Instant::now();

    event_loop.run(move |event, _, ctrl| {
        ctrl.set_poll();
        if let Event::WindowEvent { event, .. } = &event {
            if gui.handle_input(event) {
                return;
            }
        }
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested, ..
            } => {
                *ctrl = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size), ..
            } => {
                gui.handle_resize();
                if size.width == 0 && size.height == 0 {
                    renderer.take();
                } else {
                    renderer.take();
                    renderer = Some(create_renderer(&window, &scene));
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                inputs.mouse_moved(Vec2 {
                    x: position.x as f32,
                    y: position.y as f32,
                });
            }
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                ..
            } => {
                let Some(key) = input.virtual_keycode else {
                    return;
                };
                match input.state {
                    ElementState::Pressed => inputs.key_pressed(key),
                    ElementState::Released => inputs.key_released(key),
                }
            }
            Event::WindowEvent {
                event: WindowEvent::ModifiersChanged(modifiers),
                ..
            } => {
                inputs.modifier_change(modifiers);
            }
            Event::MainEventsCleared => {
                let dt = last_update.elapsed();
                last_update = Instant::now();

                scene_state = match mem::take(&mut scene_state) {
                    SceneState::Loading { thread, path, progress } => {
                        if thread.is_finished() {
                            scene = thread
                                .join()
                                .unwrap()
                                .expect("failed to load scene");
                            if let Some(renderer) = &mut renderer {
                                renderer
                                    .change_scene(&scene)
                                    .expect("failed to set scene");
                            }
                            SceneState::Loaded { path: path.clone() }
                        } else {
                            SceneState::Loading { thread, path, progress }
                        }
                    }
                    scene_state => scene_state,
                };

                if let Some(renderer) = &mut renderer {
                    renderer
                        .render_frame(&FrameRequest {
                            gui: gui.render(&window, &mut scene_state),
                            camera_move: inputs.camera_move(dt),
                        })
                        .expect("failed to render frame");
                }
                window.request_redraw();
            }
            _ => (),
        }
    });
}

struct Gui {
    scene_path: String,
    context: egui::Context,
    input_state: egui_winit::State,
}

impl Gui {
    fn new(window: &winit::window::Window) -> Self {
        Self {
            context: egui::Context::default(),
            input_state: egui_winit::State::new(&window),
            scene_path: String::default(),
        }
    }

    fn render(
        &mut self,
        window: &winit::window::Window,
        scene_loading: &mut SceneState,
    ) -> GuiRequest {
        let input = self.input_state.take_egui_input(&window);
        let output = self.context.run(input, |ctx| {
            egui::Window::new("scene").show(&ctx, |ui| {
                ui.horizontal(|ui| {
                    let label = ui.label("scene path");
                    ui.text_edit_singleline(&mut self.scene_path)
                        .labelled_by(label.id);
                    if ui.button("load").clicked() {
                        scene_loading
                            .start_loading(ScenePath::new(&self.scene_path));
                    }
                });
                match scene_loading {
                    SceneState::Empty => ui.label("no scene loaded"),
                    SceneState::Loaded { path } => {
                        ui.label(format!("'{:?}' is loaded", path.gltf))
                    }
                    SceneState::Loading { progress, .. } => {
                        let progress = *progress.lock().unwrap();
                        let progress_bar =
                            egui::ProgressBar::new(progress.percentage)
                                .show_percentage()
                                .animate(true);
                        ui.add(progress_bar).on_hover_text(format!(
                            "loading {:?}",
                            progress.stage,
                        ))
                    }
                }
            });
        });
        GuiRequest {
            primitives: self.context.tessellate(output.shapes),
            textures_delta: output.textures_delta,
            pixels_per_point: self.context.pixels_per_point(),
        }
    }

    fn handle_input(&mut self, event: &WindowEvent) -> bool {
        self.input_state.on_event(&self.context, event).consumed
    }

    fn handle_resize(&mut self) {
        self.context = egui::Context::default();
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

    fn is_key_pressed(&self, key: VirtualKeyCode) -> bool {
        self.keys_pressed.contains(key as usize)
    }

    fn mouse_delta(&mut self) -> Vec2 {
        self.mouse_delta.take().unwrap_or(Vec2::ZERO)
    }

    fn modifier_change(&mut self, modifier_state: ModifiersState) {
        self.modifier_state = modifier_state;
    }

    fn is_shift_pressed(&self) -> bool {
        self.modifier_state.shift()
    }

    fn camera_move(&mut self, dt: Duration) -> CameraMove {
        let mut camera_move = CameraMove::default();
        let dt = dt.as_secs_f32();
        let move_speed = 50.0;
        let mouse_sensitivity = 0.05;
        if self.is_key_pressed(VirtualKeyCode::W) {
            camera_move.backward += move_speed * dt;
        }
        if self.is_key_pressed(VirtualKeyCode::S) {
            camera_move.forward += move_speed * dt;
        }
        if self.is_key_pressed(VirtualKeyCode::A) {
            camera_move.left += move_speed * dt;
        }
        if self.is_key_pressed(VirtualKeyCode::D) {
            camera_move.right += move_speed * dt;
        }
        let mouse_delta = self.mouse_delta();
        if self.is_shift_pressed() {
            camera_move.yaw += mouse_sensitivity * mouse_delta.x;
            camera_move.pitch += mouse_sensitivity * mouse_delta.y;
        }
        camera_move
    }
}

fn create_renderer(
    window: &winit::window::Window,
    scene: &asset::Scene,
) -> Renderer {
    use raw_window_handle::HasRawWindowHandle;
    Renderer::new(render::RendererRequest {
        window: window.raw_window_handle(),
        display: window.raw_display_handle(),
        width: window.inner_size().width,
        height: window.inner_size().height,
        validate: true,
        scene: &scene,
    })
    .unwrap()
}
