use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::{mem, thread};

use eyre::Result;
use raw_window_handle::HasRawDisplayHandle;
use render::{FrameRequest, GuiRequest, Renderer};
use winit::event::{Event, WindowEvent};
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
            Event::MainEventsCleared => {
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
