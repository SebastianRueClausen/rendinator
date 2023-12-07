use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{fs, mem, thread};

use aftermath_rs as aftermath;
use asset_import::Progress;
use bit_set::BitSet;
use eyre::Result;
use glam::{Mat4, Quat, Vec2, Vec3};
use raw_window_handle::HasRawDisplayHandle;
use render::scene::NodeTree;
use render::{Camera, CameraMove, FrameRequest, GuiRequest, Renderer};
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
    Error {
        error: String,
    },
    Loading {
        thread: thread::JoinHandle<Result<asset::Scene>>,
        progress: Arc<Mutex<asset_import::Progress>>,
        path: ScenePath,
    },
}

impl SceneState {
    fn start_loading(&mut self, path: ScenePath) {
        let progress = Arc::new(Mutex::new(asset_import::Progress::default()));
        let thread = thread::spawn({
            let path = path.clone();
            let progress = progress.clone();
            move || load_scene(path, progress)
        });
        *self = SceneState::Loading { thread, path, progress }
    }
}

fn load_scene(
    path: ScenePath,
    progress: Arc<Mutex<Progress>>,
) -> Result<asset::Scene> {
    if let Ok(scene) = asset::Scene::deserialize(&path.cached) {
        Ok(scene)
    } else {
        let scene = asset_import::load_gltf(&path.gltf, Some(progress))?;
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
}

struct CrashReporter;

impl aftermath::AftermathDelegate for CrashReporter {
    fn dumped(&mut self, dump_data: &[u8]) {
        let report_path = "crash.nv-gpudmp";
        fs::write(report_path, dump_data)
            .expect("failed to write crash report");
        println!("wrote nvidia-aftermath crash report to {report_path}");
    }

    fn shader_debug_info(&mut self, _data: &[u8]) {}
    fn description(&mut self, _describe: &mut aftermath::DescriptionBuilder) {}
}

fn create_crash_report() -> ! {
    let status = aftermath::Status::wait_for_status(Some(
        std::time::Duration::from_secs(5),
    ));
    if status != aftermath::Status::Finished {
        panic!("Unexpected crash dump status: {:?}", status);
    }
    std::process::exit(1);
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

    let _crash_reporter = aftermath::Aftermath::new(CrashReporter);

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
                    SceneState::Loading { thread, path, .. }
                        if thread.is_finished() =>
                    {
                        match thread.join().expect("failed to join threads") {
                            Ok(new) => {
                                scene = new;
                                if let Some(renderer) = &mut renderer {
                                    renderer
                                        .change_scene(&scene)
                                        .unwrap_or_else(|_| {
                                            create_crash_report()
                                        });
                                }
                                SceneState::Loaded { path: path.clone() }
                            }
                            Err(report) => SceneState::Error {
                                error: report.root_cause().to_string(),
                            },
                        }
                    }
                    scene_state => scene_state,
                };

                if let Some(renderer) = &mut renderer {
                    let gui = gui.render(&window, &mut scene_state, renderer);
                    renderer
                        .render_frame(&FrameRequest {
                            camera_move: inputs
                                .camera_move(renderer.camera(), dt),
                            gui,
                        })
                        .unwrap_or_else(|_| {
                            create_crash_report();
                        });
                }
                window.request_redraw();
            }
            _ => (),
        }
    });
}

#[derive(Default)]
struct GuiState {
    scene_path: String,
}

struct Gui {
    state: GuiState,
    context: egui::Context,
    input_state: egui_winit::State,
}

impl Gui {
    fn new(window: &winit::window::Window) -> Self {
        Self {
            context: egui::Context::default(),
            input_state: egui_winit::State::new(&window),
            state: GuiState::default(),
        }
    }

    fn render(
        &mut self,
        window: &winit::window::Window,
        scene_state: &mut SceneState,
        renderer: &mut Renderer,
    ) -> GuiRequest {
        let input = self.input_state.take_egui_input(&window);
        let output = self.context.run(input, |ctx| {
            scene_window(
                ctx,
                &mut self.state,
                scene_state,
                renderer.node_tree_mut(),
            );
        });
        let primitives = self.context.tessellate(output.shapes);
        let pixels_per_point = self.context.pixels_per_point();
        GuiRequest {
            textures_delta: output.textures_delta,
            pixels_per_point,
            primitives,
        }
    }

    fn handle_input(&mut self, event: &WindowEvent) -> bool {
        self.input_state.on_event(&self.context, event).consumed
    }

    fn handle_resize(&mut self) {
        self.context = egui::Context::default();
    }
}

fn scene_window(
    context: &egui::Context,
    state: &mut GuiState,
    scene_state: &mut SceneState,
    tree: &mut NodeTree,
) {
    egui::Window::new("scene").scroll2([false, true]).resizable(true).show(
        &context,
        |ui| {
            ui.horizontal(|ui| {
                let label = ui.label("scene path");
                ui.text_edit_singleline(&mut state.scene_path)
                    .labelled_by(label.id);
                if ui.button("load").clicked() {
                    scene_state
                        .start_loading(ScenePath::new(&state.scene_path));
                }
            });

            ui.add_space(15.0);

            match scene_state {
                SceneState::Empty => {
                    ui.label("no scene loaded");
                }
                SceneState::Loaded { path } => {
                    ui.label(format!("'{:?}' is loaded", path.gltf));
                }
                SceneState::Error { error } => {
                    ui.label(format!("failed to load scene: {error}"));
                }
                SceneState::Loading { progress, .. } => {
                    let progress = *progress.lock().unwrap();
                    let progress_bar =
                        egui::ProgressBar::new(progress.percentage)
                            .show_percentage()
                            .animate(true);
                    ui.add(progress_bar)
                        .on_hover_text(
                            format!("loading {:?}", progress.stage,),
                        );
                }
            }

            ui.add_space(15.0);

            let mut index = 0;
            while let Some(next) = scene_tree(ui, tree, index) {
                index = next;
            }
        },
    );
}

fn scene_tree(
    ui: &mut egui::Ui,
    tree: &mut NodeTree,
    index: usize,
) -> Option<usize> {
    if let Some(node) = tree.nodes_mut().get_mut(index) {
        let mut transform = node.transform;
        ui.collapsing(format!("node {index}"), |ui| {
            transform = transform_edit(ui, transform);

            let mut child_index = index + 1;
            while tree
                .nodes_mut()
                .get_mut(child_index)
                .map(|child| child.parent())
                .flatten()
                .is_some_and(|parent| parent == index)
            {
                let Some(next) = scene_tree(ui, tree, child_index) else {
                    return ();
                };

                assert_ne!(next, child_index);
                child_index = next;
            }
        });

        let node = &mut tree.nodes_mut()[index];
        node.transform = transform;

        Some(node.sub_tree_end())
    } else {
        None
    }
}

fn transform_edit(ui: &mut egui::Ui, transform: Mat4) -> Mat4 {
    let (mut scale, mut rotation, mut translation) =
        transform.to_scale_rotation_translation();
    egui::Grid::new("transform").show(ui, |ui| {
        // FIXME: record the input difference and encode it to
        // a quaternion and multiply it onto the current rotation.
        ui.label("rotation");
        let euler = glam::EulerRot::XYZ;
        let (mut x, mut y, mut z) = rotation.to_euler(euler);
        ui.drag_angle_tau(&mut x);
        ui.drag_angle_tau(&mut y);
        ui.drag_angle_tau(&mut z);
        let [x, y, z] = [x, y, z].map(|a| a % std::f32::consts::TAU);
        rotation = Quat::from_euler(euler, x, y, z);
        ui.end_row();

        ui.label("translation");
        ui.add(egui::DragValue::new(&mut translation.x));
        ui.add(egui::DragValue::new(&mut translation.y));
        ui.add(egui::DragValue::new(&mut translation.z));
        ui.end_row();

        ui.label("scale");
        ui.add(egui::DragValue::new(&mut scale.x));
        ui.add(egui::DragValue::new(&mut scale.y));
        ui.add(egui::DragValue::new(&mut scale.z));
        scale = Vec3::from_array(scale.to_array().map(|a| {
            if a == 0.0 {
                f32::EPSILON
            } else {
                a
            }
        }));
        ui.end_row();
    });
    Mat4::from_scale_rotation_translation(scale, rotation, translation)
}

#[derive(Default)]
struct Inputs {
    keys_pressed: BitSet,
    modifier_state: ModifiersState,
    mouse_position: Option<Vec2>,
    mouse_delta: Option<Vec2>,
    camera_velocity: Vec3,
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

    fn camera_move(&mut self, camera: &Camera, dt: Duration) -> CameraMove {
        let mut camera_move = CameraMove::default();
        let dt = dt.as_secs_f32();

        let acceleration = 5.0;
        let drag = 8.0;
        let sensitivity = 0.05;

        self.camera_velocity -= self.camera_velocity * drag * dt.min(1.0);

        if self.is_key_pressed(VirtualKeyCode::W) {
            self.camera_velocity -= camera.forward * acceleration * dt;
        }
        if self.is_key_pressed(VirtualKeyCode::S) {
            self.camera_velocity += camera.forward * acceleration * dt;
        }

        let right = camera.right();

        if self.is_key_pressed(VirtualKeyCode::A) {
            self.camera_velocity -= right * acceleration * dt;
        }
        if self.is_key_pressed(VirtualKeyCode::D) {
            self.camera_velocity += right * acceleration * dt;
        }

        camera_move.position = self.camera_velocity;

        let mouse_delta = self.mouse_delta();
        if self.is_shift_pressed() {
            camera_move.yaw += sensitivity * mouse_delta.x;
            camera_move.pitch += sensitivity * mouse_delta.y;
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
