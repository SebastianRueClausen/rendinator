use raw_window_handle::HasRawWindowHandle;
use render::{FrameRequest, GuiRequest, Renderer};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = Window::new(&event_loop).unwrap();
    // window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
    window.set_cursor_visible(false);

    let scene = asset::Scene::default();
    let mut renderer = Some(create_renderer(&window, &scene));

    let gui_ctx = egui::Context::default();

    event_loop
        .run(move |event, elwt| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested, ..
            } => {
                elwt.exit();
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size), ..
            } => {
                if size.width == 0 && size.height == 0 {
                    renderer.take();
                } else {
                    renderer.take();
                    renderer = Some(create_renderer(&window, &scene));
                }
            }
            Event::AboutToWait => {
                let input = egui::RawInput::default();
                let output = gui_ctx.run(input, |ctx| {
                    egui::Window::new("window").show(&ctx, |ui| {
                        ui.label("Hello world!");
                    });
                });

                println!("{}", gui_ctx.pixels_per_point());

                if let Some(renderer) = &mut renderer {
                    renderer
                        .render_frame(&FrameRequest {
                            gui: GuiRequest {
                                textures_delta: &output.textures_delta,
                                primitives: &gui_ctx.tessellate(output.shapes),
                                pixels_per_point: gui_ctx.pixels_per_point(),
                            },
                        })
                        .expect("failed to render frame");
                }
                window.request_redraw();
            }
            _ => (),
        })
        .unwrap();
}

fn gui(ctx: &egui::Context) {}

fn create_renderer(
    window: &winit::window::Window,
    scene: &asset::Scene,
) -> Renderer {
    Renderer::new(render::RendererRequest {
        window: window.raw_window_handle(),
        width: window.inner_size().width,
        height: window.inner_size().height,
        validate: true,
        scene: &scene,
    })
    .unwrap()
}
