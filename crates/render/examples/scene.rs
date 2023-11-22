use raw_window_handle::HasRawDisplayHandle;
use render::{FrameRequest, GuiRequest, Renderer};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;

fn main() {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();

    let scene = asset::Scene::default();
    let mut renderer = Some(create_renderer(&window, &scene));

    let mut gui_ctx = egui::Context::default();
    let mut gui_state = egui_winit::State::new(&window);

    event_loop.run(move |event, _, ctrl| {
        ctrl.set_poll();

        if let Event::WindowEvent { event, .. } = &event {
            if gui_state.on_event(&gui_ctx, event).consumed {
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
                gui_ctx = egui::Context::default();
                if size.width == 0 && size.height == 0 {
                    renderer.take();
                } else {
                    renderer.take();
                    renderer = Some(create_renderer(&window, &scene));
                }
            }
            Event::MainEventsCleared => {
                let input = gui_state.take_egui_input(&window);
                let output = gui_ctx.run(input, |ctx| {
                    egui::Window::new("window").show(&ctx, |ui| {
                        ui.label("Hello world!");
                    });
                });

                if let Some(renderer) = &mut renderer {
                    renderer
                        .render_frame(&FrameRequest {
                            gui: GuiRequest {
                                textures_delta: &output.textures_delta,
                                primitives: &gui_ctx.tessellate(output.shapes),
                                pixels_per_point: 1.0,
                            },
                        })
                        .expect("failed to render frame");
                }

                window.request_redraw();
            }
            _ => (),
        }
    });
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
