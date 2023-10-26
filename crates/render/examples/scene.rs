use raw_window_handle::HasRawWindowHandle;
use render::Renderer;

fn main() {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();

    window.set_cursor_grab(false).unwrap();
    window.set_cursor_visible(false);

    let scene = asset::Scene::default();
    let renderer = Renderer::new(render::RendererRequest {
        window: window.raw_window_handle(),
        width: window.inner_size().width,
        height: window.inner_size().height,
        validate: true,
        scene: &scene,
    })
    .unwrap();

    for _ in 0..1 {
        renderer.render_frame().unwrap();
    }
}
