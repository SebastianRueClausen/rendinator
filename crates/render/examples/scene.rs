use render::Renderer;

fn main() {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();

    window.set_cursor_grab(false).unwrap();
    window.set_cursor_visible(false);

    let renderer = Renderer::new(true, &window).unwrap();

    for _ in 0..1 {
        renderer.render_frame().unwrap();
    }
}
