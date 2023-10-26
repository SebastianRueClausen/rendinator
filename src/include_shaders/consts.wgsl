#define_import_path consts
#import light

struct Consts {
    camera_pos: vec4f,
    camera_front: vec4f,
    proj_view: mat4x4f,
    proj: mat4x4f,
    prev_proj_view: mat4x4f,
    inverse_proj_view: mat4x4f,
    sun: light::DirectionalLight,
    frustrum_z_planes: vec2f,
    surface_size: vec2u,
    jitter: vec2f,
    frame_index: u32,
    camera_fov: f32,
};
