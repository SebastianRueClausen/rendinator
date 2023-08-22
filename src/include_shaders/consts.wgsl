#define_import_path consts

struct Consts {
    camera_pos: vec4f,
    proj_view: mat4x4f,
    prev_proj_view: mat4x4f,
    inverse_proj_view: mat4x4f,
    frustrum_z_planes: vec2f,
    surface_size: vec2u,
    jitter: vec2f,
    frame_index: u32,
};
