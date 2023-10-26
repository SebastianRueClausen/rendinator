#define_import_path light

struct DirectionalLight {
    direction: vec4f,
    irradiance: vec4f,
}

struct ShadowCascade {
    proj_view: mat4x4f,
    split: f32,
    split_depth: f32,
    padding: array<u32, 2>,
}
