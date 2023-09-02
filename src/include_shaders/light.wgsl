#define_import_path light

struct DirectionalLight {
    direction: vec4f,
    irradiance: vec4f,
}

struct ShadowCascade {
    matrix: mat4x4f,
    corners: array<vec4f, 8>,
    center: vec4f,
    near: f32,
    far: f32,
    padding: array<u32, 2>,
}
