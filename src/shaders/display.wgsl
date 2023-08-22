#import consts

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
}

@group(0) @binding(0)
var<uniform> consts: consts::Consts;

@group(1) @binding(0)
var display: texture_2d<f32>;

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let vertex_index = 2u - in.vertex_index;

    out.uv = vec2f(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u),
    );

    out.clip_position = vec4f(out.uv * 2.0 - 1.0, 0.0, 1.0);
    out.clip_position.y *= -1.0;

    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = vec2f(consts.surface_size);
    let texel_id = vec2i(in.uv * dimensions);
    return textureLoad(display, texel_id, 0);
}
