
struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
}

@group(0) @binding(0)
var input: texture_2d<f32>;

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
    let texel_id = vec2i(in.uv * vec2f(textureDimensions(input, 0)));
    return textureLoad(input, texel_id, 0);
}
