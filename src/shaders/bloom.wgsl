#import util
#import consts

@group(0) @binding(0)
var<uniform> consts: consts::Consts;

@group(0) @binding(2)
var linear_sampler: sampler;

@group(1) @binding(0)
var input: texture_2d<f32>;

var<push_constant> level: u32;

fn karis_average(color: vec3f) -> f32 {
    let luma = util::luminance(util::rgb_to_srgb(color)) / 4.0;
    return 1.0 / (1.0 + luma);
}

fn downscale_sample(uv: vec2f) -> vec3f {
    let a = textureSampleLevel(input, linear_sampler, uv, 0.0, vec2i(-2, 2)).rgb;
    let b = textureSampleLevel(input, linear_sampler, uv, 0.0, vec2i(0, 2)).rgb;
    let c = textureSampleLevel(input, linear_sampler, uv, 0.0, vec2i(2, 2)).rgb;
    let d = textureSampleLevel(input, linear_sampler, uv, 0.0, vec2i(-2, 0)).rgb;
    let e = textureSampleLevel(input, linear_sampler, uv, 0.0).rgb;
    let f = textureSampleLevel(input, linear_sampler, uv, 0.0, vec2i(2, 0)).rgb;
    let g = textureSampleLevel(input, linear_sampler, uv, 0.0, vec2i(-2, -2)).rgb;
    let h = textureSampleLevel(input, linear_sampler, uv, 0.0, vec2i(0, -2)).rgb;
    let i = textureSampleLevel(input, linear_sampler, uv, 0.0, vec2i(2, -2)).rgb;
    let j = textureSampleLevel(input, linear_sampler, uv, 0.0, vec2i(-1, 1)).rgb;
    let k = textureSampleLevel(input, linear_sampler, uv, 0.0, vec2i(1, 1)).rgb;
    let l = textureSampleLevel(input, linear_sampler, uv, 0.0, vec2i(-1, -1)).rgb;
    let m = textureSampleLevel(input, linear_sampler, uv, 0.0, vec2i(1, -1)).rgb;

    if level == 0u {
        var g0 = (a + b + d + e) * (0.125f / 4.0f);
        var g1 = (b + c + e + f) * (0.125f / 4.0f);
        var g2 = (d + e + g + h) * (0.125f / 4.0f);
        var g3 = (e + f + h + i) * (0.125f / 4.0f);
        var g4 = (j + k + l + m) * (0.5f / 4.0f);

        g0 *= karis_average(g0);
        g1 *= karis_average(g1);
        g2 *= karis_average(g2);
        g3 *= karis_average(g3);
        g4 *= karis_average(g4);

        return g0 + g1 + g2 + g3 + g4;
    } else {
        var sample = (a + c + g + i) * 0.03125;
        sample += (b + d + f + h) * 0.0625;
        sample += (e + j + k + l + m) * 0.125;

        return sample;
    }
}

fn upscale_sample(uv: vec2f) -> vec3f {
    let aspect = f32(consts.surface_size.x) / f32(consts.surface_size.y);

    let x = 0.004 / aspect;
    let y = 0.004;

    let a = textureSampleLevel(input, linear_sampler, vec2f(uv.x - x, uv.y + y), 0.0).rgb;
    let b = textureSampleLevel(input, linear_sampler, vec2f(uv.x, uv.y + y), 0.0).rgb;
    let c = textureSampleLevel(input, linear_sampler, vec2f(uv.x + x, uv.y + y), 0.0).rgb;

    let d = textureSampleLevel(input, linear_sampler, vec2f(uv.x - x, uv.y), 0.0).rgb;
    let e = textureSampleLevel(input, linear_sampler, vec2f(uv.x, uv.y), 0.0).rgb;
    let f = textureSampleLevel(input, linear_sampler, vec2f(uv.x + x, uv.y), 0.0).rgb;

    let g = textureSampleLevel(input, linear_sampler, vec2f(uv.x - x, uv.y - y), 0.0).rgb;
    let h = textureSampleLevel(input, linear_sampler, vec2f(uv.x, uv.y - y), 0.0).rgb;
    let i = textureSampleLevel(input, linear_sampler, vec2f(uv.x + x, uv.y - y), 0.0).rgb;

    var sample = e * 0.25;
    sample += (b + d + f + h) * 0.125;
    sample += (a + c + g + i) * 0.0625;

    return sample;
}

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
}

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
fn downsample(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(downscale_sample(in.uv), 1.0);
}

@fragment
fn upsample(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(upscale_sample(in.uv), 1.0);
}
