#import consts

@group(0) @binding(0)
var<uniform> consts: consts::Consts;

@group(0) @binding(2)
var linear_sampler: sampler;

@group(1) @binding(0)
var color_buffer: texture_2d<f32>;

@group(1) @binding(1)
var depth_buffer: texture_depth_2d;

@group(1) @binding(2)
var color_accum_buffer: texture_2d<f32>;

@group(1) @binding(3)
var post_buffer: texture_storage_2d<rgba16float, write>;

var<push_constant> reproject: mat4x4f;

const BLOCK_WIDTH = 8u;
const BORDER_WIDTH = 1u;
const CACHE_WIDTH = 10u;
const CACHE_LENGTH = 100u;

// 10x10 `color_buffer` and `depth_buffer` cache, which contains all the neighbourhood values of the workgroup.
var<workgroup> color_cache: array<vec4f, CACHE_LENGTH>;
var<workgroup> depth_cache: array<f32, CACHE_LENGTH>;

// Get cache coordinates of a cache index.
fn cache_index_to_coords(index: u32) -> vec2i {
    return vec2i(i32(index % CACHE_WIDTH), i32(index / CACHE_WIDTH));
}

// Get the cache index of cache coordinates.
fn coords_to_cache_index(coords: vec2i) -> u32 {
    return u32(coords.x) + u32(coords.y) * CACHE_WIDTH;
}

fn fill_color_cache(workgroup_id: vec2u, local_index: u32, local_id: vec2u) {
    let edge = vec2i(consts.surface_size) - 1;
    let upper_left = vec2i(workgroup_id * BLOCK_WIDTH - BORDER_WIDTH);

    for (var t = local_index; t < CACHE_LENGTH; t += BLOCK_WIDTH * BLOCK_WIDTH) {
        let pixel = upper_left + cache_index_to_coords(t);
        let coords = clamp(pixel, vec2i(0), edge);

        let color = textureLoad(color_buffer, coords, 0);
        let depth = textureLoad(depth_buffer, coords, 0);

        color_cache[t] = saturate(color);
        depth_cache[t] = depth;
    }

    workgroupBarrier();
}

// 5 tap catmull-rom sample in a 4x4 grid.
fn sample_history_catmull_rom(texcoords: vec2f, size: vec2f) -> vec4f {
    let sample_position = texcoords * size;
    let p1 = floor(sample_position - 0.5) + 0.5;

    // The offset from `p1` to `sample_position`, used to generate the filter weights.
    let f = sample_position - p1;

    // Calculate the catmull-rom weight.
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);

    // Weights and offset for bilinear filter of the inner 2x2 grid around the center.
    let w12 = w1 + w2;
    let offset12 = w2 / (w1 + w2);
    let texel_size = 1.0 / size;

    // The uv coordintes for sampling.
    let p0 = (p1 - 1.0) * texel_size;
    let p3 = (p1 + 2.0) * texel_size;
    let p12 = (p1 + offset12) * texel_size;

    var out = vec4f(0.0);
    out += textureSampleLevel(color_accum_buffer, linear_sampler, vec2f(p12.x, p0.y), 0.0) * w12.x * w0.y;
    out += textureSampleLevel(color_accum_buffer, linear_sampler, vec2f(p0.x, p12.y), 0.0) * w0.x * w12.y;
    out += textureSampleLevel(color_accum_buffer, linear_sampler, vec2f(p12.x, p12.y), 0.0) * w12.x * w12.y;
    out += textureSampleLevel(color_accum_buffer, linear_sampler, vec2f(p3.x, p12.y), 0.0) * w3.x * w12.y;
    out += textureSampleLevel(color_accum_buffer, linear_sampler, vec2f(p12.x, p3.y), 0.0) * w12.x * w3.y;

    return out;
}

struct Neighborhood {
    min: vec3f,
    max: vec3f,
    center_sample: vec4f,
    nearest_depth: f32,
};

fn sample_neighborhood(
    center: vec2i,
    size: vec2i,
    workgroup_id: vec2u,
    local_index: u32,
    local_id: vec2u,
) -> Neighborhood {
    var neighborhood: Neighborhood;
    neighborhood.nearest_depth = 1.0;

    neighborhood.min = vec3f(99.0);
    neighborhood.max = vec3f(0.0);

    for (var x: i32 = -1; x <= 1; x++) {
        for (var y: i32 = -1; y <= 1; y++) {
            let offset = vec2i(x, y);

            let cache_index = coords_to_cache_index(vec2i(local_id) + i32(BORDER_WIDTH) + offset);
            let neighbor = color_cache[cache_index];

            let depth = depth_cache[cache_index];

            if x == 0 && y == 0 {
                neighborhood.center_sample = neighbor;
            }

            if neighbor.w <= 0.5 {
                continue;
            }

            neighborhood.min = min(neighborhood.min, neighbor.xyz);
            neighborhood.max = max(neighborhood.max, neighbor.xyz);

            neighborhood.nearest_depth = min(neighborhood.nearest_depth, depth);
        }
    }

    return neighborhood;
}

fn reproject_texcoords(texcoords: vec2f, velocity: vec2f, depth: f32) -> vec2f {
    let clip = (vec2f(2.0, -2.0) * texcoords) + vec2f(-1.0, 1.0);
    let previous_clip = reproject * vec4f(clip, depth, 1.0);
    return (previous_clip.xy / previous_clip.w) - velocity;
}

@compute
@workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id) texel_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(local_invocation_id) local_id: vec3u,
) {
    let size = vec2i(consts.surface_size);
    let center = vec2i(texel_id.xy);

    fill_color_cache(workgroup_id.xy, local_index, local_id.xy);

    if any(center >= size) {
        return;
    }

    let neighborhood = sample_neighborhood(center, size, workgroup_id.xy, local_index, local_id.xy);

    let texcoords = (vec2f(center) + 0.5) / vec2f(size);
    var velocity = vec2f(0.0);

    let history_texcoords = reproject_texcoords(texcoords, velocity, neighborhood.nearest_depth);
    let history = sample_history_catmull_rom(history_texcoords, vec2f(size));

    let no_history = any(history_texcoords != saturate(history_texcoords));
    let source_weight = select(0.1, 1.0, no_history);

    let history_sample = clamp(history.xyz, neighborhood.min, neighborhood.max);
    let source_sample = neighborhood.center_sample.xyz;

    let result = mix(history_sample, source_sample, source_weight);

    textureStore(post_buffer, center, vec4f(result, 1.0));
}
