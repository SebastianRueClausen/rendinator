
@group(0) @binding(2)
var linear_sampler: sampler;

#if INITIAL_REDUCE == true
@group(1) @binding(0)
var input: texture_depth_2d;
#else
@group(1) @binding(0)
var input: texture_2d<f32>;
#endif

@group(1) @binding(1)
var output: texture_storage_2d<rg32float, write>;

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) invocation_id: vec3u) {
    let output_size = textureDimensions(output);

    if any(invocation_id.xy >= output_size) {
        return;
    }

    let uv = (vec2f(invocation_id.xy) + 0.5) / vec2f(output_size);

    #if INITIAL_REDUCE == true
    let depth_min = textureGather(input, linear_sampler, uv);
    let depth_max = textureGather(input, linear_sampler, uv);
    #else
    let depth_min = textureGather(0, input, linear_sampler, uv);
    let depth_max = textureGather(1, input, linear_sampler, uv);
    #endif

    let out = vec2f(
        min(depth_min.x, min(depth_min.y, min(depth_min.z, depth_min.w))),
        max(depth_max.x, max(depth_max.y, max(depth_max.z, depth_max.w))),
    );

    textureStore(output, invocation_id.xy, vec4f(out, 0.0, 0.0));
}
