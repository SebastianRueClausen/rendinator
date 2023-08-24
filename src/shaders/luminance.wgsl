#import util

@group(0) @binding(0)
var display: texture_2d<f32>;

#if BUILD_HISTOGRAM == true
@group(1) @binding(0)
var<storage, read_write> histogram: array<atomic<u32>, 256>;
#else
@group(1) @binding(0)
var<storage, read_write> histogram: array<u32, 256>;
#endif

@group(1) @binding(1)
var<storage, read_write> average_luminance: f32;

#if BUILD_HISTOGRAM == true
var<workgroup> shared_histogram: array<atomic<u32>, 256>;
#else
var<workgroup> shared_histogram: array<u32, 256>;
#endif

struct Params {
    min_log_luminance: f32,
    inverse_log_luminance_range: f32,
    log_luminance_range: f32,
    time_coeff: f32,
    pixel_count: u32,
}

var<push_constant> params: Params;

#if BUILD_HISTOGRAM == true

fn color_index(color: vec3f) -> u32 {
    let luminance = util::luminance(color);

    if luminance < 0.005 {
        return 0u;
    }

    var log_luminance = (log2(luminance) - params.min_log_luminance)
        * params.inverse_log_luminance_range;
    log_luminance = clamp(log_luminance, 0.0, 1.0);

    return u32(log_luminance * 254.0 + 1.0);
}

@compute
@workgroup_size(16, 16)
fn build_histogram(
    @builtin(global_invocation_id) invocation_id: vec3u,
    @builtin(local_invocation_index) local_invocation_index: u32,
) {
    let display_size = textureDimensions(display);

    // Note that this is not required to be atomic.
    let bin = &shared_histogram[local_invocation_index];
    atomicStore(bin, 0u);

    workgroupBarrier();

    if all(invocation_id.xy < display_size) {
        let color = textureLoad(display, vec2i(invocation_id.xy), 0).rgb;
        let index = color_index(color);
        atomicAdd(&shared_histogram[index], 1u);
    }

    workgroupBarrier();

    // Note that this is not required to be atomic.
    let shared_histogram_value = atomicLoad(&shared_histogram[local_invocation_index]);

    atomicAdd(&histogram[local_invocation_index], shared_histogram_value);
}

#else

@compute
@workgroup_size(256)
fn compute_average(
    @builtin(global_invocation_id) invocation_id: vec3u,
    @builtin(local_invocation_index) local_invocation_index: u32,
) {
    let count = histogram[local_invocation_index];
    shared_histogram[local_invocation_index] = count * local_invocation_index;

    workgroupBarrier();

    histogram[local_invocation_index] = 0u;

    for (var cutoff = (256u >> 1u); cutoff > 0u; cutoff >>= 1u) {
        if local_invocation_index < cutoff {
            shared_histogram[local_invocation_index]
                += shared_histogram[local_invocation_index + cutoff];
        }

        workgroupBarrier();
    }

    if local_invocation_index == 0u {
        let weighted_log_average = (
            f32(shared_histogram[0]) / f32(max(params.pixel_count - count, 1u))
        ) - 1.0;

        let weighted_average = exp2(
            ((weighted_log_average / 254.0) * params.log_luminance_range)
                + params.min_log_luminance
        );

        let last_frame_average = average_luminance;
        let adapted_luminance = last_frame_average + (weighted_average - last_frame_average) * params.time_coeff;

        average_luminance = adapted_luminance;
    }
}

#endif
