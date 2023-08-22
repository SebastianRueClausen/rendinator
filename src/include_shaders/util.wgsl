#define_import_path util

const PI = 3.14159265358979323846264;
const TAU = 6.28318530717958647692528;

fn dequantize_unorm(bits: u32, value: u32) -> f32 {
    let scale = f32((1 << bits) - 1);
    return f32(value) / scale;
}

fn quantize_unorm(bits: u32, value: f32) -> u32 {
    let scale = f32((1 << bits) - 1);
    return u32(value * scale + 0.5);
}

fn octahedron_encode(vector: vec3f) -> vec2f {
    let normal = vector / (abs(vector.x) + abs(vector.y) + abs(vector.z));

    if normal.z < 0.0 {
        let wrapped = (vec2f(1.0) - abs(normal.yx)) * select(vec2f(0.0), vec2f(1.0), normal.xy >= 0.0);
        return wrapped * 0.5 + 0.5;
    } else {
        return normal.xy * 0.5 + 0.5;
    }
}

fn octahedron_decode(octahedron: vec2f) -> vec3f {
    let scaled = octahedron * 2.0 - 1.0;
    var normal = vec3f(scaled.xy, 1.0 - abs(scaled.x) - abs(scaled.y));
    let t = saturate(-normal.z);
    normal.x += select(t, -t, normal.x >= 0.0);
    normal.y += select(t, -t, normal.y >= 0.0);
    return normalize(normal);
}

fn luminance(color: vec3f) -> f32 {
    return dot(color, vec3f(0.2126, 0.7152, 0.0722));
}

fn rgb_to_srgb(color: vec3f) -> vec3f {
    return pow(color, vec3<f32>(1.0 / 2.2));
}
