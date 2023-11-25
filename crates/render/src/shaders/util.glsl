#ifndef UTIL
#define UTIL

const float TAU = 6.283185307179586;
const float PI = 3.1415926535897932384626433832795028;

float dequantize_unorm(uint bits, uint value) {
    float scale = float((1 << bits) - 1);
    return float(value) / scale;
}

float quantize_unorm(uint bits, float value) {
    float scale = float((1 << bits) - 1);
    return float(value * scale + 0.5);
}

float dequantize_snorm(uint bits, int value) {
    float scale = float((1 << (bits - 1)) - 1);
    return max(float(value) / scale, -1.0);
}

vec2 octahedron_encode(vec3 vector) {
    vec3 normal = vector / (abs(vector.x) + abs(vector.y) + abs(vector.z));

    if (normal.z < 0.0) {
        float x = normal.x >= 0.0 ? 1.0 : 0.0;
        float y = normal.y >= 0.0 ? 1.0 : 0.0;
        vec2 wrapped = (vec2(1.0) - abs(normal.yx)) * vec2(x, y);
        return wrapped * 0.5 + 0.5;
    } else {
        return normal.xy * 0.5 + 0.5;
    }
}

vec3 octahedron_decode(vec2 octahedron) {
    vec2 scaled = octahedron * 2.0 - 1.0;
    vec3 normal = vec3(scaled.xy, 1.0 - abs(scaled.x) - abs(scaled.y));
    float t = clamp(-normal.z, 0.0, 1.0);
    normal.x += normal.x >= 0.0 ? -t : t;
    normal.y += normal.y >= 0.0 ? -t : t;
    return normalize(normal);
}

struct DrawIndexedIndirectCommand {
    uint index_count;
    uint instance_count;
    uint first_index;
    int vertex_offset;
    uint first_instance;
};

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

#endif