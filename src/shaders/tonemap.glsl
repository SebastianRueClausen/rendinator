#ifndef TONEMAP
#define TONEMAP

vec3 aces_approx_tonemap(vec3 color) {
    color *= 0.6;

    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

#endif
