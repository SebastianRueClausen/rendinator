#ifndef TONEMAP
#define TONEMAP

vec3 rgb_to_ycbcr(vec3 col) {
    mat3 m = mat3(0.2126, 0.7152, 0.0722, -0.1146,-0.3854, 0.5, 0.5,-0.4542,-0.0458);
    return col * m;
}

float rgb_to_luminance(vec3 col) {
    return dot(vec3(0.2126, 0.7152, 0.0722), col);
}

float tonemap_curve(float v) {
    #if 0
        // Large linear part in the lows, but compresses highs.
        float c = v + v*v + 0.5*v*v*v;
        return c / (1.0 + c);
    #else
        return 1.0 - exp(-v);
    #endif
}

vec3 tonemap_curve(vec3 v) {
    return vec3(tonemap_curve(v.r), tonemap_curve(v.g), tonemap_curve(v.b));
}

vec3 neutral_tonemap(vec3 col) {
    vec3 ycbcr = rgb_to_ycbcr(col);

    float bt = tonemap_curve(length(ycbcr.yz) * 2.4);
    float desat = max((bt - 0.7) * 0.8, 0.0);
    desat *= desat;

    vec3 desat_col = mix(col.rgb, ycbcr.xxx, desat);

    float tm_lum = tonemap_curve(ycbcr.x);
    vec3 tm0 = col.rgb * max(0.0, tm_lum / max(1e-5, rgb_to_luminance(col.rgb)));
    vec3 tm1 = tonemap_curve(desat_col);
    col = mix(tm0, tm1, bt * bt);

    return col * 0.97;
}

#endif