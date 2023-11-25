#ifndef BRDF
#define BRDF

#include "util.glsl"

struct ShadeParameters {
    float metallic;
    float roughness;
    vec3 albedo;
    vec3 fresnel_min;
    float fresnel_max;
    vec3 view_direction;
    float normal_dot_view;
};

struct LightParameters {
    float specular_intensity;
    vec3 light_direction;
    vec3 half_vector;
    float normal_dot_half;
    float normal_dot_light;
    float view_dot_half;
    float light_dot_view;
    float light_dot_half;
};

vec3 fresnel_schlick(vec3 fresnel_min, float fresnel_max, float view_dot_half) {
    float flipped = 1.0 - view_dot_half;
    float flipped_2 = flipped * flipped;
    float flipped_5 = flipped * flipped_2 * flipped_2;
    return fresnel_min + (fresnel_max - fresnel_min) * flipped_5;
}

float ggx_visibility(ShadeParameters shade, LightParameters light) {
    float alpha_squared = shade.roughness * shade.roughness;
    float lambda_view = light.normal_dot_light * sqrt(
        (shade.normal_dot_view - alpha_squared * shade.normal_dot_view)
            * shade.normal_dot_view
            + alpha_squared
    );

    float lambda_light = shade.normal_dot_view * sqrt(
        (light.normal_dot_light - alpha_squared * light.normal_dot_light)
            * light.normal_dot_light
            + alpha_squared
    );

    return 0.5 / (lambda_view + lambda_light);
}

float ggx_normal_dist(ShadeParameters shade, LightParameters light) {
    float alpha = light.normal_dot_half * shade.roughness;
    float k = shade.roughness / ((1.0 - light.normal_dot_half * light.normal_dot_half) + alpha * alpha);
    return k * k * (1.0 / PI);
}

vec3 ggx_specular(ShadeParameters shade, LightParameters light) {
    float d = ggx_normal_dist(shade, light);
    float v = ggx_visibility(shade, light);
    vec3 f = fresnel_schlick(shade.fresnel_min, shade.fresnel_max, light.view_dot_half);
    return (light.specular_intensity * d * v) * f;
}

vec3 lambert_diffuse(ShadeParameters shade) {
    vec3 diffuse_color = shade.albedo * (1.0 - shade.metallic);
    return diffuse_color * (1.0 / PI);
}

vec3 burley_diffuse(ShadeParameters shade, LightParameters light) {
    vec3 light_scatter = fresnel_schlick(vec3(1.0), shade.fresnel_max, light.normal_dot_light);
    vec3 view_scatter  = fresnel_schlick(vec3(1.0), shade.fresnel_max, shade.normal_dot_view);
    return light_scatter * view_scatter * (1.0 / PI);
}

#endif
