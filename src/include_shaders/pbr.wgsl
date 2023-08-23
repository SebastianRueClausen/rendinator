#define_import_path pbr
#import util

struct ShadeParameters {
    metallic: f32,
    roughness: f32,
    albedo: vec3f,
    fresnel_min: vec3f,
    fresnel_max: f32,
    view_direction: vec3f,
    normal_dot_view: f32,
};

struct LightParameters {
    specular_intensity: f32,
    light_direction: vec3f,
    half_vector: vec3f,
    normal_dot_half: f32,
    normal_dot_light: f32,
    view_dot_half: f32,
    light_dot_view: f32,
    light_dot_half: f32,
};

fn fresnel_schlick(fresnel_min: vec3f, fresnel_max: f32, view_dot_half: f32) -> vec3f {
    let flipped = 1.0 - view_dot_half;
    let flipped_2 = flipped * flipped;
    let flipped_5 = flipped * flipped_2 * flipped_2;
    return fresnel_min + (fresnel_max - fresnel_min) * flipped_5;
}

fn ggx_visibility(shade: ShadeParameters, light: LightParameters) -> f32 {
    let alpha_squared = shade.roughness * shade.roughness;
    let lambda_view = light.normal_dot_light * sqrt(
        (shade.normal_dot_view - alpha_squared * shade.normal_dot_view)
            * shade.normal_dot_view
            + alpha_squared
    );

    let lambda_light = shade.normal_dot_view * sqrt(
        (light.normal_dot_light - alpha_squared * light.normal_dot_light)
            * light.normal_dot_light
            + alpha_squared
    );

    return 0.5 / (lambda_view + lambda_light);
}

fn ggx_normal_dist(shade: ShadeParameters, light: LightParameters) -> f32 {
    let alpha = light.normal_dot_half * shade.roughness;
    let k = shade.roughness / ((1.0 - light.normal_dot_half * light.normal_dot_half) + alpha * alpha);
    return k * k * (1.0 / util::PI);
}

fn specular(shade: ShadeParameters, light: LightParameters) -> vec3f {
    let d = ggx_normal_dist(shade, light);
    let v = ggx_visibility(shade, light);
    let f = fresnel_schlick(shade.fresnel_min, shade.fresnel_max, light.view_dot_half);
    return (light.specular_intensity * d * v) * f;
}

fn lambert_diffuse(shade: ShadeParameters) -> vec3f {
    let diffuse_color = shade.albedo * (1.0 - shade.metallic);
    return diffuse_color * (1.0 / util::PI);
}

fn burley_diffuse(shade: ShadeParameters, light: LightParameters) -> vec3f {
    let light_scatter = fresnel_schlick(vec3f(1.0), shade.fresnel_max, light.normal_dot_light);
    let view_scatter  = fresnel_schlick(vec3f(1.0), shade.fresnel_max, shade.normal_dot_view);
    return light_scatter * view_scatter * (1.0 / util::PI);
}
