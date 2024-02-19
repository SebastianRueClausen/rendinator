#version 460
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_nonuniform_qualifier: require
#extension GL_EXT_ray_query: enable

#include "../scene.glsl"
#include "../util.glsl"
#include "../brdf.glsl"
#include "../constants.glsl"
#include "../tonemap.glsl"

layout (binding = 0) uniform ConstantData {
    Constants constants;
};

layout (binding = 5) buffer Materials {
    Material materials[];
};

layout (binding = 6) uniform accelerationStructureEXT acceleration_structure;
layout (binding = 7) uniform sampler2D textures[];

layout (location = 0) in vec4 world_position;
layout (location = 1) in vec3 world_normal;
layout (location = 2) in vec3 world_tangent;
layout (location = 3) in vec3 world_bitangent;
layout (location = 4) in vec2 texcoord;
layout (location = 5) in flat uint material;
layout (location = 6) in flat uint draw_index;

layout (location = 0) out vec4 color_result;
layout (location = 1) out uvec4 visibility_result;

float sun_shadow() {
    float min_dist = 1.0e-3;
    float max_dist = 1000.0;
    vec3 direction = normalize(constants.sun.direction.xyz);

    rayQueryEXT ray_query;
    rayQueryInitializeEXT(
        ray_query,
        acceleration_structure,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xff,
        world_position.xyz,
        min_dist,
        direction,
        max_dist
    );
    rayQueryProceedEXT(ray_query);

    bool occluder_hit = (rayQueryGetIntersectionTypeEXT(ray_query, true) != gl_RayQueryCommittedIntersectionNoneEXT);

    return occluder_hit ? 0.0 : 1.0;
}

void main() {
    Material material = materials[material];
    vec4 tangent_normal = texture(
        textures[material.normal_texture],
        texcoord
    );

    vec3 normal = octahedron_decode(tangent_normal.xy);
    normal = normalize(
        normal.x * world_tangent
            + normal.y * world_bitangent
            + normal.z * world_normal
    );

    ShadeParameters shade;

    shade.albedo = texture(
        textures[material.albedo_texture],
        texcoord
    ).rgb;

    vec4 specular_parameters = texture(
        textures[material.specular_texture],
        texcoord
    );

    shade.metallic = specular_parameters.r * material.metallic;
    shade.roughness = pow(specular_parameters.g * material.roughness, 2);

    float dielectric_specular = pow((material.ior - 1.0) / (material.ior + 1.0), 2);
    shade.fresnel_min = mix(vec3(dielectric_specular), shade.albedo, shade.metallic);
    shade.fresnel_max = saturate(dot(shade.fresnel_min, vec3(50.0 * 0.33)));

    vec3 emissive = texture(
        textures[material.emissive_texture],
        texcoord
    ).rgb * material.emissive.rgb;

    shade.view_direction = normalize(constants.camera_position.xyz - world_position.xyz);
    shade.normal_dot_view = clamp(dot(normal, shade.view_direction), 0.0001, 1.0);

    LightParameters light;
    light.specular_intensity = 1.0;
    light.light_direction = constants.sun.direction.xyz;
    light.half_vector = normalize(shade.view_direction + light.light_direction);
    light.normal_dot_half = saturate(dot(normal, light.half_vector));
    light.normal_dot_light = saturate(dot(normal, light.light_direction));
    light.view_dot_half = saturate(dot(shade.view_direction, light.half_vector));
    light.light_dot_view = saturate(dot(light.light_direction, shade.view_direction));
    light.light_dot_half = saturate(dot(light.light_direction, light.half_vector));

    vec3 diffuse_color = shade.albedo * (1.0 - shade.metallic);
    vec3 specular = ggx_specular(shade, light);
    vec3 diffuse = diffuse_color * burley_diffuse(shade, light);

    vec3 radiance = (diffuse + specular)
        * light.normal_dot_light
        * constants.sun.irradiance.xyz
        * sun_shadow();

    vec3 ambient = shade.albedo * 0.2;
    vec3 final = radiance + emissive + ambient;

    color_result = vec4(pow(neutral_tonemap(final), vec3(1.0 / 2.2)), 1.0);
    visibility_result = uvec4(
        gl_PrimitiveID,
        draw_index,
        0,
        0
    );
}