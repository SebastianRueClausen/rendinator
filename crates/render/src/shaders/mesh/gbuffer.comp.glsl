#version 450
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_control_flow_attributes: require
#extension GL_EXT_nonuniform_qualifier: require

#include "../constants.glsl"
#include "../scene.glsl"

layout (binding = 0) uniform ConstantData {
    Constants constants;
};

layout (binding = 1) readonly buffer Instances {
    Instance instances[];
};

layout (binding = 2) readonly buffer Draws {
    Draw draws[];
};

layout (binding = 3) readonly buffer Indices {
    uint indices[];
};

layout (binding = 4) readonly buffer Vertices {
    Vertex vertices[];
};

layout (binding = 5) readonly buffer Materials {
    Material materials[];
};

layout (binding = 6, rg32ui) readonly uniform uimage2D visibility_buffer;

layout (binding = 7, r11f_g11f_b10f) writeonly uniform image2D gbuffer0;
layout (binding = 8, rgba8) writeonly uniform image2D gbuffer1;
layout (binding = 9, rgba8) writeonly uniform image2D gbuffer2;

layout (binding = 10) uniform sampler2D textures[];

layout(push_constant) uniform PushConstants {
	mat4 ray_matrix;
};

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

vec3 triangle_intersection(Triangle triangle, vec3 origin, vec3 direction) {
    vec3 edge_to_origin = origin - triangle.v0;
    vec3 edge_2 = triangle.v2 - triangle.v0;
    vec3 edge_1 = triangle.v1 - triangle.v0;
    vec3 r = cross(direction, edge_2);
    vec3 s = cross(edge_to_origin, edge_1);
    float inverse_det = 1.0 / dot(r, edge_1);
    float v1 = dot(r, edge_to_origin);
    float v2 = dot(s, direction);
    float b = v1 * inverse_det;
    float c = v2 * inverse_det;
    float a = 1.0 - b - c;
    return vec3(a, b, c);
}

struct Barycentric {
    vec3 lambda;
    vec3 ddx;
    vec3 ddy;
};

Barycentric barycentric(Triangle triangle, vec2 ndc, vec2 screen_size) {
    vec3 origin = constants.camera_position.xyz;
    vec3 direction = (ray_matrix * vec4(ndc, 1.0, 1.0)).xyz;

    Barycentric barycentric;
    barycentric.lambda = triangle_intersection(triangle, origin, direction);

    vec2 texel_size = 2.0 / screen_size;

    direction = (ray_matrix * vec4(ndc.x + texel_size.x, ndc.y, 1.0, 1.0)).xyz;
    vec3 hx = triangle_intersection(triangle, origin, direction);

    direction = (ray_matrix * vec4(ndc.x, ndc.y + texel_size.y, 1.0, 1.0)).xyz;
    vec3 hy = triangle_intersection(triangle, origin, direction);

    barycentric.ddx = barycentric.lambda - hx;
    barycentric.ddy = barycentric.lambda - hy;

    return barycentric;
}

float interp_scalar(vec3 lambda, float values[3]) {
    return dot(vec3(values[0], values[1], values[2]), lambda);
}

vec2 interp_vec2(vec3 lambda, vec2 values[3]) {
    return lambda[0] * values[1] + lambda[1] * values[2] + lambda[2] * values[2];
}

vec3 interp_vec3(vec3 lambda, vec3 values[3]) {
    return lambda[0] * values[1] + lambda[1] * values[2] + lambda[2] * values[2];
}

void main() {
    uvec2 pixel_index = gl_GlobalInvocationID.xy;
    uvec2 target_size = imageSize(visibility_buffer).xy;

    if (any(greaterThanEqual(pixel_index, target_size))) {
        return;
    }

    uvec2 visibility = imageLoad(visibility_buffer, ivec2(pixel_index)).xy;

    uint base_index = visibility.x * 3;
    uint i0 = indices[base_index];
    uint i1 = indices[base_index + 1];
    uint i2 = indices[base_index + 2];

    Vertex v0 = vertices[i0];
    Vertex v1 = vertices[i1];
    Vertex v2 = vertices[i2];

    Draw draw = draws[visibility.y];
    Instance instance = instances[draw.instance_index];

    BoundingSphere bounding_sphere;
    bounding_sphere.center = vec3(draw.x, draw.y, draw.z);
    bounding_sphere.radius = draw.radius;

    Triangle triangle;
    triangle.v0 = decode_position(bounding_sphere, v0.position);
    triangle.v1 = decode_position(bounding_sphere, v1.position);
    triangle.v2 = decode_position(bounding_sphere, v2.position);

    triangle.v0 = (instance.transform * vec4(triangle.v0, 1.0)).xyz;
    triangle.v1 = (instance.transform * vec4(triangle.v1, 1.0)).xyz;
    triangle.v2 = (instance.transform * vec4(triangle.v2, 1.0)).xyz;

    Barycentric barycentric = barycentric(triangle, vec2(0.0), vec2(constants.screen_size));

    vec3 world_space_position = interp_vec3(
        barycentric.lambda,
        vec3[3](triangle.v0, triangle.v1, triangle.v2)
    );

    TangentFrame tangent_frame0 = decode_tangent_frame(v0.tangent_frame);
    TangentFrame tangent_frame1 = decode_tangent_frame(v1.tangent_frame);
    TangentFrame tangent_frame2 = decode_tangent_frame(v2.tangent_frame);

    mat3 normal_transform = mat3(instance.normal_transform);

    vec3 world_space_normal = interp_vec3(barycentric.lambda, vec3[3](
        normal_transform * tangent_frame0.normal,
        normal_transform * tangent_frame1.normal,
        normal_transform * tangent_frame2.normal
    ));

    vec3 world_space_tangent = interp_vec3(barycentric.lambda, vec3[3](
        normal_transform * tangent_frame0.tangent,
        normal_transform * tangent_frame1.tangent,
        normal_transform * tangent_frame2.tangent
    ));

    float bitangent_sign = interp_scalar(barycentric.lambda, float[3](
        tangent_frame0.bitangent_sign,
        tangent_frame1.bitangent_sign,
        tangent_frame2.bitangent_sign
    ));

    vec3 world_space_bitangent = sign(bitangent_sign)
        * cross(world_space_normal, world_space_tangent);

    vec2 texcoords[3] = vec2[3](
        decode_texcoord(v0.texcoord),
        decode_texcoord(v1.texcoord),
        decode_texcoord(v2.texcoord)
    );

    vec2 texcoord = interp_vec2(barycentric.lambda, texcoords);
    vec2 texcoord_ddx = interp_vec2(barycentric.ddx, texcoords);
    vec2 texcoord_ddy = interp_vec2(barycentric.ddy, texcoords);

    Material material = materials[uint(v0.material)];

    vec4 tangent_normal = textureGrad(
        textures[material.normal_texture],
        texcoord,
        texcoord_ddx,
        texcoord_ddy
    );

    vec3 normal = octahedron_decode(tangent_normal.xy);
    normal = normalize(
        normal.x * world_space_tangent
            + normal.y * world_space_bitangent
            + normal.z * world_space_normal
    );

    vec3 albedo = textureGrad(
        textures[material.albedo_texture],
        texcoord,
        texcoord_ddx,
        texcoord_ddy
    ).rgb;

    vec4 specular_parameters = textureGrad(
        textures[material.specular_texture],
        texcoord,
        texcoord_ddx,
        texcoord_ddy
    );

    float metallic = specular_parameters.r * material.metallic;

    float roughness = specular_parameters.g * material.roughness;
    roughness *= roughness;

    vec3 emissive = textureGrad(
        textures[material.emissive_texture],
        texcoord,
        texcoord_ddx,
        texcoord_ddy
    ).rgb;

    vec4 gbuffer0_data = vec4(octahedron_encode(normal), 1.0, 1.0);
    vec4 gbuffer1_data = vec4(albedo, metallic);
    vec4 gbuffer2_data = vec4(emissive, roughness);

    imageStore(gbuffer0, ivec2(pixel_index), gbuffer0_data);
    imageStore(gbuffer1, ivec2(pixel_index), gbuffer1_data);
    imageStore(gbuffer2, ivec2(pixel_index), gbuffer2_data);
}