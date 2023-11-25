#version 450
#extension GL_GOOGLE_include_directive: require
#extension GL_ARB_shader_draw_parameters: require

#include "../scene.glsl"
#include "../constants.glsl"

layout (binding = 0) uniform ConstantData {
    Constants constants;
};

layout (binding = 1) buffer Vertices {
    Vertex vertices[];
};

layout (binding = 2) buffer Meshes {
    Mesh meshes[];
};

layout (binding = 3) buffer Instances {
    Instance instances[];
};

layout (binding = 4) buffer Draws {
    Draw draws[];
};

layout (binding = 5) buffer Materials {
    Material materials[];
};

layout (binding = 6) uniform sampler2D textures[];

layout (location = 0) out vec4 world_position;
layout (location = 1) out vec3 world_normal;
layout (location = 2) out vec3 world_tangent;
layout (location = 3) out vec3 world_bitangent;
layout (location = 4) out vec2 texcoord;
layout (location = 5) out flat uint material;

void main() {
    Vertex vertex = vertices[gl_VertexIndex];
    Instance instance = instances[gl_InstanceIndex];
    Draw draw = draws[gl_DrawIDARB];

    BoundingSphere bounding_sphere;
    bounding_sphere.center = vec3(draw.x, draw.y, draw.z);
    bounding_sphere.radius = draw.radius;

    vec3 position = decode_position(bounding_sphere, vertex.position);
    TangentFrame tangent_frame = decode_tangent_frame(vertex.tangent_frame);

    world_normal = normalize(mat3(instance.normal_transform) * tangent_frame.normal);
    world_tangent = normalize(mat3(instance.transform) * tangent_frame.tangent);
    world_bitangent = tangent_frame.bitangent_sign * cross(world_normal, world_tangent);
    texcoord = decode_texcoord(vertex.texcoord);
    material = uint(vertex.material);
    world_position = instance.transform * vec4(position, 1.0);
    gl_Position = constants.proj_view * world_position;

    // color = vec4(tangent_frame.normal * 0.5 + vec3(0.5), 1.0);
}