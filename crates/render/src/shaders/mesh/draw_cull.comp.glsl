#version 450
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_control_flow_attributes: require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#include "../scene.glsl"
#include "../constants.glsl"

layout (constant_id = 0) const bool POST_DRAW = false;

layout (binding = 0) uniform ConstantData {
    Constants constants;
};

layout (binding = 1) buffer Meshes {
    Mesh meshes[];
};

layout (binding = 2) buffer Draws {
    Draw draws[];
};

layout (binding = 3) buffer Instances {
    Instance instances[];
};

layout (binding = 4) buffer DrawCommands {
    DrawCommand draw_commands[];
};

layout (binding = 5) buffer DrawCount {
    uint draw_command_count;
};

layout (binding = 6) uniform sampler2D depth_pyramid;

layout(push_constant) uniform PushConstants {
	uint total_draw_count;
};

bool frustrum_cull(vec3 center, float radius) {
    bool visible = true;

    [[unroll]]
    for (uint i = 0; i < 6; i++) {
        // Note that w of the frustrum planes is the distance from the plane to the origin.
        visible = visible && dot(vec4(center, 1.0), constants.frustrum_planes[i]) > -radius;
    }

    return visible;
}

bool proj_sphere(vec3 center, float radius, out vec4 aabb) {
	if (center.z < radius + constants.z_near) {
		return false;
    }

	vec3 cr = center * radius;
	float czr2 = center.z * center.z - radius * radius;

	float vx = sqrt(center.x * center.x + czr2);
	float min_x = (vx * center.x - cr.z) / (vx * center.z + cr.x);
	float max_x = (vx * center.x + cr.z) / (vx * center.z - cr.x);

	float vy = sqrt(center.y * center.y + czr2);
	float min_y = (vy * center.y - cr.z) / (vy * center.z + cr.y);
	float max_y = (vy * center.y + cr.z) / (vy * center.z - cr.y);

    float p00 = constants.proj[0][0];
    float p11 = constants.proj[1][1];

	aabb = vec4(min_x * p00, min_y * p11, max_x * p00, max_y * p11);
	aabb = aabb.xwzy * vec4(0.5, -0.5, 0.5, -0.5) + vec4(0.5); // clip space -> uv space

	return true;
}

bool occlusion_cull(vec3 center, float radius) {
    vec2 pyramid_extent = vec2(textureSize(depth_pyramid, 0).xy);

    vec4 aabb;
    if (proj_sphere(center, radius, aabb)) {
        float width = (aabb.z - aabb.x) * pyramid_extent.x;
        float height = (aabb.w - aabb.y) * pyramid_extent.y;

        float level = floor(log2(max(width, height)));

        float depth = textureLod(depth_pyramid, (aabb.xy + aabb.zw) * 0.5, level).x;
        float depth_sphere = constants.z_near / (center.z - radius);

        return depth_sphere > depth;
    } else {
        return true;
    }
}

float length_squared(vec3 x) {
    return dot(x, x);
}

uint select_lod(vec3 center, float radius, uint lod_count) {
    const float lod_base = 10.0;
    const float lod_step = 1.5;

    float index = log2(length(center) / lod_base) / log2(lod_step);
	return min(uint(max(index + 1, 0)), lod_count - 1);
}

void main() {
    uint draw_index = gl_GlobalInvocationID.x;

    if (draw_index >= total_draw_count) {
        return;
    }

    bool was_visible = draws[draw_index].visible != 0;

    if (!POST_DRAW && !was_visible) {
        return;
    }

    Draw draw = draws[draw_index];
    Mesh mesh = meshes[draw.mesh_index];

    Instance instance = instances[draw.instance_index];

    vec3 world_space_center = (instance.transform * vec4(mesh.x, mesh.y, mesh.z, 1.0)).xyz;
    vec3 center = (constants.view * vec4(world_space_center, 1.0)).xyz;

    float x_scale = length_squared(instance.transform[0].xyz);
    float y_scale = length_squared(instance.transform[1].xyz);
    float z_scale = length_squared(instance.transform[1].xyz);
    float scale = sqrt(max(x_scale, max(y_scale, z_scale)));

    float radius = mesh.radius * scale;

    bool visible = frustrum_cull(world_space_center, radius);

    if (POST_DRAW && visible) {
        visible = visible && occlusion_cull(center, radius);
    }

    if (visible && (!POST_DRAW || !was_visible)) {
        // uint lod_index = select_lod(center, radius, mesh.lod_count);
        uint lod_index = 0;
        Lod lod = mesh.lods[lod_index];

        uint command_index = atomicAdd(draw_command_count, 1);

        DrawIndexedIndirectCommand command;
        command.index_count = lod.index_count;
        command.instance_count = 1;
        command.first_index = lod.index_offset;
        command.vertex_offset = int(mesh.vertex_offset);
        command.first_instance = draw.instance_index;

        draw_commands[command_index].radius = mesh.radius;
        draw_commands[command_index].command = command;
        draw_commands[command_index].x = mesh.x;
        draw_commands[command_index].y = mesh.y;
        draw_commands[command_index].z = mesh.z;
    }

    if (POST_DRAW) {
        draws[draw_index].visible = visible ? 1 : 0;
    }
}
