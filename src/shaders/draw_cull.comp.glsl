#version 450
#pragma shader_stage(compute)

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_nonuniform_qualifier: require

#define FRUSTRUM_CULL
#define LOD
#define OCCLUSION_CULL

#include "mesh.glsl"

const uint THREAD_COUNT = 64;

layout (local_size_x = THREAD_COUNT) in;

layout (std430, push_constant) uniform CullInfo {
	vec4 frust_planes[6];

	float z_near;
	float z_far;

	float lod_base;
	float lod_step;

	float pyramid_width;
	float pyramid_height;

} cull_info;

layout (std140, set = 0, binding = 0) readonly uniform Proj {
	mat4 proj;
	mat4 inverse_proj;
	vec2 screen_dimensions;
};

layout (std140, set = 0, binding = 1) readonly uniform View {
	vec4 eye;
	mat4 view;
	mat4 proj_view;
};

layout (std430, set = 1, binding = 0) readonly buffer Instances {
	InstanceData instances[];
};

layout (std430, set = 1, binding = 1) writeonly buffer DrawBuffer {
	DrawCommand draw_commands[];
};

layout (std430, set = 1, binding = 2) readonly buffer Primitives {
	Primitive primitives[];	
};

layout (std430, set = 1, binding = 3) buffer DrawCount {
	uint command_count;
	uint primitive_count;
};

layout (set = 2, binding = 0) uniform sampler2D depth_pyramid[];

bool project_sphere(const vec3 center, const float radius, out vec4 aabb) {
	if (center.z < radius + cull_info.z_near) return false;

	const float p00 = proj[0][0];
	const float p11 = proj[1][1];

	const vec2 cx = -center.xz;
	const vec2 vx = vec2(sqrt(dot(cx, cx) - radius * radius), radius);

	const vec2 minx = mat2(vx.x, vx.y, -vx.y, vx.x) * cx;
	const vec2 maxx = mat2(vx.x, -vx.y, vx.y, vx.x) * cx;

	const vec2 cy = -center.yz;
	const vec2 vy = vec2(sqrt(dot(cy, cy) - radius * radius), radius);
	const vec2 miny = mat2(vy.x, vy.y, -vy.y, vy.x) * cy;
	const vec2 maxy = mat2(vy.x, -vy.y, vy.y, vy.x) * cy;

	aabb = vec4(
			minx.x / minx.y * p00,
			miny.x / miny.y * p11,
			maxx.x / maxx.y * p00,
			maxy.x / maxy.y * p11
	);

	// Transform from clip-space to uv-space.
	aabb = aabb.xwzy * vec4(0.5f, -0.5f, 0.5f, -0.5f) + vec4(0.5f);

	return true;
}

void main() {
	const uint draw_id = gl_GlobalInvocationID.x;

	if (draw_id >= primitive_count) return;

	const Primitive primitive = primitives[draw_id];
	const InstanceData instance = instances[primitive.instance];

	const float scale = max(
			length(vec3(
				instance.transform[0][0],
				instance.transform[0][1],
				instance.transform[0][2]
			)),
			max(
				length(vec3(
					instance.transform[1][0],
					instance.transform[1][1],
					instance.transform[1][2]
				)),
				length(vec3(
					instance.transform[2][0],
					instance.transform[2][1],
					instance.transform[2][2]
				))
			)
	);

	vec3 center = (instance.transform * primitive.center).xyz;
	const float radius = primitive.radius * scale;

#ifdef FRUSTRUM_CULL
	bool culled = false;

	for (uint i = 0; i < 6; i++) {
		culled = culled || dot(vec4(center, 1.0), cull_info.frust_planes[i]) + radius < 0.0;
	}

	bool visible = !culled;
#else
	bool visible = true;
#endif

#ifdef OCCLUSION_CULL
	vec4 aabb;
	vec3 view_center = (view * vec4(center, 1.0)).xyz;

	view_center.y *= -1;
	view_center.z *= -1;

	if (project_sphere(view_center, radius, aabb)) {
		const float width = (aabb.z - aabb.x) * cull_info.pyramid_width;
		const float height = (aabb.w - aabb.y) * cull_info.pyramid_height;

		const uint level = uint(log2(max(width, height))) + 1;

		const float depth = texture(depth_pyramid[level], (aabb.xy + aabb.zw) * 0.5).r;
		const float depth_sphere = cull_info.z_near / (view_center.z - radius);

		visible = visible && depth_sphere > depth;
	}
#endif

	if (visible) {
		const uint command_id = atomicAdd(command_count, 1);

#ifdef LOD
		const float lod_factor = log2(length(center) / cull_info.lod_base) / log2(cull_info.lod_step);
		const uint lod_index = min(uint(max(lod_factor + 1, 0)), primitive.lod_count - 1);
#else
		const uint lod_index = 0;
#endif

		const Lod lod = primitive.lods[lod_index];	

		draw_commands[command_id] = DrawCommand(
			lod.index_count,
			1,
			lod.first_index,
			primitive.vertex_offset,
			primitive.instance,
			primitive.albedo_map,
			primitive.specular_map,
			primitive.normal_map
		);
	}
}
