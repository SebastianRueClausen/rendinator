#version 450
#pragma shader_stage(compute)

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_nonuniform_qualifier: require
#extension GL_EXT_control_flow_attributes: require

#define FRUSTRUM_CULL
#define LOD
#define OCCLUSION_CULL

#include "mesh.glsl"
#include "camera.glsl"

const uint THREAD_COUNT = 64;

layout (local_size_x = THREAD_COUNT) in;

layout (std430, push_constant) uniform CullInfo {
	vec4 frust_planes[6];

	float lod_base;
	float lod_step;

	float pyramid_width;
	float pyramid_height;

	uint pyramid_mip_count;
	uint phase;

} cull_info;

layout (std140, set = 0, binding = 0) readonly uniform ProjBuf {
	Proj proj;
};

layout (std140, set = 0, binding = 1) readonly uniform ViewBuf {
	View view;
};

layout (std430, set = 1, binding = 0) writeonly buffer DrawBuf {
	DrawCommand draw_commands[];
};

layout (std430, set = 1, binding = 1) buffer DrawCountBuf {
	uint command_count;
	uint primitive_count;
};

layout (std430, set = 1, binding = 2) buffer DrawFlagBuf {
	uint draw_flags[];
};

layout (std430, set = 2, binding = 0) readonly buffer Instances {
	InstanceData instances[];
};

layout (std430, set = 2, binding = 1) readonly buffer Primitives {
	Primitive primitives[];	
};

// Not used
layout (std430, set = 2, binding = 2) readonly buffer Verts {
	Vert verts;
};

// Not used
layout (set = 2, binding = 3) uniform sampler2D textures[];

layout (set = 3, binding = 0) uniform sampler2D depth_pyramid[];

bool has_been_drawn(const uint primitive) {
	return draw_flags[primitive] == 1;
}

// Find the bounding box of a sphere in screen-space.
bool project_sphere(const vec3 center, const float z_near, const float radius, out vec4 aabb) {
	// Check if we are inside the sphere. In which case we can't do any occlusion culling.
	if (center.z < radius + z_near) {
		return false;
	}

	const float p00 = proj.mat[0][0];
	const float p11 = proj.mat[1][1];

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

	// Transform from clip-space to screen-space.
	aabb = aabb.xwzy * vec4(0.5f, -0.5f, 0.5f, -0.5f) + vec4(0.5f);

	return true;
}

void main() {
	const uint draw_id = gl_GlobalInvocationID.x;

	if (draw_id >= primitive_count) {
		return;
	}

	const bool phase_2 = cull_info.phase == 2;
	const bool has_been_drawn = has_been_drawn(draw_id);

	// In this case the primitive was not visible after rendering last frame.
	if (!phase_2 && !has_been_drawn) {
		return;
	}

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

	[[unroll]]
	for (uint i = 0; i < 6; i++) {
		culled = culled || dot(vec4(center, 1.0), cull_info.frust_planes[i]) + radius < 0.0;
	}

	bool visible = !culled;
#else
	bool visible = true;
#endif

	vec3 view_center = (view.mat * vec4(center, 1.0)).xyz;
	view_center.z *= -1;

#ifdef OCCLUSION_CULL
	if (phase_2 && visible) {
		vec4 aabb;

		if (project_sphere(view_center, proj.z_near, radius, aabb)) {
			const float width = (aabb.z - aabb.x) * cull_info.pyramid_width;
			const float height = (aabb.w - aabb.y) * cull_info.pyramid_height;

			const float mip = ceil(log2(max(width, height)));
			const uint level = min(uint(mip), cull_info.pyramid_mip_count) + 1;

			const vec4 samples = textureGather(depth_pyramid[level], (aabb.xy + aabb.zw) * 0.5, 0);
			const float depth = max(samples.x, max(samples.y, max(samples.z, samples.w)));

			const float depth_sphere = proj.z_near / (view_center.z - radius);

			visible = visible && depth_sphere >= depth;
		}
	}
#endif

	// If the primitive is visible, there is two scenarios here.
	//
	// If we're in the 1st phase then the draw buffers should always be updated.
	// If we're in the 2nd phase, it should only be updated if it wasn't drawn after the 1st phase.
	if (visible && (!phase_2 || !has_been_drawn)) {
		const uint command_id = atomicAdd(command_count, 1);

#ifdef LOD
		const float dist = max(length(view_center) - radius, 0.0);
		const float lod_factor = log2(dist / cull_info.lod_base) / log2(cull_info.lod_step);
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

	if (phase_2) {
		draw_flags[draw_id] = uint(visible);
	}
}
