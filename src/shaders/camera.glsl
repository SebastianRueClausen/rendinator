#ifndef CAMERA_GLSL
#define CAMERA_GLSL

struct Proj {
	mat4 mat;
	mat4 inverse_proj;

	vec2 surface_size;

	float z_near;
	float z_far;
};

struct View {
	vec4 eye;
	mat4 mat;
	mat4 proj_view;
};

#endif
