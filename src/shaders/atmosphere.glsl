#ifndef ATMOSPHERE_GLSL
#define ATMOSPHERE_GLSL

#extension GL_GOOGLE_include_directive: require

#include "tonemap.glsl"

const float PI = 3.14159265358979323846264;

const float EARTH_RADIUS = 6371000;
const vec3 EARTH_CENTER = vec3(0.0, EARTH_RADIUS, 0.0);

const float ATMOS_HEIGHT = 100000.0;
const float MIE_HEIGHT = ATMOS_HEIGHT * 0.012;
const float RAYLEIGH_HEIGHT = ATMOS_HEIGHT * 0.08;

// The ozone layer start at around 10 km height and has a width of 30 km.
const float OZONE_CENTER_HEIGHT = 25000.0;
const float OZONE_WIDTH = 30000.0;

const vec3 RAYLEIGH_COEFF = vec3(5.802, 13.558, 33.100) * 1e-6;
const vec3 MIE_COEFF = vec3(3.996) * 1e-6;
const vec3 OZONE_COEFF = vec3(0.650, 1.881, 0.085) * 1e-6;

// The amount of time a ray should sample.
const uint RAY_SAMPLE_COUNT = 16;

const float SUN_ANGULAR_DIAMETER = cos(0.01);

// The intersection between a ray and sphere.
//
// Assume theres always a hit.
vec2 ray_sphere(
	const vec3 origin,
	const vec3 dir,
	const vec3 center,
	const float radius
) {
	const vec3 ray_origin = origin - center;

	const float a = dot(dir, dir);
	const float b = 2.0 * dot(ray_origin, dir);
	const float c = dot(ray_origin, ray_origin) - (radius * radius);

	// Square root of discriminant.
	const float disc = sqrt(b * b - 4 * a * c);

	return vec2(-b - disc, -b + disc) / (2.0 * a);
}

float rayleigh_density(const float height) {
	return exp(-max(0.0, height / RAYLEIGH_HEIGHT));
}

float mie_density(const float height) {
	return exp(-max(0.0, height / MIE_HEIGHT));
}

float ozone_density(const float height) {
	return max(0, 1 - abs(height - OZONE_CENTER_HEIGHT) / (OZONE_WIDTH / 2.0));
}

float rayleigh_phase(const float cos_theta) {
	return 3.0 * (1.0 + cos_theta * cos_theta) / (16.0 * PI);
}

float mie_phase(const float cos_theta) {
	const float G = 0.85;
	const float K = 1.55 * G - 0.55 * G * G * G;

	return (1.0 - K * K) / ((4.0 * PI) * pow(1.0 - K * cos_theta, 2));
}

vec3 int_optical_depth(const vec3 origin, const vec3 dir) {
	const vec2 intersection = ray_sphere(origin, dir, EARTH_CENTER, EARTH_RADIUS + ATMOS_HEIGHT);
	const float ray_length = intersection.y;

	const float step_size = ray_length / RAY_SAMPLE_COUNT;

	vec3 optical_depth = vec3(0);

	const uint SAMPLE_COUNT = 8;
	for (uint i = 0; i < SAMPLE_COUNT; ++i) {
		const vec3 pos = origin + dir * (i + 0.5) * step_size;
		const float height = distance(pos, EARTH_CENTER) - EARTH_RADIUS;

		const vec3 density =
			vec3(rayleigh_density(height), mie_density(height), ozone_density(height));

		optical_depth += density * step_size;
	}

	return optical_depth;
}

vec3 absorb(const vec3 optical_depth) {
	return exp(-(
		optical_depth.x * RAYLEIGH_COEFF
			+ optical_depth.y * MIE_COEFF
			+ optical_depth.z * OZONE_COEFF
	));
}

vec3 int_scattering(
	const vec3 ray_origin,
	const vec3 ray_dir,
	const vec3 light_dir,
	const vec3 light_color
) {
	// Intersection with the edge of the atmosphere.
	const vec2 intersection = ray_sphere(ray_origin, ray_dir, EARTH_CENTER, EARTH_RADIUS + ATMOS_HEIGHT);

	// The length of the ray will be from the origin to the edge of the atmosphere.
	const float ray_len = intersection.y;

	const float cos_theta = dot(ray_dir, light_dir);
	const float rayleigh_phase = rayleigh_phase(cos_theta);
	const float mie_phase = mie_phase(cos_theta);

	vec3 optical_depth = vec3(0.0);
	vec3 rayleigh = vec3(0.0);
	vec3 mie = vec3(0.0);

	// The distance marched before each iteration.
	float prev_marched = 0.0;

	const uint SAMPLE_COUNT = 16;
	for (uint i = 0; i < SAMPLE_COUNT; ++i) {
		// Sample at greater and greater intervals. 7 is just an arbitrary exponent.
		const float marched = pow(float(i) / SAMPLE_COUNT, 7) * ray_len;
		const float step_size = marched - prev_marched;

		const vec3 pos = ray_origin + ray_dir * marched;
		const float height = distance(pos, EARTH_CENTER) - EARTH_RADIUS;

		const vec3 density =
			vec3(rayleigh_density(height), mie_density(height), ozone_density(height));

		optical_depth += density * step_size;

		const vec3 view_transmittance = absorb(optical_depth);
		const vec3 light_transmittance = absorb(int_optical_depth(pos, light_dir));

		rayleigh += view_transmittance
			* light_transmittance
			* rayleigh_phase
			* density.x
			* step_size;

		mie += view_transmittance
			* light_transmittance
			* mie_phase
			* density.y
			* step_size;

		prev_marched = marched;
	}

	const vec3 transmittance = absorb(optical_depth);
	const vec3 sun = vec3(
		max(0.0, pow(cos_theta, 42.0) * 18000 + pow(cos_theta, 80.0) * 12.0)
	) * transmittance;


	const float EXPOSURE = 20.0;
	const vec3 sky = (rayleigh * RAYLEIGH_COEFF + mie * MIE_COEFF) * light_color * EXPOSURE;

	return aces_approx_tonemap(sky) + sun;
}

#endif
