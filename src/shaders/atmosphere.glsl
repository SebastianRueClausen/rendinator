#ifndef ATMOSPHERE_INC
#define ATMOSPHERE_INC

const float PI = 3.14159265358979323846264;

// Earth radius in meters.
const float EARTH_RADIUS = 6371000;

// The center of the earth assuming `vec3(0.0)` is on the surface.
const vec3 EARTH_CENTER = vec3(0.0, EARTH_RADIUS, 0.0);

// The height of the atmosphere in meters.
const float ATMOSPHERE_HEIGHT = 100000.0;

// Height of which mie and reyleigh scattering happens in the atmosphere in meters.
const float MIE_HEIGHT = ATMOSPHERE_HEIGHT * 0.012;
const float RAYLEIGH_HEIGHT = ATMOSPHERE_HEIGHT * 0.08;

// The ozone layer start at around 10 km height and has a width of 30 km.
const float OZONE_CENTER_HEIGHT = 25000.0;
const float OZONE_WIDTH = 30000.0;

const vec3 RAYLEIGH_COEFF = vec3(5.802, 13.558, 33.100) * 1e-6;
const vec3 MIE_COEFF = vec3(3.996) * 1e-6;
const vec3 OZONE_COEFF = vec3(0.650, 1.881, 0.085) * 1e-6;

// The amount of time a ray should sample.
const uint RAY_SAMPLE_COUNT = 16;

vec2 ray_sphere_intersection(
		vec3 origin,
		const vec3 dir,
		const vec3 center,
		const float radius
) {
	origin -= center;

	const float a = dot(dir, dir);
	const float b = 2.0 * dot(dir, origin);
	const float c = dot(origin, origin) - (radius * radius);

	const float disc = (b * b) - 4 * a * c;

	if (disc < 0.0) {
		return vec2(-1);
	}

	return vec2(-b - disc, -b + sqrt(disc)) / (2.0 * a);
}

float atmosphere_height(const vec3 world_pos) {
	return distance(world_pos, EARTH_CENTER) - EARTH_RADIUS;
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

vec3 calc_optical_depth(const vec3 origin, const vec3 dir) {
	const vec2 intersection = ray_sphere_intersection(origin, dir, EARTH_CENTER, EARTH_RADIUS);
	const float ray_length = intersection.y;

	const float step_size = ray_length / RAY_SAMPLE_COUNT;

	vec3 optical_depth = vec3(0);

	for (uint i = 0; i < RAY_SAMPLE_COUNT; ++i) {
		const vec3 pos = origin + dir * (i + 0.5) * step_size;
		const float height = atmosphere_height(pos);
		const vec3 density = vec3(
			rayleigh_density(height),
			mie_density(height),
			ozone_density(height)
		);

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

vec3 integrate_scattering(
	vec3 ray_origin,
	const vec3 ray_dir,
	const vec3 light_dir,
	const vec3 light_color
) {
	const float ray_height = atmosphere_height(ray_origin);
	const float sample_dist_exp = 7.0;
	
	const vec2 intersection =
		ray_sphere_intersection(ray_origin, ray_dir, EARTH_CENTER, EARTH_RADIUS + ATMOSPHERE_HEIGHT);

	float ray_len = intersection.y;

	if (intersection.x > 0.0) {
		ray_origin += ray_dir * intersection.x;	
		ray_len -= intersection.x;
	}

	const float cos_theta = dot(ray_dir, light_dir);

	const float rayleigh_phase = rayleigh_phase(cos_theta);
	const float mie_phase = mie_phase(cos_theta);

	vec3 optical_depth = vec3(0.0);
	vec3 rayleigh = vec3(0.0);
	vec3 mie = vec3(0.0);

	float prev_ray_time = 0.0;

	for (uint i = 0; i < RAY_SAMPLE_COUNT; ++i) {
		const float ray_time = pow(float(i) / RAY_SAMPLE_COUNT, sample_dist_exp) * ray_len;
		const float step_size = ray_time - prev_ray_time;

		const vec3 pos = ray_origin + ray_dir * mix(prev_ray_time, ray_time, 0.5);
		const float height = atmosphere_height(pos);

		const vec3 density =
			vec3(rayleigh_density(height), mie_density(height), ozone_density(height));

		optical_depth += density * step_size;

		const vec3 view_transmittance = absorb(optical_depth);
		const vec3 optical_depth_light = calc_optical_depth(pos, light_dir);
		const vec3 light_transmittance = absorb(optical_depth_light);

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

		prev_ray_time = ray_time;
	}
	
	const vec3 color = (rayleigh * RAYLEIGH_COEFF + mie * MIE_COEFF) * light_color * 20.0;
	const vec3 transmittance = absorb(optical_depth);

	return color;
}

#endif
