#ifndef BRDF
#define BRDF

const float PI = 3.14159265358979323846264;

float norm_dist(const float norm_dot_half, const float rough) {
	const float a = rough * rough;
	const float ggx = fma(fma(norm_dot_half, a, -norm_dot_half), norm_dot_half, 1.0);
	return a / (PI * ggx * ggx);
}

float visibility(const float norm_dot_view, const float norm_dot_light, const float rough) {
	const float a = rough * rough;

	const float mask = norm_dot_view * sqrt(fma(fma(-norm_dot_light, a, norm_dot_light), norm_dot_light, a));
	const float shadow = norm_dot_light * sqrt(fma(fma(-norm_dot_view, a, norm_dot_view), norm_dot_view, a));

	return 0.5 / (mask + shadow);
}

// Faster than `pow(x, 5.0)`.
float pow5(const float val) {
	return val * val * val * val * val;
}

vec3 fresnel_schlick(const vec3 f0, const float view_dot_half) {
	return f0 + (vec3(1.0) - f0) * pow5(1.0 - view_dot_half);
}

float fresnel_schlick(const float cos_theta, const float f0, const float f90) {
	return f0 + (f90 - f0) * pow5(1.0 - cos_theta);
}

float lambert_diffuse() {
	return 1.0 / PI;
}

float burley_diffuse(
	const float norm_dot_view,
	const float norm_dot_light,
	const float light_dot_half,
	const float rough
) {
	const float f90 = fma(light_dot_half * light_dot_half, 2.0 * rough, 0.5);
	const float light_scatter = fresnel_schlick(norm_dot_light, 1.0, f90);
	const float view_scatter = fresnel_schlick(norm_dot_view, 1.0, f90);
	return light_scatter * view_scatter * lambert_diffuse();
}

float geometric_aa(const vec3 normal, const float rough) {
  const float sigma2 = 0.25;
  const float kappa  = 0.18;

	const vec3 dndu = dFdx(normal);
  const vec3 dndv = dFdy(normal);

  const float variance = sigma2 * (dot(dndu, dndu) + dot(dndv, dndv));
  const float kernel_roughness = min(2.0 * variance, kappa);

  return clamp(rough + kernel_roughness, 0.0, 1.0);	
}

#endif
