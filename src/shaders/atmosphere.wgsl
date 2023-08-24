#import consts
#import util
#import light

const EARTH_RADIUS = 6371000.0;
const EARTH_CENTER = vec3f(0.0, -6371000.0, 0.0);

const ATMOS_HEIGHT = 100000.0;
// const MIE_HEIGHT = ATMOS_HEIGHT * 0.012;
const MIE_HEIGHT = 1200.0;
// const RAYLEIGH_HEIGHT = ATMOS_HEIGHT * 0.08;
const RAYLEIGH_HEIGHT = 8000.0;

// The ozone layer start at around 10 km height and has a width of 30 km.
const OZONE_CENTER_HEIGHT = 25000.0;
const OZONE_WIDTH = 30000.0;

// const RAYLEIGH_COEFF = vec3f(5.802, 13.558, 33.100) * 1e-6;
const RAYLEIGH_COEFF = vec3f(0.000005802, 0.000013558, 0.0000331);
// const MIE_COEFF = vec3f(3.996) * 1e-6;
const MIE_COEFF = vec3f(0.000003996, 0.000003996, 0.000003996);
// const OZONE_COEFF = vec3f(0.650, 1.881, 0.085) * 1e-6;
const OZONE_COEFF = vec3f(6.5e-7, 0.000001881, 8.5e-8);

// The amount of time a ray should sample.
const RAY_SAMPLE_COUNT = 16;

fn ray_sphere_intersection(ray: util::Ray, sphere: util::Sphere) -> vec2f {
    let ray_origin = ray.origin - sphere.center;

    let a = dot(ray.direction, ray.direction);
    let b = 2.0 * dot(ray_origin, ray.direction);
    let c  = dot(ray_origin, ray_origin) - (sphere.radius * sphere.radius);

    let sqrt_disc = sqrt(b * b - 4.0 * a * c);

    return vec2f((-b) - sqrt_disc, (-b) + sqrt_disc) / (2.0 * a);
}

fn rayleigh_density(height: f32) -> f32 {
	return exp(-max(0.0, height / RAYLEIGH_HEIGHT));
}

fn mie_density(height: f32) -> f32 {
	return exp(-max(0.0, height / MIE_HEIGHT));
}

fn ozone_density(height: f32) -> f32 {
	return max(0.0, 1.0 - abs(height - OZONE_CENTER_HEIGHT) / (OZONE_WIDTH / 2.0));
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
	return 3.0 * (1.0 + cos_theta * cos_theta) / (16.0 * util::PI);
}

fn mie_phase(cos_theta: f32) -> f32 {
	let G = 0.85;
	let K = 1.55 * G - 0.55 * G * G * G;
	return (1.0 - K * K) / ((4.0 * util::PI) * pow(1.0 - K * cos_theta, 2.0));
}

struct OpticalDepth {
    rayleigh: f32,
    mie: f32,
    ozone: f32,
}

fn integrate_optical_depth(ray: util::Ray, atmosphere: util::Sphere) -> OpticalDepth {
    let intersection = ray_sphere_intersection(ray, atmosphere);
    let ray_lenght = intersection.y;

    let step_size = ray_lenght / f32(RAY_SAMPLE_COUNT);

    var optical_depth: OpticalDepth;

    for (var i = 0; i < 8; i += 1) {
        let position = ray.origin + ray.direction * (f32(i) + 0.5) * step_size;
        let height = distance(position, EARTH_CENTER) - EARTH_RADIUS;

        optical_depth.rayleigh += rayleigh_density(height) * step_size;
        optical_depth.mie += mie_density(height) * step_size;
        optical_depth.ozone += ozone_density(height) * step_size;
    }

    return optical_depth;
}

fn absorb(optical_depth: OpticalDepth) -> vec3f {
	return exp(-(
		optical_depth.rayleigh * RAYLEIGH_COEFF
			+ optical_depth.mie * MIE_COEFF
			+ optical_depth.ozone * OZONE_COEFF
	));
}

fn integrate_atmospheric_scattering(
    light: light::DirectionalLight,
    ray: util::Ray,
    atmosphere: util::Sphere,
) -> vec3f {
    let intersection = ray_sphere_intersection(ray, atmosphere);

	// The length of the ray will be from the origin to the edge of the atmosphere.
    let ray_length = intersection.y;

    let cos_theta = dot(ray.direction, light.direction.xyz);

	let rayleigh_phase = rayleigh_phase(cos_theta);
	let mie_phase = mie_phase(cos_theta);

	var optical_depth: OpticalDepth;
    var rayleigh = vec3f(0.0);
    var mie = vec3f(0.0);

    var prev_marched = 0.0;

    for (var i = 0; i < 16; i += 1) {
		// Sample at greater and greater intervals. 7 is just an arbitrary exponent.
		let marched = pow(f32(i) / 16.0, 7.0) * ray_length;
		let step_size = marched - prev_marched;

		let position = ray.origin + ray.direction * marched;
		let height = distance(position, EARTH_CENTER) - EARTH_RADIUS;

        let rayleigh_density = rayleigh_density(height);
        let mie_density = mie_density(height);

        optical_depth.rayleigh += rayleigh_density * step_size;
        optical_depth.mie += mie_density * step_size;
        optical_depth.ozone += ozone_density(height) * step_size;

		let view_transmittance = absorb(optical_depth);

	    var light_ray: util::Ray;
		light_ray.origin = position;
		light_ray.direction = light.direction.xyz;

		let light_transmittance = absorb(integrate_optical_depth(light_ray, atmosphere));

		rayleigh += view_transmittance
			* light_transmittance
			* rayleigh_phase
			* rayleigh_density
			* step_size;

		mie += view_transmittance
			* light_transmittance
			* mie_phase
			* mie_density
			* step_size;

		prev_marched = marched;
    }

	let transmittance = absorb(optical_depth);

    let exposure = 20.0;
    let sky = (rayleigh * RAYLEIGH_COEFF + mie * MIE_COEFF) * light.irradiance.xyz * exposure;

    return sky;
}

fn cube_map_face(index: u32) -> mat3x3f {
    switch index {
        case 0u: {
            return mat3x3f(
                vec3f(0.0, 0.0, 1.0),
                vec3f(0.0, 1.0, 0.0),
                vec3f(1.0, 0.0, 0.0),
            );
        }
        case 1u: {
            return mat3x3f(
                vec3f(0.0, 0.0, -1.0),
                vec3f(0.0, 1.0, 0.0),
                vec3f(-1.0, 0.0, 0.0),
            );
        }
        case 2u: {
            return mat3x3f(
                vec3f(1.0, 0.0, 0.0),
                vec3f(0.0, 0.0, -1.0),
                vec3f(0.0, -1.0, 0.0),
            );
        }
        case 3u: {
            return mat3x3f(
                vec3f(1.0, 0.0, 0.0),
                vec3f(0.0, 0.0, 1.0),
                vec3f(0.0, 1.0, 0.0),
            );
        }
        case 4u: {
            return mat3x3f(
                vec3f(-1.0, 0.0, 0.0),
                vec3f(0.0, 1.0, 0.0),
                vec3f(0.0, 0.0, 1.0),
            );
        }
        default: {
            return mat3x3f(
                vec3f(1.0, 0.0, 0.0),
                vec3f(0.0, 1.0, 0.0),
                vec3f(0.0, 0.0, -1.0),
            );
        }
    }
}

@group(0) @binding(0)
var skybox: texture_storage_2d_array<rgba16float, write>;

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3u) {
    let face = invocation_id.z;

    let skybox_size = textureDimensions(skybox);
    var uv = (vec2f(invocation_id.xy) + 0.5) / vec2f(skybox_size.xy);

    var light: light::DirectionalLight;
    light.direction = vec4f(0.0, 1.0, 0.0, 1.0);
    light.irradiance = vec4f(1.0);

    var ndc = vec3f(uv * 2.0 - 1.0, -1.0);
    ndc.y *= -1.0;

    var ray: util::Ray;
    ray.direction = normalize(cube_map_face(face) * ndc);
    ray.origin = vec3f(200.0);

    var atmosphere: util::Sphere;
    atmosphere.center = EARTH_CENTER;
    atmosphere.radius = EARTH_RADIUS + ATMOS_HEIGHT;

    let color = integrate_atmospheric_scattering(light, ray, atmosphere);

    textureStore(skybox, invocation_id.xy, face, vec4f(color, 1.0));
}
