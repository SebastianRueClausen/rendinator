#import consts

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
}

@group(0) @binding(0)
var display: texture_2d<f32>;

@group(1) @binding(1)
var<storage, read_write> average_luminance: f32;

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let vertex_index = 2u - in.vertex_index;

    out.uv = vec2f(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u),
    );

    out.clip_position = vec4f(out.uv * 2.0 - 1.0, 0.0, 1.0);
    out.clip_position.y *= -1.0;

    return out;
}

// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
fn rgb_to_xyz(rgb: vec3f) -> vec3f {
	var xyz: vec3f;
	xyz.x = dot(vec3f(0.4124564, 0.3575761, 0.1804375), rgb);
	xyz.y = dot(vec3f(0.2126729, 0.7151522, 0.0721750), rgb);
	xyz.z = dot(vec3f(0.0193339, 0.1191920, 0.9503041), rgb);
	return xyz;
}

// http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_xyY.html
fn xyz_to_yxy(xyz: vec3f) -> vec3f {
	let inverse = 1.0 / dot(xyz, vec3f(1.0));
	return vec3f(xyz.y, xyz.x * inverse, xyz.y * inverse);
}

// http://www.brucelindbloom.com/index.html?Eqn_xyY_to_XYZ.html
fn yxy_to_xyz(yxy: vec3f) -> vec3f {
	var xyz: vec3f;
	xyz.x = yxy.x * yxy.y / yxy.z;
	xyz.y = yxy.x;
	xyz.z = yxy.x * (1.0 - yxy.y - yxy.z) / yxy.z;
	return xyz;
}

fn xyz_to_rgb(xyz: vec3f) -> vec3f {
    var rgb: vec3f;
	rgb.x = dot(vec3f(3.2404542, -1.5371385, -0.4985314), xyz);
	rgb.y = dot(vec3f(-0.9692660, 1.8760108, 0.0415560), xyz);
	rgb.z = dot(vec3f(0.0556434, -0.2040259, 1.0572252), xyz);
	return rgb;
}

fn aces_tonemap(x: f32) -> f32 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4f {
    let texel_id = vec2i(in.uv * vec2f(textureDimensions(display, 0)));
    var rgb = textureLoad(display, texel_id, 0).rgb;

    var yxy = xyz_to_yxy(rgb_to_xyz(rgb));
    yxy.x /= (9.6 * average_luminance + 0.0001);
    rgb = xyz_to_rgb(yxy_to_xyz(yxy));

    rgb.r = aces_tonemap(rgb.r);
    rgb.g = aces_tonemap(rgb.g);
    rgb.b = aces_tonemap(rgb.b);

    return vec4f(rgb, 1.0);
}
