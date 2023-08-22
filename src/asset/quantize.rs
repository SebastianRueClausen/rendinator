#[inline]
pub fn quantize_unorm<const BITS: u32>(value: f32) -> u32 {
    let scale = ((1_i32 << BITS) - 1) as f32;
    let value = value.clamp(0.0, 1.0);

    (value * scale + 0.5) as u32
}

#[inline]
pub fn dequantize_unorm<const BITS: u32>(value: u32) -> f32 {
    let scale = ((1_i32 << BITS) - 1) as f32;
    value as f32 / scale
}

#[inline]
pub fn quantize_snorm<const BITS: u32>(value: f32) -> i32 {
    let scale = ((1 << (BITS - 1)) - 1) as f32;

    let round: f32 = if value >= 0.0 { 0.5 } else { -0.5 };
    let value: f32 = if value >= -1f32 { value } else { -1f32 };
    let value: f32 = if value <= 1f32 { value } else { 1f32 };

    (value * scale + round) as i32
}

#[cfg(test)]
#[inline]
pub fn dequantize_snorm<const BITS: u32>(value: i32) -> f32 {
    let scale = (1_i32 << (BITS - 1)) - 1;

    value as f32 / scale as f32
}

#[test]
fn snorm() {
    let value = std::f32::consts::PI - 3.0;
    let dequantized = dequantize_snorm::<31>(quantize_snorm::<31>(value));

    assert!(
        (value - dequantized).abs() < 0.001,
        "{value} -> {dequantized}"
    );
}
