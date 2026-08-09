#[no_mangle]
pub extern "C" fn v86_libm_sin_f32(bits: u32) -> u32 {
    (libm::sin(f32::from_bits(bits) as f64) as f32).to_bits()
}

#[no_mangle]
pub extern "C" fn v86_libm_cos_f32(bits: u32) -> u32 {
    (libm::cos(f32::from_bits(bits) as f64) as f32).to_bits()
}
