use std::u128;

#[inline(always)]
pub fn square_f32(n: f32) -> f32 {
    n * n
}


const SEED_XOR: u128 = 0b10101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010;
pub fn stable_hash_seed(s: &str) -> [u8; 16] {
    let mut val = 17u128;
    for byte in s.as_bytes() {
        val = 31u128.wrapping_mul(val).wrapping_add(*byte as u128);
        val ^= SEED_XOR;
    }
    unsafe {
        std::mem::transmute::<u128, [u8; 16]>(val)
    }
}