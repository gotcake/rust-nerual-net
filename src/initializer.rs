use rand::{SeedableRng, FromEntropy};
use rand::Rng;

pub trait NetInitializer {
    fn get_weight(&mut self, layer_index: usize, sub_index: usize) -> f32;
    fn get_bias(&mut self, layer_index: usize, sub_index: usize) -> f32;
}

pub struct RandomNetInitializer {
    weight_std_dev: f32,
    bias_std_dev: f32,
    rng: rand_xorshift::XorShiftRng
}

#[allow(dead_code)]
impl RandomNetInitializer {

    pub fn new_standard_from_entropy() -> Self {
        RandomNetInitializer {
            weight_std_dev: 0.01,
            bias_std_dev: 0.01,
            rng: rand_xorshift::XorShiftRng::from_entropy()
        }
    }

    pub fn new_standard_with_seed(seed: [u32; 4]) -> Self {
        let seed_bytes = unsafe {
            std::mem::transmute::<[u32; 4], [u8; 16]>(seed)
        };
        RandomNetInitializer {
            weight_std_dev: 0.01,
            bias_std_dev: 0.01,
            rng: rand_xorshift::XorShiftRng::from_seed(seed_bytes)
        }
    }

}

impl NetInitializer for RandomNetInitializer {
    fn get_weight(&mut self, _layer_index: usize, _sub_index: usize) -> f32 {
        self.rng.gen::<f32>() * self.weight_std_dev
    }
    fn get_bias(&mut self, _layer_index: usize, _sub_index: usize) -> f32 {
        self.rng.gen::<f32>() * self.bias_std_dev
    }
}