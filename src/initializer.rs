use rand::{SeedableRng, FromEntropy};
use rand::Rng;
use crate::utils::stable_hash_seed;
use rand::distributions::StandardNormal;

#[derive(Clone)]
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

    pub fn new_standard_with_seed(val: &str) -> Self{
        let seed_bytes = stable_hash_seed(val);
        RandomNetInitializer {
            weight_std_dev: 0.01,
            bias_std_dev: 0.01,
            rng: rand_xorshift::XorShiftRng::from_seed(seed_bytes)
        }
    }

    pub fn get_weight(&mut self) -> f32 {
        self.rng.sample(StandardNormal) as f32 * self.weight_std_dev
    }

    pub fn get_bias(&mut self) -> f32 {
        self.rng.sample(StandardNormal) as f32 * self.bias_std_dev
    }

}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_from_seed() {
        let mut init = RandomNetInitializer::new_standard_with_seed("a random string");
        assert!((init.get_weight() - 0.0052833073).abs() < 0.0001);
        assert!((init.get_bias() - 0.0018487974).abs() < 0.0001);
        assert!((init.get_bias() - 0.0068561565).abs() < 0.0001);
        assert!((init.get_bias() - -0.005462957).abs() < 0.0001);
    }

}