use std::rc::Rc;
use std::cell::RefCell;
use crate::utils::stable_hash_seed;
use crate::train::task::TaskResult;
use rand::{Rng, FromEntropy, SeedableRng};

pub trait ParamFactory {
    fn range_usize(&mut self, key: String, low: usize, high: usize) -> usize;
    fn range_f32(&mut self, low: f32, high: f32) -> f32;
}

pub trait Optimizer {
    fn next_parameters(&mut self, id: &str) -> Box<dyn ParamFactory>;
    fn report(&mut self, results: &TaskResult);
}

#[derive(Clone)]
pub struct RandomOptimizer {
    rng: Rc<RefCell<rand_xorshift::XorShiftRng>>
}

impl RandomOptimizer {
    pub fn from_entropy() -> Self {
        RandomOptimizer {
            rng: Rc::new(RefCell::new(rand_xorshift::XorShiftRng::from_entropy()))
        }
    }
    #[allow(dead_code)]
    pub fn from_seed(seed: &str) -> Self {
        let seed_bytes = stable_hash_seed(seed);
        RandomOptimizer {
            rng: Rc::new(RefCell::new(rand_xorshift::XorShiftRng::from_seed(seed_bytes)))
        }
    }
}

impl Optimizer for RandomOptimizer {

    fn next_parameters(&mut self, _id: &str) -> Box<dyn ParamFactory> {
        Box::new(RandomParamFactory {
            rng: self.rng.clone(),
        })
    }

    fn report(&mut self, _results: &TaskResult) {
        // no-op
    }
}

struct RandomParamFactory {
    rng: Rc<RefCell<rand_xorshift::XorShiftRng>>
}

#[allow(dead_code)]
impl ParamFactory for RandomParamFactory {

    fn range_usize(&mut self, _key: String, low: usize, high: usize) -> usize {
        return (&*self.rng).borrow_mut().gen_range(low, high);
    }

    fn range_f32(&mut self, low: f32, high: f32) -> f32 {
        return (&*self.rng).borrow_mut().gen_range(low, high);
    }

}