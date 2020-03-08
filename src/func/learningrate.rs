#[derive(Clone, Copy, Debug)]
pub enum LearningRateFn {
    Constant(f32),
    TanhLogarithmicDescent {
        constant_factor: f32,
        log_of_log_base: f32,
        scale: f32,
    }
}

impl LearningRateFn {

    pub fn tanh_logarithmic_descent(log_base: f32, scale: f32) -> Self {
        let log_of_log_base = fast_math::log2(log_base);
        let constant_factor = 1.0 / (1.0 - f32::tanh(fast_math::log2(1.0) / log_of_log_base as f32));
        LearningRateFn::TanhLogarithmicDescent {
            constant_factor,
            log_of_log_base,
            scale
        }
    }

    pub fn standard_tanh_logarithmic_descent() -> Self {
        Self::tanh_logarithmic_descent(100.0, 1.0)
    }

    pub fn get_learning_rate(&self, batch_num: usize) -> f32 {
        match self {
            &LearningRateFn::Constant(val) => val,
            &LearningRateFn::TanhLogarithmicDescent { constant_factor, log_of_log_base, scale } => {
                scale * (1.0 - f32::tanh(fast_math::log2(batch_num as f32 + 1.0) as f32 / log_of_log_base) * constant_factor)
            },
        }
    }
}