use crate::utils::square_f32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActivationFn {
    LogisticSigmoid {
        steepness: f32,
        scale: f32,
        y_offset: f32
    }
}

#[allow(dead_code)]
impl ActivationFn {

    pub fn standard_logistic_sigmoid() -> Self {
        ActivationFn::LogisticSigmoid {
            steepness: 1.0,
            scale: 1.0,
            y_offset: 0.0
        }
    }

    pub fn standard_logistic_sigmoid_neg() -> Self {
        ActivationFn::LogisticSigmoid {
            steepness: 1.0,
            scale: 2.0,
            y_offset: -1.0
        }
    }

    pub fn get_activation(&self, n: f32) -> f32 {
        match self {
            &ActivationFn::LogisticSigmoid { steepness, scale, y_offset } => {
                scale / (1.0 + f32::exp(-steepness * n)) + y_offset
            },
        }
    }

    pub fn get_activation_derivative(&self, n: f32) -> f32 {
        match self {
            &ActivationFn::LogisticSigmoid { steepness, scale, y_offset: _ } => {
                let z = f32::exp(-steepness * n);
                scale * steepness * z / square_f32(z + 1.0)
            },
        }
    }

}