use crate::utils::square_f32;

#[derive(Clone, Copy, Debug)]
pub enum ErrorFn {
    SquaredError,
    // TOOD: cross-entropy loss?
}

impl ErrorFn {
    pub fn get_error(&self, expected: f32, actual: f32) -> f32 {
        match self {
            ErrorFn::SquaredError => 0.5 * square_f32(expected - actual),
        }
    }
    pub fn get_error_derivative(&self, expected: f32, actual: f32) -> f32 {
        match self {
            ErrorFn::SquaredError => actual - expected,
        }
    }
}