use std::num::NonZeroU32;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum MiniBatchSize {
    Full,
    Constant(NonZeroU32),
    Linear {
        initial: u32,
        slope: f32,
        max: u32,
    },
}

impl MiniBatchSize {
    pub fn get_mini_batch_size(&self, batch_num: usize) -> Option<NonZeroU32> {
        match self {
            MiniBatchSize::Full => None,
            &MiniBatchSize::Constant(val) => Some(val),
            &MiniBatchSize::Linear { initial, slope, max } => {
                let val = initial + (slope * batch_num as f32) as u32;
                NonZeroU32::new(if val > max { max } else { val })
            },
        }
    }
}

