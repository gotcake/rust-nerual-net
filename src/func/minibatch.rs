#[derive(Clone, Copy, Debug)]
pub enum MiniBatchSize {
    Full,
    Constant(usize),
    Linear {
        initial: usize,
        slope: f32,
        max: usize,
    },
}

impl MiniBatchSize {
    pub fn get_mini_batch_size(&self, batch_num: usize) -> Option<usize> {
        match self {
            MiniBatchSize::Full => None,
            &MiniBatchSize::Constant(val) => Some(val),
            &MiniBatchSize::Linear { initial, slope, max } => {
                let val = initial + (slope * batch_num as f32) as usize;
                Some(if val > max { max } else { val })
            },
        }
    }
}

