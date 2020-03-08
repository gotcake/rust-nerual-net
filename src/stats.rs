use std::{f32, fmt};
use std::cmp::Ordering;

#[derive(Clone)]
pub struct Stats {
    sum: f32,
    count: u32,
    max: f32,
    min: f32
}

impl fmt::Debug for Stats {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), std::fmt::Error> {
        f.debug_struct("Stats")
            .field("count", &self.count)
            .field("min", &self.min)
            .field("max", &self.max)
            .field("sum", &self.sum)
            .field("mean", &self.mean())
            .finish()
    }
}

#[allow(dead_code)]
impl Stats {

    pub fn new() -> Self {
        Stats {
            sum: 0.0,
            count: 0,
            max: f32::NAN,
            min: f32::NAN
        }
    }

    #[inline]
    pub fn report(&mut self, value: f32) {
        self.sum += value;
        self.count += 1;
        if self.min.is_nan() || self.min > value {
            self.min = value;
        }
        if self.max.is_nan() || self.max < value {
            self.max = value;
        }
    }

    #[inline]
    pub fn mean(&self) -> f32 {
        self.sum / self.count as f32
    }

    #[inline]
    pub fn max(&self) -> f32 {
        self.max
    }

    #[inline]
    pub fn min(&self) -> f32 {
        self.min
    }

    #[inline]
    pub fn sum(&self) -> f32 {
        self.sum
    }

    #[inline]
    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
        self.max = f32::NAN;
        self.min = f32::NAN;
    }
}


#[derive(Clone)]
pub struct ConfusionMatrix {
    count: u32,
    true_positive: u32,
    true_negative: u32,
    false_positive: u32,
    false_negative: u32,
}

#[allow(dead_code)]
impl ConfusionMatrix {

    pub fn new() -> Self {
        ConfusionMatrix {
            count: 0,
            true_positive: 0,
            true_negative: 0,
            false_positive: 0,
            false_negative: 0
        }
    }

    #[inline]
    pub fn record(&mut self, estimated: bool, actual: bool) {
        if estimated {
            if actual {
                self.true_positive += 1;
            } else {
                self.false_positive += 1;
            }
        } else {
            if actual {
                self.false_negative += 1;
            } else {
                self.true_negative += 1;
            }
        }
        self.count += 1;
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.true_positive = 0;
        self.true_negative = 0;
        self.false_positive = 0;
        self.false_negative = 0;
    }

    pub fn true_positive_rate(&self) -> f32 {
        return self.true_positive as f32 / self.count as f32
    }

    pub fn false_positive_rate(&self) -> f32 {
        return self.false_positive as f32 / self.count as f32
    }

    pub fn true_negative_rate(&self) -> f32 {
        return self.true_negative as f32 / self.count as f32
    }

    pub fn false_negative_rate(&self) -> f32 {
        return self.false_negative as f32 / self.count as f32
    }

    pub fn error_rate(&self) -> f32 {
        return (self.false_negative as f32 + self.false_positive as f32) / self.count as f32
    }

}

impl ToString for ConfusionMatrix {
    fn to_string(&self) -> String {
        format!("[t+ = {}, t- = {}, f+ = {}, f- = {}]",
            self.true_positive_rate(),
            self.true_negative_rate(),
            self.false_positive_rate(),
            self.false_negative_rate()
        )
    }
}

#[allow(dead_code)]
pub struct ConfusionMatrices {
    matrices: Vec<(usize, Option<String>, ConfusionMatrix)>
}

#[allow(dead_code)]
impl ConfusionMatrices {

    #[inline]
    pub fn record_for_output_index(&mut self, output_index: usize, estimated: bool, actual: bool) {
        self.matrices[output_index].2.record(estimated, actual);
    }

    pub fn get_for_column_index(&self, column_index: usize) -> Option<ConfusionMatrix> {
        let mut iter = self.matrices.iter();
        while let Some((col_idx, _, matrix)) = iter.next() {
            if *col_idx == column_index {
                return Some(matrix.clone());
            }
        }
        None
    }

    pub fn get_for_column_name(&self, column_name: &str) -> Option<ConfusionMatrix> {
        let mut iter = self.matrices.iter();
        while let Some((_, name, matrix)) = iter.next() {
            if name.is_some() && name.as_ref().map(String::as_str) == Some(column_name) {
                return Some(matrix.clone());
            }
        }
        None
    }

}