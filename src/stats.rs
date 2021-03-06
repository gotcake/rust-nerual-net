use std::{f32, fmt};
use crate::utils::square_f32;

#[derive(Clone)]
pub struct Stats {
    sum: f64,
    count: u32,
    max: f32,
    min: f32,
    // variables for variance computation
    // see https://stackoverflow.com/a/897463
    var_m: f64,
    var_s: f64,
}

impl fmt::Debug for Stats {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), std::fmt::Error> {
        f.debug_struct("Stats")
            .field("count", &self.count)
            .field("min", &self.min)
            .field("max", &self.max)
            .field("sum", &self.sum)
            .field("mean", &self.mean())
            .field("std_dev", &self.std_dev())
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
            min: f32::NAN,
            var_m: 0.0,
            var_s: 0.0,
        }
    }

    pub fn report(&mut self, value: f32) {
        let val64 = value as f64;
        self.sum += val64;
        self.count += 1;
        if self.min.is_nan() || self.min > value {
            self.min = value;
        }
        if self.max.is_nan() || self.max < value {
            self.max = value;
        }
        // variance computation
        // see https://stackoverflow.com/a/897463
        let val_minus_m = val64 - self.var_m;
        self.var_m += val_minus_m / self.count as f64;
        self.var_s += val_minus_m * (val64 - self.var_m);

    }

    #[inline]
    pub fn mean(&self) -> f64 {
        self.sum / self.count as f64
    }

    #[inline]
    pub fn variance(&self) -> f64 {
        if self.count > 1 {
            self.var_s / self.count as f64
        } else {
            0.0
        }
    }

    #[inline]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
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
    pub fn sum(&self) -> f64 {
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
        self.var_m = 0.0;
        self.var_s = 0.0;
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
        for (col_idx, _, matrix) in &self.matrices {
            if column_index == *col_idx {
                return Some(matrix.clone())
            }
        }
        None
    }

    pub fn get_for_column_name(&self, column_name: &str) -> Option<ConfusionMatrix> {
        for (_, ref name, matrix) in &self.matrices {
            if let Some(name) = name {
                if name.as_str() == column_name {
                    return Some(matrix.clone());
                }
            }
        }
        None
    }

}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_stats() {

        let mut s = Stats::new();

        assert!(s.min().is_nan());
        assert!(s.max().is_nan());
        assert_eq!(s.sum(), 0.0);
        assert_eq!(s.count(), 0);
        assert!(s.mean().is_nan());

        s.report(1.0);
        s.report(1.0);
        s.report(2.5);
        s.report(10.0);
        s.report(-2.0);

        assert_eq!(s.min(), -2.0);
        assert_eq!(s.max(), 10.0);
        assert_eq!(s.sum(), 12.5);
        assert_eq!(s.count(), 5);
        assert_eq!(s.mean(), 2.5);
        assert!((s.std_dev() - 4.0249223594996).abs() < 0.001);

    }

    #[test]
    fn test_confusion_matrix() {

        let mut m = ConfusionMatrix::new();
        m.record(true, false);
        m.record(true, true);
        m.record(true, false);
        m.record(false, false);
        m.record(false, false);
        m.record(false, true);
        assert_eq!(m.to_string(), "[t+ = 0.16666667, t- = 0.33333334, f+ = 0.33333334, f- = 0.16666667]".to_string());


    }

}