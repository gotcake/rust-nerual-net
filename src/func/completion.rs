use crate::stats::Stats;
use std::time::{SystemTime, Duration};

#[derive(Clone, Copy, Debug)]
pub struct CompletionFn {
    max_epoch: Option<usize>,
    max_duration: Option<Duration>,
    target_avg_error: f32,
}

impl CompletionFn {

    pub fn stop_after_epoch(epoch: usize) -> Self {
        CompletionFn {
            max_epoch: Some(epoch),
            max_duration: None,
            target_avg_error: 0.0
        }
    }

    pub fn stop_after_duration(duration: Duration) -> Self {
        CompletionFn {
            max_epoch: None,
            max_duration: Some(duration),
            target_avg_error: 0.0
        }
    }

    pub fn should_stop_training(&self, epoch: usize, start_time: SystemTime, error_stats: &Stats) -> bool {
        if self.target_avg_error >= error_stats.mean() {
            return true;
        }
        if let Some(max_batch_count) = self.max_epoch {
            if max_batch_count - 1 <= epoch {
                return true;
            }
        }
        if let Some(max_duration) = self.max_duration {
            if max_duration <= SystemTime::now().duration_since(start_time).unwrap_or(max_duration) {
                return true
            }
        }
        false
    }

}