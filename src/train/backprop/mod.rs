mod singlethreaded;
mod multithreaded;

use self::multithreaded::*;
use self::singlethreaded::*;
use crate::{
    net::Net,
    data::PreparedDataSet,
    func::{CompletionFn, MiniBatchSize, LearningRateFn, ErrorFn},
    stats::Stats
};

#[derive(Clone, Debug)]
pub struct BackpropOptions {
    pub completion_fn: CompletionFn,
    pub mini_batch_size_fn: MiniBatchSize,
    pub learning_rate_fn: LearningRateFn,
    pub error_fn: ErrorFn,
    pub multi_threading: Option<BackpropMultithreadingOptions>,
}

#[derive(Clone, Debug)]
pub struct BackpropMultithreadingOptions {
    pub worker_threads: Option<usize>,
    pub partitions: usize,
    pub batches_per_sync: usize,
}

pub fn backprop_stage_task_impl(
    net: &mut Net,
    training_set: &PreparedDataSet,
    options: &BackpropOptions,
) -> (Stats, usize) {

    if let Some(ref multi_threading) = options.multi_threading {

        let mut worker_threads = match multi_threading.worker_threads {
            None => num_cpus::get(),
            Some(threads) => threads,
        };
        if worker_threads > multi_threading.partitions {
            worker_threads = multi_threading.partitions;
        }

        train_backprop_multi_threaded(
            net,
            training_set,
            options.completion_fn,
            options.mini_batch_size_fn,
            options.learning_rate_fn,
            options.error_fn,
            multi_threading.batches_per_sync,
            worker_threads,
            multi_threading.partitions,
        )

    } else {

        train_backprop_single_threaded(
            net,
            training_set,
            options.completion_fn,
            options.mini_batch_size_fn,
            options.learning_rate_fn,
            options.error_fn,
        )

    }
}