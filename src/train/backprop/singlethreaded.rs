use std::time::SystemTime;

use crate::train::context::NetTrainingContext;
use crate::data::PreparedDataSet;
use crate::data::PreparedDataSetIterator;
use crate::net::Net;
use crate::layer::NetLayerBase;
use crate::func::CompletionFn;
use crate::func::MiniBatchSize;
use crate::func::LearningRateFn;
use crate::func::ErrorFn;
use crate::stats::Stats;

pub fn train_backprop_single_threaded(
    net: &mut Net,
    data_set: &PreparedDataSet,
    completion_fn: CompletionFn,
    mini_batch_size_fn: MiniBatchSize,
    learning_rate_fn: LearningRateFn,
    error_fn: ErrorFn,
) -> (Stats, usize) {

    let stage_start_time = SystemTime::now();
    let mut context: NetTrainingContext = net.get_training_context();

    let mut batch_num = 0;

    loop {

        context.train_backprop_single_batch(
            data_set,
            learning_rate_fn.get_learning_rate(batch_num),
            &error_fn,
            mini_batch_size_fn.get_mini_batch_size(batch_num),
        );

        let error_stats = context.compute_error_for_batch(
            data_set,
            &error_fn,
        );

        batch_num += 1;

        if completion_fn.should_stop_training(batch_num, stage_start_time, &error_stats) {
            return (error_stats, batch_num)
        }

    }

    unreachable!();

}