use std::time::SystemTime;

use crate::{
    train::{
        buffers::TrainingBuffers,
        error::{
            forward_pass_and_compute_error,
            compute_error_for_batch
        },
    },
    data::{
        PreparedDataSet,
        PreparedDataSetIterator
    },
    net::Net,
    layer::NetLayerBase,
    func::{
        CompletionFn,
        MiniBatchSize,
        LearningRateFn
    },
    func::ErrorFn,
    stats::Stats
};

pub fn train_backprop_single_threaded(
    net: &mut Net,
    training_set: &PreparedDataSet,
    completion_fn: CompletionFn,
    mini_batch_size_fn: MiniBatchSize,
    learning_rate_fn: LearningRateFn,
    error_fn: ErrorFn,
) -> (Stats, usize) {

    let stage_start_time = SystemTime::now();
    let mut buffers = TrainingBuffers::for_net(net);

    let mut batch_num = 0;

    loop {

        let learning_rate = learning_rate_fn.get_learning_rate(batch_num);

        train_backprop_single_batch(
            net,
            &mut training_set.iter(),
            &mut buffers,
            mini_batch_size_fn.get_mini_batch_size(batch_num),
            learning_rate,
            &error_fn,
        );

        compute_error_for_batch(
            net,
            &training_set,
            &error_fn,
            &mut buffers
        );

        batch_num += 1;

        if completion_fn.should_stop_training(batch_num, stage_start_time, &buffers.error_stats) {
            break;
        }

    }

    (buffers.error_stats, batch_num)

}

pub fn train_backprop_single_batch(
    net: &mut Net,
    iter: &mut PreparedDataSetIterator,
    buffers: &mut TrainingBuffers,
    mini_batch_size: Option<usize>,
    learning_rate: f32,
    error_fn: &ErrorFn,
) {

    debug_assert!(iter.has_next());
    debug_assert!(learning_rate > 0.0 && learning_rate <= 10.0);

    while iter.has_next() {

        //buffers.error_gradient_buffers.reset_to(0.0);

        buffers.weight_deltas.reset_to(0.0);

        let mut remaining_epochs = match mini_batch_size {
            None => -1,
            Some(size) => size as i64,
        };

        while remaining_epochs != 0 && iter.has_next() {

            let (inputs, expected_outputs) = iter.next_unchecked();

            forward_pass_and_compute_error(
                net,
                inputs,
                expected_outputs,
                error_fn,
                buffers,
            );

            // back-propagate errors without updating the net
            for layer_index in (1..net.num_layers()).rev() {
                let (input_errors, output_errors) = buffers.error_gradient_buffers.split_rows(layer_index - 1, layer_index);
                net.layer(layer_index).backprop(
                    output_errors,
                    buffers.output_buffers.get_row(layer_index - 1),
                    buffers.output_buffers.get_row(layer_index),
                    learning_rate,
                    input_errors,
                    buffers.weight_deltas.get_row_mut(layer_index),
                );
            }

            net.first_layer().backprop(
                buffers.error_gradient_buffers.get_first_row(),
                inputs,
                buffers.output_buffers.get_first_row(),
                learning_rate,
                buffers.input_error_buffer.as_mut_slice(),
                buffers.weight_deltas.get_first_row_mut(),
            );

            if remaining_epochs > 0 {
                remaining_epochs -= 1;
            }
        }

        // apply weight updates
        net.add_weights_from(&buffers.weight_deltas);

    }

}