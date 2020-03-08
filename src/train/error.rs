use crate::{
    data::{TrainingSetIterator, TrainingSet},
    net::Net,
    train::{
        buffers::TrainingBuffers,
        TrainingContext
    },
    layer::NetLayerBase
};
use crate::func::ErrorFn;

pub fn forward_pass_and_compute_error(
    net: &Net,
    iter: &TrainingSetIterator,
    context: &TrainingContext,
    error_fn: &ErrorFn,
    buffers: &mut TrainingBuffers,
) {

    iter.get_columns(&context.dependent_columns, buffers.input_buffer.as_mut_slice());
    iter.get_columns(&context.independent_columns, buffers.expected_output_buffer.as_mut_slice());

    // forward pass
    {
        let layer_output = buffers.output_buffers.get_first_row_mut();
        net.first_layer().forward_pass(buffers.input_buffer.as_slice(), layer_output);
    }

    for layer_index in 1..net.num_layers() {
        let (layer_input, layer_output) = buffers.output_buffers.split_rows(layer_index - 1, layer_index);
        net.layer(layer_index).forward_pass(layer_input, layer_output);
    }

    // compute error
    {
        let mut error_sum = 0.0;
        let last_error_grad_buffer = buffers.error_gradient_buffers.get_last_row_mut();
        let output = buffers.output_buffers.get_last_row();
        for output_index in 0..net.output_size() {
            error_sum += error_fn.get_error(buffers.expected_output_buffer[output_index], output[output_index]);
            last_error_grad_buffer[output_index] = error_fn.get_error_derivative(buffers.expected_output_buffer[output_index], output[output_index]);
        }
        buffers.error_stats.report(error_sum);
    }
}

pub fn compute_error_for_batch(
    net: &mut Net,
    data: &TrainingSet,
    context: &TrainingContext,
    error_fn: &ErrorFn,
    buffers: &mut TrainingBuffers,
) {
    buffers.error_stats.reset();
    let mut iter = data.iter();
    while iter.next() {
        forward_pass_and_compute_error(
            net,
            &iter,
            context,
            error_fn,
            buffers
        );
    }
}