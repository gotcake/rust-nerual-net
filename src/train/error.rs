use crate::{
    data::PreparedDataSet,
    net::Net,
    train::buffers::TrainingBuffers,
    layer::NetLayerBase,
    func::ErrorFn
};

pub fn forward_pass_and_compute_error(
    net: &Net,
    inputs: &[f32],
    expected_outputs: &[f32],
    error_fn: &ErrorFn,
    buffers: &mut TrainingBuffers,
) {

    debug_assert_eq!(net.first_layer().input_size(), inputs.len());
    debug_assert_eq!(net.last_layer().output_size(), expected_outputs.len());

    // forward pass
    {
        let layer_output = buffers.output_buffers.get_first_row_mut();
        net.first_layer().forward_pass(inputs, layer_output);
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
            error_sum += error_fn.get_error(expected_outputs[output_index], output[output_index]);
            last_error_grad_buffer[output_index] = error_fn.get_error_derivative(expected_outputs[output_index], output[output_index]);
        }
        buffers.error_stats.report(error_sum);
    }
}

pub fn compute_error_for_batch(
    net: &mut Net,
    data: &PreparedDataSet,
    error_fn: &ErrorFn,
    buffers: &mut TrainingBuffers,
) {
    buffers.error_stats.reset();
    for (inputs, expected_outputs) in data {
        forward_pass_and_compute_error(
            net,
            inputs,
            expected_outputs,
            error_fn,
            buffers
        );
    }
}