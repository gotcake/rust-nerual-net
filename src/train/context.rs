use std::num::NonZeroU32;

use crate::net::Net;
use crate::buffer::RowBuffer;
use crate::stats::Stats;
use crate::layer::{NetLayer, NetLayerBase};
use crate::func::ErrorFn;
use crate::data::PreparedDataSet;

pub struct NetTrainingContext<'a> {
    net: &'a mut Net,
    output_buffers: RowBuffer,
    error_gradient_buffers: RowBuffer,
    input_error_buffer: Vec<f32>,
    error_stats: Stats,
    weight_deltas: RowBuffer,
}

impl<'a> NetTrainingContext<'a> {

    pub(crate) fn new(net: &'a mut Net) -> Self {
        let layer_sizes: Vec<usize> = net.layer_iter()
            .map(NetLayer::output_size)
            .collect();
        let input_size = net.input_size();
        let weight_deltas = net.new_zeroed_weight_buffer();
        NetTrainingContext {
            net,
            output_buffers: RowBuffer::new_with_row_sizes(0.0, &layer_sizes),
            error_gradient_buffers: RowBuffer::new_with_row_sizes(0.0, &layer_sizes),
            input_error_buffer: vec![0f32; input_size],
            error_stats: Stats::new(),
            weight_deltas,
        }
    }

    fn forward_pass_and_compute_error(
        &mut self,
        inputs: &[f32],
        expected_outputs: &[f32],
        error_fn: &ErrorFn,
    ) {

        debug_assert_eq!(self.net.first_layer().input_size(), inputs.len());
        debug_assert_eq!(self.net.last_layer().output_size(), expected_outputs.len());

        // forward pass
        {
            let layer_output = self.output_buffers.get_first_row_mut();
            self.net.first_layer().forward_pass(
                self.net.get_weights().get_first_row(),
                inputs,
                layer_output
            );
        }

        for layer_index in 1..self.net.num_layers() {
            let (layer_input, layer_output) = self.output_buffers.split_rows(layer_index - 1, layer_index);
            self.net.layer(layer_index).forward_pass(
                self.net.get_weights().get_row(layer_index),
                layer_input,
                layer_output
            );
        }

        // compute error
        {
            let mut error_sum = 0.0;
            let last_error_grad_buffer = self.error_gradient_buffers.get_last_row_mut();
            let output = self.output_buffers.get_last_row();
            for output_index in 0..self.net.output_size() {
                error_sum += error_fn.get_error(expected_outputs[output_index], output[output_index]);
                last_error_grad_buffer[output_index] = error_fn.get_error_derivative(expected_outputs[output_index], output[output_index]);
            }
            self.error_stats.report(error_sum);
        }
    }

    fn backprop(
        &mut self,
        inputs: &[f32],
        learning_rate: f32,
    ) {

        debug_assert_eq!(inputs.len(), self.net.input_size());
        debug_assert!(learning_rate > 0.0 && learning_rate <= 10.0);

        // back-propagate errors without updating the net
        for layer_index in (1..self.net.num_layers()).rev() {
            let (input_errors, output_errors) = self.error_gradient_buffers.split_rows(layer_index - 1, layer_index);
            self.net.layer(layer_index).backprop(
                self.net.get_weights().get_row(layer_index),
                output_errors,
                self.output_buffers.get_row(layer_index - 1),
                self.output_buffers.get_row(layer_index),
                learning_rate,
                input_errors,
                self.weight_deltas.get_row_mut(layer_index),
            );
        }

        self.net.first_layer().backprop(
            self.net.get_weights().get_first_row(),
            self.error_gradient_buffers.get_first_row(),
            inputs,
            self.output_buffers.get_first_row(),
            learning_rate,
            self.input_error_buffer.as_mut_slice(),
            self.weight_deltas.get_first_row_mut(),
        );
    }

    pub fn train_backprop_single_batch(
        &mut self,
        data_set: &PreparedDataSet,
        learning_rate: f32,
        error_fn: &ErrorFn,
        mini_batch_size: Option<NonZeroU32>
    ) {

        let mut iter = data_set.iter();

        debug_assert!(iter.has_next());

        while iter.has_next() {

            //buffers.error_gradient_buffers.reset_to(0.0);

            self.weight_deltas.reset_to(0.0);

            let mut remaining_epochs = match mini_batch_size {
                None => -1,
                Some(size) => size.get() as i64,
            };

            while remaining_epochs != 0 && iter.has_next() {

                let (inputs, expected_outputs) = iter.next_unchecked();

                self.forward_pass_and_compute_error(
                    inputs,
                    expected_outputs,
                    error_fn,
                );

                self.backprop(inputs, learning_rate);

                if remaining_epochs > 0 {
                    remaining_epochs -= 1;
                }
            }

            // apply weight updates
            self.net.get_weights_mut().add(&self.weight_deltas);

        }
    }

    pub fn compute_error_for_batch(&mut self, data_set: &PreparedDataSet, error_fn: &ErrorFn) -> Stats {
        self.error_stats.reset();
        for (inputs, expected_outputs) in data_set {
            self.forward_pass_and_compute_error(inputs, expected_outputs, error_fn);
        }
        self.error_stats.clone()
    }

    #[inline]
    pub fn get_net(&mut self) -> &Net {
        &self.net
    }

    #[inline]
    pub fn get_net_mut(&mut self) -> &mut Net {
        &mut self.net
    }

}