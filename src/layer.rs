use crate::initializer::RandomNetInitializer;
use crate::func::ActivationFn;
use std::fmt;
use crate::utils::{split_slice_mut, split_slice};

pub trait NetLayerBase {
    fn forward_pass(&self, weight_buffer: &[f32], input: &[f32], output: &mut[f32]);
    fn backprop(&self, weight_buffer: &[f32], output_errors: &[f32], inputs: &[f32], outputs: &[f32],
                learning_rate: f32, input_errors: &mut[f32], delta_target: &mut [f32]);
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn weight_buffer_size(&self) -> usize;
    fn initialize_weights(&self, weight_buffer: &mut [f32], initializer: &mut RandomNetInitializer);
    fn get_config(&self) -> NetLayerConfig;
}

#[derive(Clone, Debug, PartialEq)]
pub enum NetLayerConfig {
    FullyConnected(usize, ActivationFn)
}

impl NetLayerConfig {
    pub fn create_layer(
        &self,
        input_size: usize,
    ) -> NetLayer {
        match self {
            &NetLayerConfig::FullyConnected(size, activation_fn) => {
                NetLayer::FullyConnected(
                    FullyConnectedNetLayer::new(
                        input_size,
                        size,
                        activation_fn,
                    )
                )
            },
        }
    }
}

#[derive(Clone, Debug)]
pub enum NetLayer {
    FullyConnected(FullyConnectedNetLayer)
}

impl NetLayer {

    #[inline]
    fn get_delegate(&self) -> &dyn NetLayerBase {
        match self {
            NetLayer::FullyConnected(layer) => layer,
        }
    }

}

impl NetLayerBase for NetLayer {

    // NOTE: not using delegate functions for most frequently called methods to avoid dynamic dispatch

    fn forward_pass(&self, weight_buffer: &[f32], input: &[f32], output: &mut [f32]) {
        match self {
            NetLayer::FullyConnected(layer) => layer.forward_pass(weight_buffer, input, output),
        }
    }

    fn backprop(&self, weight_buffer: &[f32], output_errors: &[f32], inputs: &[f32], outputs: &[f32], learning_rate: f32, input_errors: &mut [f32], delta_target: &mut [f32]) {
        match self {
            NetLayer::FullyConnected(layer) => layer.backprop(weight_buffer, output_errors, inputs, outputs, learning_rate, input_errors, delta_target),
        }
    }

    fn input_size(&self) -> usize {
        self.get_delegate().input_size()
    }

    fn output_size(&self) -> usize {
        self.get_delegate().output_size()
    }

    fn weight_buffer_size(&self) -> usize {
        self.get_delegate().weight_buffer_size()
    }

    fn initialize_weights(&self, weight_buffer: &mut [f32], initializer: &mut RandomNetInitializer) {
        self.get_delegate().initialize_weights(weight_buffer, initializer);
    }

    fn get_config(&self) -> NetLayerConfig {
        self.get_delegate().get_config()
    }
}

#[derive(Clone)]
pub struct FullyConnectedNetLayer {
    input_size: usize,
    size: usize,
    num_weights: usize,
    activation_fn: ActivationFn,
}

impl fmt::Debug for FullyConnectedNetLayer {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        f.debug_struct("FullyConnectedNetLayer")
            .field("input_size", &self.size)
            .field("size", &self.size)
            .field("num_weights", &self.num_weights)
            .field("num_biases", &self.size)
            .field("activation_fn", &self.activation_fn)
            .finish()
    }
}

impl FullyConnectedNetLayer {

    pub fn new(
        input_size: usize,
        size: usize,
        activation_fn: ActivationFn,
    ) -> Self {
        FullyConnectedNetLayer {
            input_size,
            size,
            num_weights: size * input_size,
            activation_fn,
        }
    }

    #[inline(always)]
    fn get_weight(&self, weights: &[f32], input_index: usize, node_index: usize) -> f32 {
        weights[input_index * self.size + node_index]
    }

}

impl NetLayerBase for FullyConnectedNetLayer {

    fn forward_pass(&self, weight_buffer: &[f32], input: &[f32], output: &mut[f32]) {

        debug_assert_eq!(input.len(), self.input_size);

        let (weights, biases) = split_slice(weight_buffer, self.num_weights, self.size);

        for node_index in 0..self.size {
            let mut sum = biases[node_index];
            for input_index in 0..self.input_size {
                sum = sum + input[input_index] * self.get_weight(weights, input_index, node_index);
            }
            output[node_index] = self.activation_fn.get_activation(sum);
        }
    }

    fn backprop(&self, weight_buffer: &[f32], output_errors: &[f32], inputs: &[f32], outputs: &[f32],
                learning_rate: f32, input_errors: &mut [f32], delta_target: &mut [f32]) {

        debug_assert_eq!(output_errors.len(), self.size);
        debug_assert_eq!(input_errors.len(), self.input_size);
        debug_assert_eq!(inputs.len(), self.input_size);
        debug_assert_eq!(outputs.len(), self.size);
        debug_assert_eq!(delta_target.len(), self.num_weights + self.size);

        let (weights, biases) = split_slice(weight_buffer, self.num_weights, self.size);
        let (weight_deltas, bias_deltas) = split_slice_mut(delta_target, self.num_weights, self.size);

        for error in input_errors.as_mut() {
            *error = 0.0;
        }
        for node_index in 0..self.size {
            let node_error = output_errors[node_index];
            // gradient describes the rate of change of the activation function at the output value,
            // reflecting how much change in the output we would see for a given change in the input
            let node_gradient = self.activation_fn.get_activation_derivative(outputs[node_index]);
            let node_error_gradient = node_gradient * node_error;
            // compute the error for each connection and update the weight
            for input_index in 0..self.input_size {
                let input = inputs[input_index];
                let connection_weight = self.get_weight(weights, input_index, node_index);
                let weight_delta = -learning_rate * node_error_gradient * input;
                //let new_weight = connection_weight + weight_delta;
                // TODO
                weight_deltas[input_index * self.size + node_index] += weight_delta;
                // self.set_weight(input_index, node_index, new_weight);
                input_errors[input_index] += connection_weight * node_error_gradient;
            }
            // update bias
            // TODO, not sure if this should be multiplied by bias value such as weight_delta
            bias_deltas[node_index] -= learning_rate * node_error_gradient;
            // TODO self.biases[node_index] -= learning_rate * node_error_gradient;
        }
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn output_size(&self) -> usize {
        self.size
    }

    fn weight_buffer_size(&self) -> usize {
        self.num_weights + self.size
    }


    fn initialize_weights(&self, weight_buffer: &mut [f32], initializer: &mut RandomNetInitializer) {
        let (weights, biases) = split_slice_mut(weight_buffer, self.num_weights, self.size);

        for i in 0..weights.len() {
            weights[i] = initializer.get_weight();
        }
        for i in 0..biases.len() {
            biases[i] = initializer.get_bias();
        }
    }

    fn get_config(&self) -> NetLayerConfig {
        NetLayerConfig::FullyConnected(self.size, self.activation_fn)
    }
}