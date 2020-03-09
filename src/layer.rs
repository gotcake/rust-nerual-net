use crate::initializer::RandomNetInitializer;
use crate::func::ActivationFn;
use std::fmt;

pub trait NetLayerBase {
    fn forward_pass(&self, input: &[f32], output: &mut[f32]);
    fn backprop(&self, output_errors: &[f32], inputs: &[f32], outputs: &[f32],
                learning_rate: f32, input_errors: &mut[f32], delta_target: &mut [f32]);
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn weight_buffer_size(&self) -> usize;
    fn store_weights_into(&self, target: &mut [f32]);
    fn load_weights_from(&mut self, source: &[f32]);
    fn add_weights_from(&mut self, source: &[f32]);
    fn initialize_weights(&mut self, initializer: &mut RandomNetInitializer);
    fn get_config(&self) -> NetLayerConfig;
}

#[derive(Clone, Debug, PartialEq)]
pub enum NetLayerConfig {
    FullyConnected(usize, ActivationFn)
}

impl NetLayerConfig {
    pub fn create_layer(
        &self,
        input_size: usize
    ) -> NetLayer {
        match self {
            &NetLayerConfig::FullyConnected(size, activation_fn) => {
                NetLayer::FullyConnected(
                    FullyConnectedNetLayer::new(
                        input_size,
                        size,
                        activation_fn
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

    #[inline]
    fn get_delegate_mut(&mut self) -> &mut dyn NetLayerBase {
        match self {
            NetLayer::FullyConnected(layer) => layer,
        }
    }

}

impl NetLayerBase for NetLayer {

    // NOTE: not using delegate functions for most frequently called methods to avoid dynamic dispatch

    fn forward_pass(&self, input: &[f32], output: &mut [f32]) {
        match self {
            NetLayer::FullyConnected(layer) => layer.forward_pass(input, output),
        }
    }

    fn backprop(&self, output_errors: &[f32], inputs: &[f32], outputs: &[f32], learning_rate: f32, input_errors: &mut [f32], delta_target: &mut [f32]) {
        match self {
            NetLayer::FullyConnected(layer) => layer.backprop(output_errors, inputs, outputs, learning_rate, input_errors, delta_target),
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

    fn store_weights_into(&self, target: &mut [f32]) {
        self.get_delegate().store_weights_into(target)
    }

    fn load_weights_from(&mut self, source: &[f32]) {
        self.get_delegate_mut().load_weights_from(source)
    }

    fn add_weights_from(&mut self, source: &[f32]) {
        self.get_delegate_mut().add_weights_from(source)
    }

    fn initialize_weights(&mut self, initializer: &mut RandomNetInitializer) {
        self.get_delegate_mut().initialize_weights(initializer);
    }

    fn get_config(&self) -> NetLayerConfig {
        self.get_delegate().get_config()
    }
}

#[derive(Clone)]
pub struct FullyConnectedNetLayer {
    input_size: usize,
    size: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
    activation_fn: ActivationFn,
}

impl fmt::Debug for FullyConnectedNetLayer {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        f.debug_struct("FullyConnectedNetLayer")
            .field("input_size", &self.size)
            .field("size", &self.size)
            .field("weights", &self.weights.len())
            .field("biases", &self.biases.len())
            .field("activation_fn", &self.activation_fn)
            .finish()
    }
}

impl FullyConnectedNetLayer {

    pub fn new(
        input_size: usize,
        size: usize,
        activation_fn: ActivationFn
    ) -> Self {
        FullyConnectedNetLayer {
            input_size,
            size,
            weights: vec![0.0; input_size * size],
            biases: vec![0.0; size],
            activation_fn,
        }
    }

    #[inline(always)]
    fn get_weight(&self, input_index: usize, node_index: usize) -> f32 {
        self.weights[input_index * self.size + node_index]
    }

    #[allow(dead_code)]
    #[inline(always)]
    fn set_weight(&mut self, input_index: usize, node_index: usize, value: f32) {
        self.weights[input_index * self.size + node_index] = value;
    }


}

impl NetLayerBase for FullyConnectedNetLayer {

    fn forward_pass(&self, input: &[f32], output: &mut[f32]) {

        debug_assert_eq!(input.len(), self.input_size);

        for node_index in 0..self.size {
            let mut sum = self.biases[node_index];
            for input_index in 0..self.input_size {
                sum = sum + input[input_index] * self.get_weight(input_index, node_index);
            }
            output[node_index] = self.activation_fn.get_activation(sum);
        }
    }

    fn backprop(&self, output_errors: &[f32], inputs: &[f32], outputs: &[f32],
                learning_rate: f32, input_errors: &mut [f32], delta_target: &mut [f32]) {

        debug_assert_eq!(output_errors.len(), self.size);
        debug_assert_eq!(input_errors.len(), self.input_size);
        debug_assert_eq!(inputs.len(), self.input_size);
        debug_assert_eq!(outputs.len(), self.size);
        debug_assert_eq!(delta_target.len(), self.weights.len() + self.biases.len());

        let bias_offset = self.weights.len();

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
                let connection_weight = self.get_weight(input_index, node_index);
                let weight_delta = -learning_rate * node_error_gradient * input;
                //let new_weight = connection_weight + weight_delta;
                // TODO
                delta_target[input_index * self.size + node_index] += weight_delta;
                // self.set_weight(input_index, node_index, new_weight);
                input_errors[input_index] += connection_weight * node_error_gradient;
            }
            // update bias
            delta_target[bias_offset + node_index] -= learning_rate * node_error_gradient;
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
        self.weights.len() + self.biases.len()
    }

    fn store_weights_into(&self, target: &mut [f32]) {

        debug_assert_eq!(target.len(), self.weights.len() + self.biases.len());

        let num_weights = self.weights.len();
        for i in 0..num_weights {
            target[i] = self.weights[i];
        }
        for i in 0..self.biases.len() {
            target[i + num_weights] = self.biases[i];
        }
    }

    fn load_weights_from(&mut self, source: &[f32]) {

        debug_assert_eq!(source.len(), self.weights.len() + self.biases.len());

        let num_weights = self.weights.len();
        for i in 0..num_weights {
            self.weights[i] = source[i];
        }
        for i in 0..self.biases.len() {
            self.biases[i] = source[i + num_weights];
        }
    }

    fn add_weights_from(&mut self, source: &[f32]) {
        debug_assert_eq!(source.len(), self.weights.len() + self.biases.len());
        let num_weights = self.weights.len();
        for i in 0..num_weights {
            self.weights[i] += source[i];
        }
        for i in 0..self.biases.len() {
            self.biases[i] += source[i + num_weights];
        }
    }

    fn initialize_weights(&mut self,initializer: &mut RandomNetInitializer) {
        for i in 0..self.weights.len() {
            self.weights[i] = initializer.get_weight();
        }
        for i in 0..self.biases.len() {
            self.biases[i] = initializer.get_bias();
        }
    }

    fn get_config(&self) -> NetLayerConfig {
        NetLayerConfig::FullyConnected(self.size, self.activation_fn)
    }
}