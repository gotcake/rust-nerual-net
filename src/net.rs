use std::slice;
use std::cell::RefCell;

use crate::layer::NetLayer;
use crate::layer::NetLayerBase;
use crate::layer::NetLayerConfig;
use crate::buffer::RowBuffer;
use crate::initializer::RandomNetInitializer;
use crate::func::ActivationFn;
use crate::utils::split_slice_mut;
use crate::func::ErrorFn;
use crate::train::NetTrainingContext;


#[derive(Clone, Debug, PartialEq)]
pub struct NetConfig {
    input_size: usize,
    layers: Vec<NetLayerConfig>,
}

impl NetConfig {

    pub fn new_fully_connected(
        input_size: usize,
        output_size: usize,
        hidden_layer_sizes: impl AsRef<[usize]>,
        activation_fn: ActivationFn,
    ) -> Self {
        let hidden_layer_sizes = hidden_layer_sizes.as_ref();
        assert!(input_size > 0);
        assert!(output_size > 0);
        let mut layers: Vec<NetLayerConfig> = Vec::with_capacity(hidden_layer_sizes.len() + 1);
        for layer_size in hidden_layer_sizes {
            assert!(*layer_size > 0);
            layers.push(NetLayerConfig::FullyConnected(*layer_size, activation_fn));
        }
        layers.push(NetLayerConfig::FullyConnected(output_size, activation_fn));
        NetConfig {
            input_size,
            layers
        }
    }

    pub fn create_net(&self) -> Net {

        let mut layers = Vec::with_capacity(self.layers.len());
        let mut layer_input_size = self.input_size;

        for layer_config in &self.layers {
            let layer = layer_config.create_layer(layer_input_size);
            layer_input_size = layer.output_size();
            layers.push(layer);
        }

        Net::new(self.input_size, layers)

    }

}

#[derive(Clone, Debug)]
pub struct Net {
    weight_buffer: RowBuffer,
    input_size: usize,
    output_size: usize,
    layers: Vec<NetLayer>,
    prediction_buffers: RefCell<RowBuffer>, // RefCell is needed to allow mutable borrow
}

#[allow(dead_code)]
impl<'a> Net {

    fn new(input_size: usize, layers: Vec<NetLayer>) -> Self {

        assert!(input_size > 0);
        assert!(layers.len() > 0);

        for layer in &layers {
            assert!(layer.input_size() > 0);
            assert!(layer.output_size() > 0);
        }

        let row_buffer_sizes: Vec<usize> = layers.iter()
            .map(NetLayer::weight_buffer_size)
            .collect();

        let weight_buffer = RowBuffer::new_with_row_sizes(0.0, row_buffer_sizes);
        let max_output_size = layers.iter().map(NetLayer::output_size).max().unwrap();

        Net {
            weight_buffer,
            input_size,
            output_size: layers.last().unwrap().output_size(),
            layers,
            prediction_buffers: RefCell::new(RowBuffer::new_with_row_sizes(0.0, [max_output_size, max_output_size])),
        }

    }

    fn predict_with(&mut self, input: &[f32], output: &mut[f32]) {

        let num_layers = self.layers.len();

        debug_assert_eq!(input.len(), self.input_size);
        debug_assert_eq!(output.len(), self.output_size);

        // TODO: handle 1 layer??
        debug_assert!(num_layers > 1);

        let mut prediction_buffers = self.prediction_buffers.borrow_mut();
        let (mut input_buffer, mut output_buffer) = prediction_buffers.split_rows(0, 1);

        self.first_layer().forward_pass(
            self.weight_buffer.get_first_row(),
            input,
            input_buffer,
        );
        for row_index in 1..num_layers-1 {
            self.layer(row_index).forward_pass(
                self.weight_buffer.get_row(row_index),
                input_buffer,
                output_buffer,
            );
            std::mem::swap(&mut input_buffer, &mut output_buffer);
        }
        self.last_layer().forward_pass(
            self.weight_buffer.get_last_row(),
            input_buffer,
            output,
        );

    }

    pub fn predict(&mut self, input: impl AsRef<[f32]>) -> Vec<f32> {
        let mut output = vec![0f32; self.output_size];
        self.predict_with(input.as_ref(), output.as_mut_slice());
        output
    }

    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    #[inline]
    pub fn layer(&self, index: usize) -> &NetLayer {
        &self.layers[index]
    }

    #[inline]
    pub fn first_layer(&self) -> &NetLayer {
        &self.layers[0]
    }

    #[inline]
    pub fn last_layer(&self) -> &NetLayer {
        &self.layers[self.layers.len() - 1]
    }

    #[inline]
    pub fn layer_iter(&self) -> slice::Iter<NetLayer> {
        self.layers.iter()
    }

    #[inline]
    pub fn layer_iter_mut(&mut self) -> slice::IterMut<NetLayer> {
        self.layers.iter_mut()
    }

    #[inline]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    #[inline]
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub(crate) fn new_zeroed_weight_buffer(&self) -> RowBuffer {
        let layer_sizes: Vec<usize> = self.layer_iter()
            .map(NetLayer::weight_buffer_size)
            .collect();
        RowBuffer::new_with_row_sizes(0.0, layer_sizes)
    }

    #[inline]
    pub fn get_weights(&self) -> &RowBuffer {
        &self.weight_buffer
    }

    #[inline]
    pub fn get_weights_mut(&mut self) -> &mut RowBuffer {
        &mut self.weight_buffer
    }

    pub fn initialize_weights(&mut self, initializer: &mut RandomNetInitializer) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.initialize_weights(self.weight_buffer.get_row_mut(i), initializer);
        }
    }

    pub fn get_config(&self) -> NetConfig {
        let layers: Vec<NetLayerConfig> = self.layer_iter()
            .map(NetLayer::get_config)
            .collect();
        NetConfig {
            input_size: self.input_size,
            layers
        }
    }

    pub fn get_training_context(&'a mut self) -> NetTrainingContext<'a> {
        NetTrainingContext::new(self)
    }

}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_weight_buffer() {

        let config = NetConfig::new_fully_connected(
            4,
            2,
            [3],
            ActivationFn::standard_logistic_sigmoid()
        );

        let mut net = config.create_net();

        let mut buf = net.new_zeroed_weight_buffer();
        let mut buf2 = net.new_zeroed_weight_buffer();

        assert_eq!(buf.num_rows(), 2);
        assert_eq!(buf.get_row(0).len(), 4 * 3 + 3);
        assert_eq!(buf.get_row(1).len(), 3 * 2 + 2);

        for (i, element) in buf.get_buffer_mut().iter_mut().enumerate() {
            *element = i as f32;
        }

        buf.copy_into(net.get_weights_mut());
        net.get_weights().copy_into(&mut buf2);

        for (i, element) in buf2.get_buffer().iter().enumerate() {
            assert_eq!(i as f32, *element);
        }

    }

    #[test]
    fn test_config_round_trip_fully_connected() {

        let config = NetConfig::new_fully_connected(
            4,
            2,
            [5, 4, 3],
            ActivationFn::standard_logistic_sigmoid()
        );

        let mut net = config.create_net();

        let config2 = net.get_config();

        let mut net2 = config2.create_net();

        let config3 = net2.get_config();

        assert_eq!(config3, config);

    }

}
