use core::slice;

use crate::{
    layer::{
        NetLayer,
        NetLayerBase,
        NetLayerConfig,
    },
    buffer::RowBuffer,
    initializer::RandomNetInitializer,
    func::ActivationFn,
};

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

        assert!(self.input_size > 0);
        assert!(self.layers.len() > 0);

        let mut layers = Vec::with_capacity(self.layers.len());
        let mut layer_input_size = self.input_size;
        for layer_config in &self.layers {
            let layer = layer_config.create_layer(layer_input_size);
            layer_input_size = layer.output_size();
            layers.push(layer);
        }
        let max_buffer_size = layers.iter()
            .map(NetLayer::weight_buffer_size)
            .max()
            .unwrap();
        Net {
            input_size: self.input_size,
            output_size: layers.last().unwrap().output_size(),
            max_buffer_size,
            layers
        }

    }

}

#[derive(Clone, Debug)]
pub struct Net {
    input_size: usize,
    output_size: usize,
    max_buffer_size: usize,
    layers: Vec<NetLayer>,
}

#[allow(dead_code)]
impl Net {

    // TODO:
    // Consider using contiguous memory as the backing store for weights.
    // Will probably need some wild unsafe code to make it work, but it will help with
    // Cache locality and writing data to and from a weight buffer.

    fn predict_with_buffers(&self, input: &[f32], output: &mut[f32], buffer_a: &mut[f32], buffer_b: &mut[f32]) {

        let num_layers = self.layers.len();

        debug_assert_eq!(input.len(), self.input_size);
        debug_assert_eq!(output.len(), self.output_size);
        debug_assert!(buffer_a.len() >= self.max_buffer_size);
        debug_assert!(buffer_b.len() >= self.max_buffer_size);

        // TODO: handle 1 layer??
        debug_assert!(num_layers > 1);


        let mut input_buffer = buffer_a;
        let mut output_buffer = buffer_b;

        self.layers[0].forward_pass(
            input,
            input_buffer,
        );
        for row_index in 1..num_layers-1 {
            self.layers[row_index].forward_pass(
                input_buffer,
                output_buffer,
            );
            std::mem::swap(&mut input_buffer, &mut output_buffer);
        }
        self.layers[num_layers - 1].forward_pass(
            input_buffer,
            output,
        );

    }

    pub fn predict(&self, input: Vec<f32>) -> Vec<f32> {
        let mut output = vec![0f32; self.output_size];
        self.predict_with_buffers(
            input.as_slice(),
            output.as_mut_slice(),
            vec![0f32; self.max_buffer_size].as_mut_slice(),
            vec![0f32; self.max_buffer_size].as_mut_slice()
        );
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

    #[inline]
    pub fn max_buffer_size(&self) -> usize { self.max_buffer_size }

    pub fn store_weights_into(&self, buffer: &mut RowBuffer<f32>) {
        debug_assert_eq!(buffer.num_rows(), self.layers.len());
        for i in 0..self.layers.len() {
            self.layers[i].store_weights_into(buffer.get_row_mut(i));
        }
    }

    pub fn load_weights_from(&mut self, buffer: &RowBuffer<f32>) {
        debug_assert_eq!(buffer.num_rows(), self.layers.len());
        for i in 0..self.layers.len() {
            self.layers[i].load_weights_from(buffer.get_row(i));
        }
    }

    pub fn add_weights_from(&mut self, buffer: &RowBuffer<f32>) {
        debug_assert_eq!(buffer.num_rows(), self.layers.len());
        for i in 0..self.layers.len() {
            self.layers[i].add_weights_from(buffer.get_row(i));
        }
    }

    pub fn new_zeroed_weight_buffer(&self) -> RowBuffer<f32> {
        let layer_sizes: Vec<usize> = self.layer_iter()
            .map(NetLayer::weight_buffer_size)
            .collect();
        RowBuffer::new_with_row_sizes(0.0, layer_sizes)
    }

    pub fn get_weights(&self) -> RowBuffer<f32> {
        let mut buf = self.new_zeroed_weight_buffer();
        self.store_weights_into(&mut buf);
        buf
    }

    pub fn initialize_weights(&mut self, initializer: &mut RandomNetInitializer) {
        for layer in &mut self.layers {
            layer.initialize_weights(initializer);
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

        net.load_weights_from(&buf);
        net.store_weights_into(&mut buf2);

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
