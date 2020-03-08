use core::slice;

use crate::{
    layer::{
        NetLayer,
        FullyConnectedNetLayer,
        NetLayerBase,
        NetLayerConfig
    },
    initializer::NetInitializer,
    func::ActivationFn,
    buffer::RowBuffer,
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
        hidden_layer_sizes: Vec<usize>,
        activation_fn: ActivationFn,
    ) -> Self {
        assert!(input_size > 0);
        assert!(output_size > 0);
        let mut layers: Vec<NetLayerConfig> = Vec::with_capacity(hidden_layer_sizes.len() + 1);
        for layer_size in hidden_layer_sizes {
            assert!(layer_size > 0);
            layers.push(NetLayerConfig::FullyConnected(layer_size, activation_fn));
        }
        layers.push(NetLayerConfig::FullyConnected(output_size, activation_fn));
        NetConfig {
            input_size,
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

    pub fn from_config(
        config: NetConfig,
        initializer: Box<dyn NetInitializer>
    ) -> Self {

        assert!(config.input_size > 0);
        assert!(config.layers.len() > 0);

        let mut layers = Vec::with_capacity(config.layers.len());
        let mut layer_input_size = config.input_size;
        let mut layer_idx = 0;
        let mut initializer = initializer;
        for ref layer_config in config.layers {
            let layer = layer_config.create_layer(layer_input_size, layer_idx, initializer.as_mut());
            layers.push(layer);
        }
        let max_buffer_size = layers.iter()
            .map(NetLayer::weight_buffer_size)
            .max()
            .unwrap();
        Net {
            input_size: config.input_size,
            output_size: layers.last().unwrap().output_size(),
            max_buffer_size,
            layers
        }
    }

    fn predict_with_buffers(&self, input: &[f32], output: &mut[f32], buffer_a: &mut[f32], buffer_b: &mut[f32]) {

        let num_layers = self.layers.len();

        debug_assert_eq!(input.len(), self.input_size);
        debug_assert_eq!(output.len(), self.output_size);
        debug_assert!(buffer_a.len() >= self.max_buffer_size);
        debug_assert!(buffer_b.len() >= self.max_buffer_size);
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

}