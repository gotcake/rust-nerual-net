use crate::buffer::RowBuffer;
use crate::net::Net;
use crate::layer::{NetLayer, NetLayerBase};

pub struct WeightBuffer {
    buffer: RowBuffer<f32>
}

#[allow(dead_code)]
impl WeightBuffer {

    pub fn new_for_net(net: &Net) -> Self {
        let layer_sizes: Vec<usize> = net.layer_iter()
            .map(NetLayer::weight_buffer_size)
            .collect();
        WeightBuffer {
            buffer: RowBuffer::new_with_row_sizes(0.0, layer_sizes.as_slice())
        }
    }

    pub fn extract_from_net(&mut self, net: &Net) {
        for (i, layer) in net.layer_iter().enumerate() {
            layer.write_weights_into(self.buffer.get_row_mut(i));
        }
    }

    pub fn load_into_net(&self, net: &mut Net) {
        for (i, layer) in net.layer_iter_mut().enumerate() {
            layer.read_weights_from(self.buffer.get_row(i));
        }
    }

    pub fn add_to_net(&self, net: &mut Net) {
        for (i, layer) in net.layer_iter_mut().enumerate() {
            layer.add_weights_from(self.buffer.get_row(i));
        }
    }

    pub fn copy_into(&self, other: &mut WeightBuffer) {
        self.buffer.copy_into(&mut other.buffer);
    }

    pub fn subtract(&mut self, subtract: &WeightBuffer) {
        self.buffer.subtract(&subtract.buffer);
    }

    pub fn add(&mut self, other: &WeightBuffer) {
        self.buffer.add(&other.buffer);
    }

    pub fn add_with_multiplier(&mut self, other: &WeightBuffer, multiplier: f32) {
        self.buffer.add_with_multiplier(&other.buffer, multiplier);
    }

    pub fn get_buffer_mut(&mut self) -> &mut RowBuffer<f32> {
        &mut self.buffer
    }

}