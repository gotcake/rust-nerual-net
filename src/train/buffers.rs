use crate::{
    stats::Stats,
    net::Net,
    layer::NetLayerBase,
    buffer::{
        RowBuffer,
        WeightBuffer
    }
};


pub struct TrainingBuffers {
    pub input_buffer: Vec<f32>,
    pub expected_output_buffer: Vec<f32>,
    pub output_buffers: RowBuffer<f32>,
    pub error_gradient_buffers: RowBuffer<f32>,
    pub input_error_buffer: Vec<f32>,
    pub error_stats: Stats,
    pub weight_deltas: WeightBuffer,
}

impl TrainingBuffers {
    pub fn for_net(net: &Net) -> Self {
        let layer_sizes: Vec<usize> = net.layer_iter()
            .map(NetLayerBase::output_size)
            .collect();
        TrainingBuffers {
            input_buffer: vec![0f32; net.input_size()],
            expected_output_buffer: vec![0f32; net.output_size()],
            output_buffers: RowBuffer::new_with_row_sizes(0.0, layer_sizes.as_ref()),
            error_gradient_buffers: RowBuffer::new_with_row_sizes(0.0, layer_sizes.as_ref()),
            input_error_buffer: vec![0f32; net.input_size()],
            error_stats: Stats::new(),
            weight_deltas: WeightBuffer::new_for_net(net),
        }
    }
}