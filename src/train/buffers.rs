use crate::{
    stats::Stats,
    net::Net,
    layer::NetLayerBase,
    buffer::{
        RowBuffer,
    }
};


pub struct TrainingBuffers {
    pub output_buffers: RowBuffer<f32>,
    pub error_gradient_buffers: RowBuffer<f32>,
    pub input_error_buffer: Vec<f32>,
    pub error_stats: Stats,
    pub weight_deltas: RowBuffer<f32>,
}

impl TrainingBuffers {
    pub fn for_net(net: &Net) -> Self {
        let layer_sizes: Vec<usize> = net.layer_iter()
            .map(NetLayerBase::output_size)
            .collect();
        TrainingBuffers {
            output_buffers: RowBuffer::new_with_row_sizes(0.0, &layer_sizes),
            error_gradient_buffers: RowBuffer::new_with_row_sizes(0.0, &layer_sizes),
            input_error_buffer: vec![0f32; net.input_size()],
            error_stats: Stats::new(),
            weight_deltas: net.new_zeroed_weight_buffer(),
        }
    }
}