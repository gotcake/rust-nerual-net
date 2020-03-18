#![feature(slice_index_methods)]

#[macro_use]
extern crate quick_error;

#[macro_use]
extern crate derive_builder;

mod layer;
mod net;
mod initializer;
mod utils;
mod data;
mod stats;
mod train;
mod buffer;
mod func;

use std::error::Error;
use std::time::Duration;

use crate::{
    train::NetTrainerBuilder,
    train::BackpropOptions
};
use crate::train::{NetTrainer, TrainingResult, ParamFactory, TrainingEvent};
use crate::net::NetConfig;
use crate::func::{ActivationFn, CompletionFn, MiniBatchSize, LearningRateFn, ErrorFn};
use crate::data::PreparedDataSet;

fn main() -> Result<(), Box<dyn Error>> {


    //let seed = [0x1235, 0x5663, 0x8392, 0x1211];

    let data_set = PreparedDataSet::from_csv(
        "data/2x2_lines_binary.csv",
        ["0_0", "0_1", "1_0", "1_1"],
        ["has_horizontal", "has_vertical"]
    )?;

    let mut net_trainer: NetTrainer = NetTrainerBuilder::default()
        .data_set(data_set)
        .net_config_factory(Box::new(net_factory))
        .backprop_options_factory(Box::new(backprop_options_factory))
        .observer(Box::new(observer_callback))
        .build()?;

    let result: TrainingResult = net_trainer.execute()?;

    println!("duration = {}s, error_stats = {:?}", result.duration.as_secs_f32(), &result.error_stats);

    Ok(())

}


fn observer_callback(event: &TrainingEvent) {

}

fn net_factory(_params: &mut dyn ParamFactory) -> NetConfig {
    NetConfig::new_fully_connected(
        4,
        2,
        vec![3],
        ActivationFn::standard_logistic_sigmoid()
    )
}

fn backprop_options_factory(_params: &mut dyn ParamFactory) -> BackpropOptions {
    BackpropOptions {
        completion_fn: CompletionFn::stop_after_duration(Duration::from_secs(15)),
        mini_batch_size_fn: MiniBatchSize::Full,
        learning_rate_fn: LearningRateFn::standard_tanh_logarithmic_descent(),
        error_fn: ErrorFn::SquaredError,
        multi_threading: None
    }
}
