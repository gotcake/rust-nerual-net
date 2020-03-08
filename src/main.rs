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
    data::TrainingSet,
    train::NetTrainerBuilder,
    train::BackpropOptions
};
use crate::train::{ParamFactory, NetTrainer, TrainingResult};
use crate::net::NetConfig;
use crate::func::{ActivationFn, CompletionFn, MiniBatchSize, LearningRateFn, ErrorFn};

fn main() -> Result<(), Box<dyn Error>> {


    //let seed = [0x1235, 0x5663, 0x8392, 0x1211];

    let training_set = TrainingSet::new_from_csv("data/2x2_lines_binary.csv")?;
    let independent_columns = training_set.get_named_column_selection(&vec!["0_0", "0_1", "1_0", "1_1"])?;
    let dependent_columns = training_set.get_named_column_selection(&vec!["has_horizontal", "has_vertical"])?;

    let mut net_trainer: NetTrainer = NetTrainerBuilder::default()
        .training_set(training_set)
        .independent_columns(independent_columns)
        .dependent_columns(dependent_columns)
        .net_config_factory(Box::new(net_factory))
        .backprop_options_factory(Box::new(backprop_options_factory))
        .build()?;

    let result: TrainingResult = net_trainer.execute()?;

    println!("duration = {}s, error_stats = {:?}", result.duration.as_secs_f32(), &result.error_stats);

    Ok(())

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
