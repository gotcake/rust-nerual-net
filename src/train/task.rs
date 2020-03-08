use crate::train::{TrainingContext, BackpropOptions};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::net::Net;
use crate::data::{TrainingSet, ColumnSelection};
use crate::train::backprop::backprop_stage_task_impl;
use crate::stats::Stats;
use std::error::Error;
use std::time::Duration;


pub struct Task {
    pub id: usize,
    pub training_set: TrainingSet,
    pub independent_columns: ColumnSelection,
    pub dependent_columns: ColumnSelection,
    pub net: Net,
    pub op: TaskOp,
}

quick_error! {
    #[derive(Debug)]
    pub enum TaskError {
        None {
            description("None")
        }
    }
}

pub struct TaskResult {
    pub id: usize,
    pub net: Net,
    pub error_stats: Stats,
    pub epoch: usize,
}

pub enum TaskOp {
    Backprop(BackpropOptions)
}

impl Task {
    pub fn exec(mut self) -> Result<TaskResult, TaskError> {
        match self.op {
            TaskOp::Backprop(ref options) => {
                let context = TrainingContext {
                    dependent_columns: self.dependent_columns,
                    independent_columns: self.independent_columns
                };
                let (error_stats, batch_count) = backprop_stage_task_impl(&mut self.net, &context, &self.training_set, options);
                Ok(TaskResult {
                    id: self.id,
                    net: self.net,
                    error_stats,
                    epoch: batch_count
                })
            },
        }
    }
}





