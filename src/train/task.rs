use crate::{
    data::PreparedDataSet,
    net::Net,
    train::{
        BackpropOptions,
        backprop::backprop_stage_task_impl
    },
    stats::Stats
};


pub struct Task {
    pub id: usize,
    pub training_set: PreparedDataSet,
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
                let (error_stats, batch_count) = backprop_stage_task_impl(&mut self.net, &self.training_set, options);
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





