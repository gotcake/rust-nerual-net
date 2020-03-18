use rand::{Rng, FromEntropy, SeedableRng};

use crate::func::CompletionFn;
use crate::net::{Net, NetConfig};
use crate::data::PreparedDataSet;
use crate::stats::Stats;
use crate::train::backprop::BackpropOptions;
use crate::train::executor::Executor;
use crate::train::task::{Task, TaskResult, TaskOp, TaskUpdate};
use crate::train::executor::ExecutorControlMaster;
use crate::initializer::RandomNetInitializer;
use crate::utils::stable_hash_seed;
use crate::train::optimizer::{Optimizer, ParamFactory, RandomOptimizer};
use std::time::SystemTime;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;
use std::thread;
use std::error::Error;
use crate::train::executor::ExecutorEvent;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum NetTrainerMode {
    Standard,
    Evolutionary { trials_per_generation: usize }
}

#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct NetTrainer {
    data_set: PreparedDataSet,
    #[builder(default = "Executor::Local(1)")]
    executor: Executor,
    #[builder(default = "Box::new(default_optimizer_factory)")]
    optimizer_factory: Box<dyn Fn() -> Box<dyn Optimizer>>,
    #[builder(default = "NetTrainerMode::Standard")]
    mode: NetTrainerMode,
    net_config_factory: Box<dyn Fn(&mut dyn ParamFactory) -> NetConfig>,
    backprop_options_factory: Box<dyn Fn(&mut dyn ParamFactory) -> BackpropOptions>,
    #[builder(default = "CompletionFn::stop_after_epoch(1)")]
    global_completion_fn: CompletionFn,
    #[builder(setter(skip))]
    #[builder(default = "RandomNetInitializer::new_standard_from_entropy()")]
    initializer: RandomNetInitializer,
    #[builder(setter(strip_option))]
    observer: Option<Box<dyn Fn(&TrainingEvent)>>
}

fn default_optimizer_factory() -> Box<dyn Optimizer> {
    Box::new(RandomOptimizer::from_entropy())
}

#[allow(dead_code)]
impl NetTrainerBuilder {

    pub fn net_config(self, net_config: NetConfig) -> Self {
        let mut new = self;
        new.net_config_factory = Some(Box::new(move |_| -> NetConfig { net_config.clone() }));
        new
    }

    pub fn backprop_options(self, options: BackpropOptions) -> Self {
        let mut new = self;
        new.backprop_options_factory = Some(Box::new(move |_| -> BackpropOptions { options.clone() }));
        new
    }

}

pub struct TrainingResult {
    pub net: Net,
    pub error_stats: Stats,
    pub duration: Duration,
}

pub enum TrainingEvent<'a> {
    TaskSubmit(&'a Task),
    TaskAccepted {
        task_id: String,
        executor_id: String,
    },
    TaskResult(&'a TaskResult),
    TaskUpdate(TaskUpdate)
}

impl NetTrainer {

    pub fn execute(&mut self) -> Result<TrainingResult, Box<dyn Error>> {

        let executor = self.executor.get_instance()?;

        let ctrl_master = executor.start()?;

        let result = match self.mode {
            NetTrainerMode::Standard => StandardTrainerImpl::new(self).train(ctrl_master),
            NetTrainerMode::Evolutionary { trials_per_generation: _ } => { unimplemented!(); },
        };

        executor.stop();

        result
    }

}

trait TrainerImpl {

    fn get_config(&self) -> &NetTrainer;
    fn handle_result(&mut self, result: &TaskResult);
    fn next_task(&mut self, task_id: usize) -> Task;

    fn omit_event(&self, event: &TrainingEvent) {
        // TODO: logging?
        if let Some(observer) = self.get_config().observer.as_ref() {
            observer.as_ref()(event);
        }
    }

    fn gen_net(&self, params: &mut dyn ParamFactory) -> Net {
        let mut net: Net = self.get_config().net_config_factory.as_ref()(params).create_net();
        net.initialize_weights(&mut self.get_config().initializer.clone());
        net
    }

    fn gen_backprop_task(&self, task_id: usize, optimizer: &mut dyn Optimizer, data_set: PreparedDataSet, initial_state: Option<Net>) -> Task {

        let task_id = format!("backprop_{}", task_id);

        let mut params = optimizer.next_parameters(task_id.as_str());

        let net = initial_state.unwrap_or_else(|| self.gen_net(params.as_mut()));

        let backprop_options: BackpropOptions = self.get_config().backprop_options_factory.as_ref()(params.as_mut());

        Task {
            task_id,
            data_set,
            net,
            op: TaskOp::Backprop(backprop_options)
        }

    }

    fn train(&mut self, ctrl_master: ExecutorControlMaster) -> Result<TrainingResult, Box<dyn Error>> {

        let start_time = SystemTime::now();
        let epoch: usize = 0;
        let mut best: Option<TaskResult> = None;

        'train: loop {

            // wait until a executor is ready, processing results in the meantime
            'wait: loop {

                // process any pending results
                for event in ctrl_master.try_get_events() {
                    match event {
                        ExecutorEvent::TaskAccepted { task_id, executor_id } => {
                            self.omit_event(&TrainingEvent::TaskAccepted {
                                task_id,
                                executor_id,
                            });
                        },
                        ExecutorEvent::TaskResult(result) => {
                            self.handle_result(&result);
                            self.omit_event(&TrainingEvent::TaskResult(&result));
                            best = Some(match best {
                                None => result,
                                Some(best) => {
                                    if result.error_stats.mean() < best.error_stats.mean() {
                                        result
                                    } else {
                                        best
                                    }
                                },
                            });
                        },
                        ExecutorEvent::ExecutorError { task_id, executor_id, error} => {
                            // TODO?
                            eprintln!("Error: {:?}", error);
                        }
                        ExecutorEvent::TaskUpdate(update) => {
                            self.omit_event(&TrainingEvent::TaskUpdate(update));
                        }
                    }
                }

                // check if we should stop training
                if let Some(best) = &best {
                    if self.get_config().global_completion_fn.should_stop_training(epoch, start_time, &best.error_stats) {
                        break 'train;
                    }
                }

                // check if an executor is waiting
                if ctrl_master.has_waiting_executor() {
                    break 'wait;
                } else {
                    thread::sleep(Duration::from_millis(50));
                }

            }

            // send next task to execute
            let task = self.next_task(epoch);
            self.omit_event(&TrainingEvent::TaskSubmit(&task));
            ctrl_master.send_task(task)?;

        }

        let best = best.unwrap();

        Ok(TrainingResult {
            net: best.net,
            error_stats: best.error_stats,
            duration: SystemTime::now().duration_since(start_time)?,
        })

    }
}

impl TrainerImpl for StandardTrainerImpl<'_> {

    fn get_config(&self) -> &NetTrainer {
        self.config
    }

    fn handle_result(&mut self, result: &TaskResult) {
        self.optimizer.borrow_mut().report(result);
    }

    fn next_task(&mut self, task_id: usize) -> Task {
        self.gen_backprop_task(task_id, self.optimizer.borrow_mut().as_mut(), self.config.data_set.clone(), None)
    }
}

struct StandardTrainerImpl<'a> {
    config: &'a NetTrainer,
    optimizer: RefCell<Box<dyn Optimizer>>,
}

impl<'a> StandardTrainerImpl<'a> {
    fn new(config: &'a NetTrainer) -> Self {
        let optimizer = RefCell::new(config.optimizer_factory.as_ref()());
        StandardTrainerImpl {
            config,
            optimizer
        }
    }
}