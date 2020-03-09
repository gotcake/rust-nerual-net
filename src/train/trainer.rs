use rand::{Rng, FromEntropy, SeedableRng};

use crate::{
    func::CompletionFn,
    net::{Net, NetConfig},
    data::PreparedDataSet,
    stats::Stats,
    train::{
        BackpropOptions,
        Executor,
        task::{Task, TaskResult},
        ExecutorControlMaster,
        task::TaskOp
    },
    initializer::RandomNetInitializer,
    utils::stable_hash_seed
};
use std::{
    time::SystemTime,
    cell::RefCell,
    rc::Rc,
    time::Duration,
    thread,
    error::Error,
};

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
enum ParamSpec {
    RangeUsize(usize, usize),
    RangeF32(f32, f32),
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
enum ParamValue {
    Usize(usize),
    F32(f32)
}

pub trait ParamFactory {
    fn range_usize(&mut self, key: String, low: usize, high: usize) -> usize;
    fn range_f32(&mut self, low: f32, high: f32) -> f32;
}

pub trait Optimizer {
    fn next_parameters(&mut self, id: usize) -> Box<dyn ParamFactory>;
    fn report(&mut self, results: &TaskResult);
}

pub struct RandomOptimizer {
    rng: Rc<RefCell<rand_xorshift::XorShiftRng>>
}

impl RandomOptimizer {
    pub fn new_from_entropy() -> Self {
        RandomOptimizer {
            rng: Rc::new(RefCell::new(rand_xorshift::XorShiftRng::from_entropy()))
        }
    }
    #[allow(dead_code)]
    pub fn new_from_seed(seed: &str) -> Self {
        let seed_bytes = stable_hash_seed(seed);
        RandomOptimizer {
            rng: Rc::new(RefCell::new(rand_xorshift::XorShiftRng::from_seed(seed_bytes)))
        }
    }
}

impl Optimizer for RandomOptimizer {

    fn next_parameters(&mut self, _id: usize) -> Box<dyn ParamFactory> {
        Box::new(RandomParamFactory {
            rng: self.rng.clone(),
        })
    }

    fn report(&mut self, _results: &TaskResult) {
        // no-op
    }
}

struct RandomParamFactory {
    rng: Rc<RefCell<rand_xorshift::XorShiftRng>>
}

#[allow(dead_code)]
impl ParamFactory for RandomParamFactory {

    fn range_usize(&mut self, _key: String, low: usize, high: usize) -> usize {
        return (&*self.rng).borrow_mut().gen_range(low, high);
    }

    fn range_f32(&mut self, low: f32, high: f32) -> f32 {
        return (&*self.rng).borrow_mut().gen_range(low, high);
    }

}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum NetTrainerMode {
    Standard,
    Evolutionary { trials_per_generation: usize }
}

#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct NetTrainer {
    training_set: PreparedDataSet,
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
    #[builder(setter(skip))]
    next_task_id: usize,
}

fn default_optimizer_factory() -> Box<dyn Optimizer> {
    Box::new(RandomOptimizer::new_from_entropy())
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

impl NetTrainer {

    pub fn execute(&mut self) -> Result<TrainingResult, Box<dyn Error>> {

        let executor = self.executor.get_instance()?;

        let ctrl_master = executor.start()?;

        let result = match self.mode {
            NetTrainerMode::Standard => self.train_standard(ctrl_master),
            NetTrainerMode::Evolutionary { trials_per_generation: _ } => { unimplemented!(); },
        };

        executor.stop();

        result
    }

    fn train_standard(&mut self, ctrl_master: ExecutorControlMaster) -> Result<TrainingResult, Box<dyn Error>> {

        let mut optimizer: Box<dyn Optimizer> = self.optimizer_factory.as_ref()();
        let mut num_trials = 0;
        let start_time = SystemTime::now();
        let mut best_result: Option<TaskResult> = None;

        'train: loop {


            // wait until a executor is ready, processing results in the meantime
            'wait: loop {

                // process any pending results
                for result in ctrl_master.try_get_results() {
                    match result {
                        Ok(res) => {
                            optimizer.report(&res);
                            best_result = Some(match best_result {
                                None => res,
                                Some(best) => {
                                    if best.error_stats.mean() > res.error_stats.mean() {
                                        res
                                    } else {
                                        best
                                    }
                                },
                            });
                            num_trials += 1;
                        },
                        // TODO: error handling?
                        Err(err) => println!("ERROR: {}", err),
                    }
                }

                // quit if results meet criteria
                if let Some(best) = &best_result {
                    if self.global_completion_fn.should_stop_training(num_trials, start_time, &best.error_stats) {
                        break 'train;
                    }
                }

                if ctrl_master.has_waiting_executor() {
                    break 'wait;
                } else {
                    thread::sleep(Duration::from_millis(20));
                }

            }

            // send next task to execute
            ctrl_master.send_task(self.gen_backprop_task(optimizer.as_mut()))?;

        }

        let best = best_result.unwrap();

        Ok(TrainingResult {
            net: best.net,
            error_stats: best.error_stats,
            duration: SystemTime::now().duration_since(start_time)?,
        })

    }

    fn next_task_id(&mut self) -> usize {
        let id = self.next_task_id;
        self.next_task_id += 1;
        id
    }

    fn gen_backprop_task(&mut self, optimizer: &mut dyn Optimizer) -> Task {

        let task_id = self.next_task_id();

        let mut params = optimizer.next_parameters(task_id);

        let net_config_factory = self.net_config_factory.as_ref();

        let net_config: NetConfig = net_config_factory(params.as_mut());

        let mut net = net_config.create_net();

        net.initialize_weights(&mut self.initializer);

        // TODO: cleanup
        println!("{:?}", net);

        let backprop_options: BackpropOptions = self.backprop_options_factory.as_ref()(params.as_mut());

        Task {
            id: task_id,
            training_set: self.training_set.clone(),
            net,
            op: TaskOp::Backprop(backprop_options)
        }

    }

}