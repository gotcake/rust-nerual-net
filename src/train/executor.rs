use crate::{
    train::{
        TrainingContext,
        task::{Task, TaskResult},
    },
    net::Net,
    train::task::TaskError
};
use std::{
    sync::mpsc,
    sync::atomic::{AtomicBool, Ordering},
    sync::Arc,
    task::{Context, Poll},
    error::Error,
    net::IpAddr,
    future::Future,
    thread,
    sync::mpsc::{Receiver, Sender},
    sync::mpsc::{SendError, TryIter}
};
use crossbeam::internal::SelectHandle;


quick_error! {
    #[derive(Debug)]
    pub enum ExecutorError {
        TaskError(err: TaskError) {
            description("TaskError")
            display("TaskError: {}", err)
        }
    }
}


pub enum Executor {
    Local(usize),
    Distributed { discover_addr: IpAddr, discover_port: u16 },
}

impl Executor {
    pub fn get_instance(&self) -> Result<Box<dyn ExecutorInstance>, ExecutorError> {
        match self {
            &Executor::Distributed { discover_addr, discover_port } => {
                unimplemented!();
            },
            &Executor::Local(num_workers) => Ok(Box::new(LocalExecutor::new(num_workers))),
        }
    }
}

pub trait ExecutorInstance {
    fn start(&self) -> Result<ExecutorControlMaster, ExecutorError>;
    fn stop(&self);
}

struct LocalExecutor {
    num_workers: usize,
    stopped: Arc<AtomicBool>
}

impl LocalExecutor {
    fn new(num_workers: usize) -> Self {
        LocalExecutor {
            num_workers,
            stopped: Arc::new(AtomicBool::new(false))
        }
    }
}

impl ExecutorInstance for LocalExecutor {
    fn start(&self) -> Result<ExecutorControlMaster, ExecutorError> {

        let (ctrl_master, ctrl_slave) = executor_control();
        self.stopped.store(false, Ordering::Relaxed);

        for _ in 0..self.num_workers {
            let ctrl_slave = ctrl_slave.clone();
            let stopped_flag = self.stopped.clone();
            thread::spawn(move || {
                // wrap logic in a function to allow error cascading with "?"
                let inner_fn = || -> Result<(), Box<Error>> {
                    while !stopped_flag.load(Ordering::Relaxed) {

                        // try to get next task
                        let task = ctrl_slave.get_next_task()?;

                        // execute task
                        match task.exec() {
                            Ok(result) => {
                                ctrl_slave.results_sender.send(Ok(result))?;
                            },
                            Err(err) => {
                                ctrl_slave.results_sender.send(Err(ExecutorError::TaskError(err)))?;
                            },
                        }
                    }
                    Ok(())
                };
                // if a channel-based error occurred, signal all to stop
                if inner_fn().is_err() {
                    stopped_flag.store(true, Ordering::Relaxed);
                }
            });
        }
        Ok(ctrl_master)
    }

    fn stop(&self) {
        self.stopped.store(true, Ordering::Relaxed)
    }
}


fn executor_control() -> (ExecutorControlMaster, ExecutorControlSlave) {
    // A zero-sized mpmc (though used as spmr) channel for sending tasks to executor workers
    let (task_sender, task_receiver) = crossbeam::channel::bounded(0);
    // An unbounded mpsc channel for sending results back to the driver
    let (results_sender, results_receiver) = mpsc::channel();
    let master = ExecutorControlMaster {
        task_sender,
        results_receiver,
    };
    let slave = ExecutorControlSlave {
        task_receiver,
        results_sender,
    };
    (master, slave)
}

pub struct ExecutorControlMaster {
    task_sender: crossbeam::channel::Sender<Task>,
    results_receiver: Receiver<Result<TaskResult, ExecutorError>>,
}

impl ExecutorControlMaster {

    pub fn has_waiting_executor(&self) -> bool {
        // NOTE: use of crossbeam internal API, may break at any time...
        // If this does break, we can probably use the Select API instead.
        self.task_sender.is_ready()
    }

    pub fn send_task(&self, task: Task) -> Result<(), Box<Error>> {
        self.task_sender.send(task)?;
        Ok(())
    }

    pub fn try_get_results(&self) -> TryIter<Result<TaskResult, ExecutorError>> {
        self.results_receiver.try_iter()
    }

}

#[derive(Clone)]
pub struct ExecutorControlSlave {
    task_receiver: crossbeam::channel::Receiver<Task>,
    results_sender: Sender<Result<TaskResult, ExecutorError>>,
}

impl ExecutorControlSlave {

    fn send_results(&self, result: Result<TaskResult, ExecutorError>) -> Result<(), Box<Error>> {
        self.results_sender.send(result)?;
        Ok(())
    }

    fn get_next_task(&self) -> Result<Task, Box<Error>> {
        Ok(self.task_receiver.recv()?)
    }

}