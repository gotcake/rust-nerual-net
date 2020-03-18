use crate::{
    train::{
        task::{
            TaskResult,
            Task,
            TaskError
        }
    }
};
use std::{
    error::Error,
    net::IpAddr,
    thread,
    sync::{
        mpsc::{
            Receiver,
            Sender,
            self,
            TryIter,
        },
        Arc,
        atomic::{AtomicBool, Ordering},
    }
};
use crossbeam::internal::SelectHandle;
use crate::train::task::{TaskUpdate, TaskUpdateEmitter};


quick_error! {
    #[derive(Debug)]
    pub enum ExecutorError {
        TaskError(err: TaskError) {
            description("TaskError")
            display("TaskError: {}", err)
        }
    }
}


#[allow(dead_code)]
pub enum Executor {
    Local(usize),
    Distributed { discover_addr: IpAddr, discover_port: u16 },
}

impl Executor {
    pub fn get_instance(&self) -> Result<Box<dyn ExecutorInstance>, ExecutorError> {
        match self {
            &Executor::Distributed { discover_addr: _, discover_port: _ } => {
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

        for worker_idx in 0..self.num_workers {
            let executor_id = format!("local_executor_{}", worker_idx);
            let ctrl_slave = ctrl_slave.clone();
            let stopped_flag = self.stopped.clone();
            thread::spawn(move || {
                // wrap logic in a function to allow error cascading with "?"
                let inner_fn = || -> Result<(), Box<dyn Error>> {
                    while !stopped_flag.load(Ordering::Relaxed) {

                        // try to get next task
                        let task = ctrl_slave.get_next_task()?;
                        let task_id = task.task_id.clone();

                        ctrl_slave.accept_task(executor_id.clone(), task.task_id.clone())?;

                        // execute task
                        match task.exec(&ctrl_slave) {
                            Ok(result) => {
                                ctrl_slave.send_result(result)?;
                            },
                            Err(err) => {
                                ctrl_slave.send_err(task_id, executor_id.clone(), ExecutorError::TaskError(err))?;
                            },
                        }
                    }
                    Ok(())
                };
                // if a channel-based error occurred, signal all to stop
                if inner_fn().is_err() {
                    // TODO: log error?
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
    // An unbounded mpsc channel for sending results back to the
    let (event_sender, event_receiver) = mpsc::channel();
    let master = ExecutorControlMaster {
        task_sender,
        event_receiver,
    };
    let slave = ExecutorControlSlave {
        task_receiver,
        event_sender,
    };
    (master, slave)
}

pub enum ExecutorEvent {
    TaskAccepted {
        task_id: String,
        executor_id: String,
    },
    TaskResult(TaskResult),
    ExecutorError {
        task_id: String,
        executor_id: String,
        error: ExecutorError,
    },
    TaskUpdate(TaskUpdate),
}

pub struct ExecutorControlMaster {
    task_sender: crossbeam::channel::Sender<Task>,
    event_receiver: Receiver<ExecutorEvent>,
}

impl ExecutorControlMaster {

    pub fn has_waiting_executor(&self) -> bool {
        // NOTE: use of crossbeam internal API, may break at any time...
        // If this does break, we can probably use the Select API instead.
        self.task_sender.is_ready()
    }

    pub fn send_task(&self, task: Task) -> Result<(), Box<dyn Error>> {
        self.task_sender.send(task)?;
        Ok(())
    }

    pub fn try_get_events(&self) -> TryIter<ExecutorEvent> {
        self.event_receiver.try_iter()
    }

}

#[derive(Clone)]
pub struct ExecutorControlSlave {
    task_receiver: crossbeam::channel::Receiver<Task>,
    event_sender: Sender<ExecutorEvent>,
}

#[allow(dead_code)]
impl ExecutorControlSlave {

    fn send_result(&self, result: TaskResult) -> Result<(), Box<dyn Error>> {
        self.event_sender.send(ExecutorEvent::TaskResult(result))?;
        Ok(())
    }

    fn send_err(&self, task_id: String, executor_id: String, error: ExecutorError) -> Result<(), Box<dyn Error>> {
        self.event_sender.send(ExecutorEvent::ExecutorError {
            task_id,
            executor_id,
            error
        })?;
        Ok(())
    }

    fn accept_task(&self, executor_id: String, task_id: String) -> Result<(), Box<dyn Error>> {
        self.event_sender.send(ExecutorEvent::TaskAccepted {
            executor_id,
            task_id
        })?;
        Ok(())
    }

    fn get_next_task(&self) -> Result<Task, Box<dyn Error>> {
        Ok(self.task_receiver.recv()?)
    }

}

impl TaskUpdateEmitter for ExecutorControlSlave {
    fn emit_update(&self, update: TaskUpdate) {
        if self.event_sender.send(ExecutorEvent::TaskUpdate(update)).is_err() {
            // TODO: log error... or propagate?
        }
    }
}