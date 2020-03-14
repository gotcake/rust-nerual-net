use std::sync::atomic::AtomicBool;
use std::sync::RwLock;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread;
use std::time::SystemTime;

use crate::net::Net;
use crate::data::PreparedDataSet;
use crate::func::CompletionFn;
use crate::func::MiniBatchSize;
use crate::func::LearningRateFn;
use crate::func::ErrorFn;
use crate::stats::Stats;
use crate::buffer::RowBuffer;

pub fn train_backprop_multi_threaded(
    net: &mut Net,
    data_set: &PreparedDataSet,
    completion_fn: CompletionFn,
    mini_batch_size_fn: MiniBatchSize,
    learning_rate_fn: LearningRateFn,
    error_fn: ErrorFn,
    batches_per_sync: usize,
    num_workers: usize,
    num_partitions: usize
) -> (Stats, usize) {

    let stage_start_time = SystemTime::now();

    // shared state
    let shared_state = Arc::new(RwLock::new(SharedThreadState {
        worker_done_counter: 0,
        weight_buffer: net.get_weights().clone(),
        next_partition_index: num_workers % num_partitions,
        partition_row_shifts: vec![0; num_partitions]
    }));

    // set up channel for worker threads to communicate to main thread
    let (check_error_sender, check_error_reciever) = mpsc::channel::<()>();
    let stage_complete_flag = Arc::new(AtomicBool::new(false));

    // create worker threads
    for worker_index in 0..num_workers {

        let shared_state = Arc::clone(&shared_state);
        let check_error_sender = check_error_sender.clone();
        let mut local_net = net.clone();
        //let training_set = training_set.clone();//partitioned_training_sets.pop().unwrap();
        let partitioned_data_sets = data_set.clone().partition(num_partitions);
        let stage_complete_flag = stage_complete_flag.clone();

        thread::spawn(move || {

            let mut start_weights = local_net.new_zeroed_weight_buffer();
            let mut weight_diffs = local_net.new_zeroed_weight_buffer();
            let mut context = local_net.get_training_context();

            let mut partition_index = worker_index;
            let mut partition_shift = 0;

            //let mut shift = 0;
            //let shift_steps = 5;

            loop {

                if stage_complete_flag.load(Ordering::Relaxed) {
                    return;
                }

                let mut batch_num = {
                    // sync weights with shared state
                    let shared_state = shared_state.read().unwrap();
                    shared_state.weight_buffer.copy_into(&mut start_weights);
                    start_weights.copy_into(context.get_net_mut().get_weights_mut());
                    shared_state.worker_done_counter * batches_per_sync / num_workers
                };

                let data_set = &partitioned_data_sets[partition_index];

                for _ in 0..batches_per_sync {

                    context.train_backprop_single_batch(
                        data_set,
                        learning_rate_fn.get_learning_rate(batch_num),
                        &error_fn,
                        mini_batch_size_fn.get_mini_batch_size(batch_num),
                    );

                    batch_num += 1;
                }

                if stage_complete_flag.load(Ordering::Relaxed) {
                    return;
                }

                //shift = (shift + 1) & shift_steps;

                // compute weight diff
                context.get_net().get_weights().copy_into(&mut weight_diffs);
                weight_diffs.subtract(&start_weights);

                {
                    let mut shared_state = shared_state.write().unwrap();

                    shared_state.weight_buffer.add_with_multiplier(&mut weight_diffs, 1.0 / num_partitions as f32);
                    //shared_state.weight_buffer.add(&mut weight_diffs);

                    shared_state.worker_done_counter += 1;

                    partition_index = shared_state.next_partition_index;
                    shared_state.next_partition_index = (partition_index + 1) % num_partitions;

                    // TODO: something wrong here, variable unused
                    partition_shift = shared_state.partition_row_shifts[partition_index];
                    shared_state.partition_row_shifts[partition_index] += 1;

                    if shared_state.worker_done_counter % num_partitions == 0 {
                        if check_error_sender.send(()).is_err() {
                            // hung up, quit
                            return;
                        }
                    }

                }

            }

        });
    }

    {

        let mut batch_num = 0;
        let mut context = net.get_training_context();

        loop {

            // consume all pending notifications / wait for a notification

            let mut sync_count = check_error_reciever.try_iter().count();
            if sync_count == 0 {
                check_error_reciever.recv().unwrap();
                sync_count = 1;
            }

            batch_num += sync_count * batches_per_sync;

            // load state
            {
                // TODO: use net weight buffer as state weight buffer to avoid this copy operation?
                let state = shared_state.read().unwrap();
                state.weight_buffer.copy_into(context.get_net_mut().get_weights_mut());
            };

            let error_stats = context.compute_error_for_batch(
                &data_set,
                &error_fn,
            );

            if completion_fn.should_stop_training(batch_num, stage_start_time, &error_stats) {
                // return and close the channel, signaling that we've completed training
                stage_complete_flag.store(true, Ordering::Relaxed);
                return (error_stats, batch_num)
            }

        }

        unreachable!();

    }

}

struct SharedThreadState {
    worker_done_counter: usize,
    weight_buffer: RowBuffer,
    next_partition_index: usize,
    partition_row_shifts: Vec<usize>
}
