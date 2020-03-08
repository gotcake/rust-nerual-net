use std::time::SystemTime;

use crate::{
    data::ColumnSelection,
};
use crate::func::CompletionFn;
use crate::stats::Stats;

#[derive(Clone, Debug)]
pub struct TrainingContext {
    pub dependent_columns: ColumnSelection,
    pub independent_columns: ColumnSelection,
}