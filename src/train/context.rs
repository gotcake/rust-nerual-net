use crate::data::ColumnSelection;

#[derive(Clone, Debug)]
pub struct TrainingContext {
    pub dependent_columns: ColumnSelection,
    pub independent_columns: ColumnSelection,
}