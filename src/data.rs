use std::path::Path;
use std::error::Error;
use std::boxed::Box;
use std::collections::HashSet;
use std::sync::Arc;
use std::fmt::{Debug, Formatter};

#[derive(Debug, Clone)]
pub enum ColumnSelection {
    Range(usize, usize),
    Sparse(Vec<usize>)
}

quick_error! {
    #[derive(Debug)]
    pub enum ColumnSelectionError {
        ColumnsNotNamed {
            description("Columns not named")
        }
        ColumnNotFound(name: String) {
            description("Column with specified name not found")
            display("Column with name {} not found", name)
        }
        InvalidCount(count: usize, available: usize) {
            description("Invalid column selection count")
            display("Invalid column selection count: {}, available: {}", count, available)
        }
        DuplicateColumns(name: String) {
            description("Duplicate columns specified")
            display("Duplicate columns specified: {}", name)
        }
    }
}

#[derive(Clone)]
pub struct TrainingSet {
    data: Arc<Vec<f32>>,
    offset: usize,
    end: usize,
    num_cols: usize,
    num_rows: usize,
    column_names: Option<Vec<String>>
}

impl Debug for TrainingSet {
    fn fmt(&self, f: &mut Formatter) -> Result<(), core::fmt::Error> {
        f.debug_struct("TrainingSet")
            .field("data", &format!("<len = {}>", self.data.len()))
            .field("offset", &self.offset)
            .field("end", &self.end)
            .field("num_cols", &self.num_cols)
            .field("num_rows", &self.num_rows)
            .field("column_names", &self.column_names)
            .finish()
    }
}

quick_error! {
    #[derive(Debug)]
    enum CsvParseError {
        InvalidColumnCount(count: usize) {
            description("Invalid number of columns")
            display("Invalid number of columns {}", count)
        }
        ColumnCountMismatch(count: usize, previous: usize) {
            description("Invalid number of columns, did not match previous columns")
            display("Invalid number of columns {}, previous was {}", count, previous)
        }
    }
}

impl TrainingSet {
    #[allow(dead_code)]
    pub fn new_from_vec(num_cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len() % num_cols, 0);
        let len = data.len();
        TrainingSet {
            data: Arc::new(data),
            offset: 0,
            end: len,
            num_cols,
            num_rows: len / num_cols,
            column_names: None
        }
    }
    #[inline(always)]
    pub fn get_data_slice(&self) -> &[f32] {
        &self.data[self.offset..self.end]
    }
    pub fn new_from_csv(path: impl AsRef<Path>) -> Result<TrainingSet, Box<dyn Error>> {
        let mut reader = csv::ReaderBuilder::new()
            .trim(csv::Trim::All)
            .from_path(path)?;

        let column_names = reader.headers()?
            .iter()
            .map(|str| str.to_string())
            .collect::<Vec<String>>();

        let mut data: Vec<f32> = Vec::new();
        let mut num_cols: usize = 0;
        let mut num_rows: usize = 0;
        for row in reader.records() {
            let mut cols: usize = 0;
            for datum in row?.iter() {
                let value = datum.parse::<f32>()?;
                data.push(value);
                cols += 1;
            }
            if num_cols == 0 {
                if cols == 0 {
                    return Err(Box::new(CsvParseError::InvalidColumnCount(cols)));
                } else {
                    num_cols = cols;
                }
            } else if num_cols != cols {
                return Err(Box::new(CsvParseError::ColumnCountMismatch(cols, num_cols)));
            }
            num_rows += 1;
        }

        let len = data.len();
        Ok(TrainingSet{
            data: Arc::new(data),
            offset: 0,
            end: len,
            num_rows,
            num_cols,
            column_names: Some(column_names)
        })
    }
    pub fn iter(&self) -> TrainingSetIterator {
        TrainingSetIterator {
            data: self.get_data_slice(),
            started: false,
            offset: 0,
            row_step: self.num_cols,
            num_cols: self.num_cols
        }
    }

    #[allow(dead_code)]
    pub fn get_column_names(&self) -> Option<Vec<String>> {
        match &self.column_names {
            None => None,
            Some(column_names) => Some(column_names.clone())
        }
    }

    #[allow(dead_code)]
    pub fn get_num_rows(&self) -> usize {
        self.num_rows
    }

    #[allow(dead_code)]
    pub fn get_num_cols(&self) -> usize {
        self.num_cols
    }

    pub fn get_named_column_selection(&self, names: &Vec<&str>) -> Result<ColumnSelection, ColumnSelectionError> {
        if names.len() > self.num_cols || names.is_empty() {
            return Err(ColumnSelectionError::InvalidCount(names.len(), self.num_cols))
        }
        match &self.column_names {
            None => Err(ColumnSelectionError::ColumnsNotNamed),
            Some(column_names) => {
                let mut seen_indexes = HashSet::new();
                let mut indexes: Vec<usize> = Vec::with_capacity(column_names.len());
                for target_name in names {
                    let index = column_names.iter()
                        .position(|name| *name.as_str() == **target_name)
                        .ok_or_else(|| ColumnSelectionError::ColumnNotFound(target_name.to_string()))?;
                    if seen_indexes.contains(&index) {
                        return Err(ColumnSelectionError::DuplicateColumns(target_name.to_string()));
                    }
                    seen_indexes.insert(index);
                    indexes.push(index);
                }

                // check to see if the selection is a range
                let is_range = indexes.iter()
                    .try_fold((false, 0), |(inited, last_index), index| {
                        if !inited {
                            Ok((true, *index))
                        } else if last_index + 1 == *index {
                            Ok((true, *index))
                        } else {
                            Err(())
                        }
                    }).is_ok();

                // if it's a range return the optimized column selection
                if is_range {
                    Ok(ColumnSelection::Range(*indexes.first().unwrap(), *indexes.last().unwrap() + 1))
                } else {
                    Ok(ColumnSelection::Sparse(indexes))
                }
            }
        }
    }

    pub fn partition(&self, n: usize) -> Vec<TrainingSet> {
        assert!(n > 0 && n < self.num_rows);
        let target_rows = self.num_rows / n;
        let target_size = target_rows * self.num_cols;
        let mut vec = Vec::with_capacity(n);
        let mut end = self.end;
        for _i in 0..n-1 {
            let start = end - target_size;
            vec.push(TrainingSet {
                data: Arc::clone(&self.data),
                offset: start,
                end,
                num_cols: self.num_cols,
                num_rows: target_rows,
                column_names: self.column_names.clone()
            });
            end = start;
        }
        assert_eq!(end % self.num_cols, 0);
        vec.push(TrainingSet {
            data: Arc::clone(&self.data),
            offset: self.offset,
            end,
            num_cols: self.num_cols,
            num_rows: end / self.num_cols,
            column_names: self.column_names.clone()
        });
        vec
    }

    #[allow(dead_code)]
    pub fn iter_shift_partition(
        &self, num_partitions: usize,
        shift: usize,
        shift_steps: usize,
        partition_index: usize
    ) -> (TrainingSetIterator, Option<TrainingSetIterator>) {
        assert!(num_partitions > 0 && partition_index < num_partitions && shift_steps > 0);
        let rows_per_partition = self.num_rows / num_partitions;
        let rows_per_shift = usize::max(rows_per_partition / shift_steps, 1);
        let shift_size = ((shift * rows_per_shift) % rows_per_partition) * self.num_cols;
        let size_per_partition = rows_per_partition * self.num_cols;
        let offset = size_per_partition * partition_index + shift_size;
        if shift_size > 0 && partition_index == num_partitions - 1 {
            (
                TrainingSetIterator{
                    data: &self.get_data_slice()[offset..self.end],
                    started: false,
                    offset: 0,
                    row_step: self.num_cols,
                    num_cols: self.num_cols
                },
                Some(TrainingSetIterator{
                    data: &self.get_data_slice()[0..shift_size],
                    started: false,
                    offset: 0,
                    row_step: self.num_cols,
                    num_cols: self.num_cols
                })
            )
        } else {
            (
                TrainingSetIterator{
                    data: &self.get_data_slice()[offset..offset + size_per_partition],
                    started: false,
                    offset: 0,
                    row_step: self.num_cols,
                    num_cols: self.num_cols
                },
                None
            )
        }

    }

}

pub struct TrainingSetIterator<'a> {
    data: &'a [f32],
    started: bool,
    offset: usize,
    row_step: usize,
    num_cols: usize,
}

impl TrainingSetIterator<'_> {

    #[inline]
    pub fn next(&mut self) -> bool {
        if self.started {
            self.offset += self.row_step;
        } else {
            self.started = true;
        }
        self.offset + self.num_cols <= self.data.len()
    }

    #[inline]
    pub fn has_next(&mut self) -> bool {
        self.offset + self.num_cols <= self.data.len()
    }

    #[inline]
    pub fn get_columns(&self, cols: &ColumnSelection, buffer: &mut [f32]) {
        let row = &self.data[self.offset..self.offset + self.num_cols];
        match cols {
            &ColumnSelection::Range(start, end) => {
                debug_assert!(end <= self.num_cols);
                for i in start..end {
                    buffer[i - start] = row[i];
                }
            },
            ColumnSelection::Sparse(cols) => {
                for (i, offset) in cols.iter().enumerate() {
                    debug_assert!(*offset < self.num_cols);
                    buffer[i] = row[*offset];
                }
            }
        }

    }

}


