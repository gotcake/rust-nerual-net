use std::path::Path;
use std::error::Error;
use std::boxed::Box;
use std::sync::Arc;
use std::fmt::Debug;
use crate::utils::{into_string_vec, first_duplicate};
use itertools::chain;

quick_error! {
    #[derive(Debug)]
    enum CsvParseError {
        ZeroColumns {
            description("Zero columns in dataset")
        }
        ZeroColumnsSelected {
            description("Zero columns selected")
        }
        ColumnCountMismatch(count: usize, previous: usize) {
            description("Invalid number of columns, did not match previous columns")
            display("Invalid number of columns {}, previous was {}", count, previous)
        }
        ColumnNotFound(name: String) {
            description("Column with specified name not found")
            display("Column with name {} not found", name)
        }
        DuplicateColumns(name: String) {
            description("Duplicate columns found in file")
            display("Duplicate columns found in file: {}", name)
        }
        DuplicateColumnsSpecified(name: String) {
            description("Duplicate columns specified")
            display("Duplicate columns specified: {}", name)
        }
    }
}

#[derive(Clone)]
pub struct PreparedDataSet {
    data: Arc<Box<[f32]>>,
    offset: usize,
    end: usize,
    num_cols: usize,
    num_rows: usize,
    dependent_cols: usize,
    independent_cols: usize
}

impl PreparedDataSet {

    pub fn from_csv<T1, I1, T2, I2>(
        path: impl AsRef<Path>,
        independent_cols: T1,
        dependent_cols: T2
    ) -> Result<PreparedDataSet, Box<dyn Error>>
        where T1: AsRef<[I1]>, I1: ToString,
              T2: AsRef<[I2]>, I2: ToString
    {

        let independent_cols = into_string_vec(independent_cols);
        let dependent_cols = into_string_vec(dependent_cols);

        if independent_cols.len() == 0 || dependent_cols.len() == 0 {
            return Err(Box::new(CsvParseError::ZeroColumnsSelected));
        }

        let mut reader = csv::ReaderBuilder::new()
            .trim(csv::Trim::All)
            .from_path(path)?;

        let column_names = reader.headers()?
            .iter()
            .map(str::to_owned)
            .collect::<Vec<String>>();

        let n_cols = column_names.len();

        if n_cols == 0 {
            return Err(Box::new(CsvParseError::ZeroColumns));
        }

        if let Some(dupe) = first_duplicate(column_names.iter()) {
            return Err(Box::new(CsvParseError::DuplicateColumns(dupe.clone())));
        }

        let mut independent_indices = Vec::with_capacity(independent_cols.len());
        let mut dependent_indices = Vec::with_capacity(dependent_cols.len());

        for col_name in independent_cols.iter() {
            match column_names.iter().position(|n| n == col_name) {
                None =>  return Err(Box::new(CsvParseError::ColumnNotFound(col_name.clone()))),
                Some(i) => independent_indices.push(i),
            }
        }

        for col_name in dependent_cols.iter() {
            match column_names.iter().position(|n| n == col_name) {
                None =>  return Err(Box::new(CsvParseError::ColumnNotFound(col_name.clone()))),
                Some(i) => dependent_indices.push(i),
            }
        }

        if let Some(dupe) = first_duplicate(chain(independent_cols.iter(), dependent_cols.iter())) {
            return Err(Box::new(CsvParseError::DuplicateColumnsSpecified(dupe.clone())));
        }

        let mut row_vals = Vec::with_capacity(n_cols);
        let mut num_rows = 0usize;
        let mut data = Vec::new();

        for row in reader.records() {
            row_vals.clear();
            for datum in row?.iter() {
                row_vals.push(datum.parse::<f32>()?);
            }
            if column_names.len() != row_vals.len() {
                return Err(Box::new(CsvParseError::ColumnCountMismatch(row_vals.len(), column_names.len())));
            }

            for &i in &independent_indices {
                data.push(row_vals[i]);
            }

            for &i in &dependent_indices {
                data.push(row_vals[i]);
            }

            num_rows += 1;
        }

        Ok(Self::from_vec(data, independent_cols.len(), dependent_cols.len(), num_rows))

    }

    fn from_vec(data: Vec<f32>, independent_cols: usize, dependent_cols: usize, num_rows: usize) -> Self {
        let num_cols = dependent_cols + independent_cols;
        assert_eq!(data.len(), num_rows * num_cols, "data length mismatch");
        PreparedDataSet {
            data: Arc::new(data.into_boxed_slice()),
            offset: 0,
            end: num_rows * num_cols,
            num_cols,
            num_rows,
            independent_cols,
            dependent_cols
        }
    }

    fn make_partition(&self, row_offset: usize, num_rows: usize) -> PreparedDataSet {
        let offset = self.offset + row_offset * self.num_cols;
        let end = offset + num_rows * self.num_cols;
        assert!(offset <= end && end <= self.end);
        PreparedDataSet {
            data: Arc::clone(&self.data),
            offset,
            end,
            num_cols: self.num_cols,
            num_rows,
            independent_cols: self.independent_cols,
            dependent_cols: self.dependent_cols
        }
    }

    pub fn partition(&self, n: usize) -> Vec<PreparedDataSet> {
        assert!(n > 0 && n < self.num_rows);
        let target_rows = self.num_rows / n;
        let mut vec = Vec::with_capacity(n);
        let mut end_row = self.num_rows;
        for _ in 0..n-1 {
            let row_offset = end_row - target_rows;
            vec.push(self.make_partition(
                row_offset,
                target_rows
            ));
            end_row = row_offset;
        }
        vec.push(self.make_partition(0, end_row));
        vec
    }

}

impl<'a> PreparedDataSet {

    pub fn iter(&'a self) -> PreparedDataSetIterator<'a> {
        PreparedDataSetIterator {
            data: self.data.as_ref(),
            offset: self.offset,
            end: self.end,
            num_cols: self.num_cols,
            independent_cols: self.independent_cols
        }
    }
}

pub struct PreparedDataSetIterator<'a> {
    data: &'a [f32],
    offset: usize,
    end: usize,
    num_cols: usize,
    independent_cols: usize,
}

impl<'a> PreparedDataSetIterator<'a> {

    #[inline]
    pub fn has_next(&self) -> bool {
        self.offset != self.end
    }

    pub fn next_unchecked(&mut self) -> (&'a [f32], &'a [f32]) {
        let offset = self.offset;
        let dependent_offset = offset + self.independent_cols;
        let row_end = offset + self.num_cols;
        self.offset = row_end;
        (&self.data[offset..dependent_offset], &self.data[dependent_offset..row_end])
    }

}

impl<'a> Iterator for PreparedDataSetIterator<'a> {
    type Item = (&'a [f32], &'a [f32]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.has_next() {
            Some(self.next_unchecked())
        } else {
            None
        }
    }

}

impl<'a> IntoIterator for &'a PreparedDataSet {
    type Item = (&'a [f32], &'a [f32]);
    type IntoIter = PreparedDataSetIterator<'a>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/*

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

*/


#[cfg(test)]
mod test {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_parse_csv() -> Result<(), Box<dyn Error>> {

        /*
        0_0, 0_1, 1_0, 1_1, has_horizontal, has_vertical
        1, 1,    0, 0,   1, 0
        1, 0,    1, 0,   0, 1
        0, 0,    0, 0,   0, 0
        1, 1,    0, 1,   1, 1
        0, 1,    0, 1,   0, 1
        0, 1,    0, 0,   0, 0
        1, 1,    1, 1,   1, 1
        1, 0,    0, 0,   0, 0
        0, 1,    0, 1,   0, 1
        1, 0,    1, 0,   0, 1
        0, 1,    1, 0,   0, 0
        */

        let data = PreparedDataSet::from_csv(
            "data/2x2_lines_binary.csv",
            ["1_0", "1_1", "0_0", "0_1"], // purposely out of order
            ["has_horizontal", "has_vertical"]
        )?;
        let expected: Vec<(&[f32], &[f32])> = vec![
            (&[0., 0., 1., 1.], &[1., 0.]),
            (&[1., 0., 1., 0.], &[0., 1.]),
            (&[0., 0., 0., 0.], &[0., 0.]),
            (&[0., 1., 1., 1.], &[1., 1.]),
            (&[0., 1., 0., 1.], &[0., 1.]),
            (&[0., 0., 0., 1.], &[0., 0.]),
            (&[1., 1., 1., 1.], &[1., 1.]),
            (&[0., 0., 1., 0.], &[0., 0.]),
            (&[0., 1., 0., 1.], &[0., 1.]),
            (&[1., 0., 1., 0.], &[0., 1.]),
            (&[1., 0., 0., 1.], &[0., 0.]),
        ];
        assert_eq!(data.iter().collect::<Vec<(&[f32], &[f32])>>(), expected);
        Ok(())
    }

    fn test_partition() {

        // TODO impl
        assert!(false);
    }

}


