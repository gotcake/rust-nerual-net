use std::ops::AddAssign;
use std::ops::Mul;
use std::ops::SubAssign;
use num::traits::Float;

#[derive(Debug)]
pub struct RowBuffer<T: Copy> {
    buffer: Vec<T>,
    row_offsets_and_ends: Vec<(usize, usize)>,
}

#[allow(dead_code)]
impl<T> RowBuffer<T> where T: Copy {

    pub fn new_with_row_sizes(initial_value: T, row_sizes: &[usize]) -> Self {
        assert!(row_sizes.len() > 0);
        let total_size: usize = row_sizes.iter().sum();
        let buffer: Vec<T> = vec![initial_value; total_size];
        let mut row_offsets_and_ends: Vec<(usize, usize)> = Vec::with_capacity(row_sizes.len());
        let mut offset: usize = 0;
        for i in 0..row_sizes.len() {
            let end = offset + row_sizes[i];
            row_offsets_and_ends.push((offset, end));
            offset = end;
        }
        RowBuffer {
            buffer,
            row_offsets_and_ends
        }
    }

    #[inline]
    pub fn get_row(&self, row: usize) -> &[T] {
        let (offset, end) = self.row_offsets_and_ends[row];
        &self.buffer[offset..end]
    }

    #[inline]
    pub fn get_row_mut(&mut self, row: usize) -> &mut [T] {
        let (offset, end) = self.row_offsets_and_ends[row];
        &mut self.buffer[offset..end]
    }

    #[inline]
    pub fn split_rows(&mut self, row_first: usize, row_second: usize) -> (&mut [T], &mut [T]) {
        assert_ne!(row_first, row_second);
        let (offset_first, end_first) = self.row_offsets_and_ends[row_first];
        let (offset_second, end_second) = self.row_offsets_and_ends[row_second];
        let ptr = self.buffer.as_mut_ptr();
        unsafe {
            (core::slice::from_raw_parts_mut(ptr.add(offset_first), end_first - offset_first),
             core::slice::from_raw_parts_mut(ptr.add(offset_second), end_second - offset_second))
        }
    }

    #[inline]
    pub fn num_rows(&self) -> usize {
        self.row_offsets_and_ends.len()
    }

    #[inline]
    pub fn get_last_row(&self) -> &[T] {
        self.get_row(self.num_rows() - 1)
    }

    #[inline]
    pub fn get_last_row_mut(&mut self) -> &mut [T] {
        self.get_row_mut(self.num_rows() - 1)
    }

    #[inline]
    pub fn get_first_row(&self) -> &[T] {
        self.get_row(0)
    }

    #[inline]
    pub fn get_first_row_mut(&mut self) -> &mut [T] {
        self.get_row_mut(0)
    }

    pub fn reset_to(&mut self, value: T) {
        for item in self.buffer.iter_mut() {
            *item = value;
        }
    }

    pub fn copy_into(&self, target: &mut RowBuffer<T>) {
        assert_eq!(self.buffer.len(), target.buffer.len());
        for i in 0..self.buffer.len() {
            target.buffer[i] = self.buffer[i];
        }
    }

}

impl <T> RowBuffer<T> where T: Float + std::ops::AddAssign + std::ops::SubAssign {

    pub fn add(&mut self, other: &RowBuffer<T>) {
        assert_eq!(self.buffer.len(), other.buffer.len());
        for i in 0..self.buffer.len() {
            self.buffer[i] += other.buffer[i];
        }
    }
    pub fn add_with_multiplier(&mut self, other: &RowBuffer<T>, multiplier: T) {
        assert_eq!(self.buffer.len(), other.buffer.len());
        for i in 0..self.buffer.len() {
            self.buffer[i] += other.buffer[i] * multiplier;
        }
    }

    pub fn subtract(&mut self, subtract: &RowBuffer<T>) {
        assert_eq!(self.buffer.len(), subtract.buffer.len());
        for i in 0..self.buffer.len() {
            self.buffer[i] -= subtract.buffer[i];
        }
    }

}
