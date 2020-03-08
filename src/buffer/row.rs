#[derive(Debug, Clone)]
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
        for i in 0..self.buffer.len() {
            self.buffer[i] = value;
        }
    }

    pub fn copy_into(&self, target: &mut RowBuffer<T>) {
        assert_eq!(self.buffer.len(), target.buffer.len());
        for i in 0..self.buffer.len() {
            target.buffer[i] = self.buffer[i];
        }
    }

    #[inline]
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    #[inline]
    pub fn get_buffer(&self) -> &[T] {
        self.buffer.as_slice()
    }

    #[inline]
    pub fn get_buffer_mut(&mut self) -> &mut [T] {
        self.buffer.as_mut_slice()
    }

}

impl <T> RowBuffer<T> where T: Copy + std::ops::AddAssign {

    #[allow(dead_code)]
    pub fn add(&mut self, other: &RowBuffer<T>) {
        assert_eq!(self.buffer.len(), other.buffer.len());
        for i in 0..self.buffer.len() {
            self.buffer[i] += other.buffer[i];
        }
    }

}

impl <T> RowBuffer<T> where T: Copy + std::ops::AddAssign + std::ops::Mul<Output=T> {

    pub fn add_with_multiplier(&mut self, other: &RowBuffer<T>, multiplier: T) {
        assert_eq!(self.buffer.len(), other.buffer.len());
        for i in 0..self.buffer.len() {
            self.buffer[i] += other.buffer[i] * multiplier;
        }
    }

}

impl <T> RowBuffer<T> where T: Copy + std::ops::SubAssign {

    pub fn subtract(&mut self, subtract: &RowBuffer<T>) {
        assert_eq!(self.buffer.len(), subtract.buffer.len());
        for i in 0..self.buffer.len() {
            self.buffer[i] -= subtract.buffer[i];
        }
    }

}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[should_panic(expected = "assertion failed: row_sizes.len() > 0")]
    fn test_empty_not_allowed() {
        RowBuffer::new_with_row_sizes(0f32, &Vec::new());
    }

    #[test]
    fn test_basics() {

        let mut buf = RowBuffer::new_with_row_sizes(0usize, &vec![1, 0, 10, 2]);

        // check for correct structure
        assert_eq!(4, buf.num_rows());

        assert_eq!(1, buf.get_row(0).len());
        assert_eq!(1, buf.get_row_mut(0).len());
        assert_eq!(1, buf.get_first_row().len());
        assert_eq!(1, buf.get_first_row_mut().len());

        assert_eq!(0, buf.get_row(1).len());
        assert_eq!(0, buf.get_row_mut(1).len());

        assert_eq!(10, buf.get_row(2).len());
        assert_eq!(10, buf.get_row_mut(2).len());

        assert_eq!(2, buf.get_row(3).len());
        assert_eq!(2, buf.get_row_mut(3).len());
        assert_eq!(2, buf.get_last_row().len());
        assert_eq!(2, buf.get_last_row_mut().len());

        // populate data
        for i in 0..buf.num_rows() {
            let row = buf.get_row_mut(i);
            for j in 0..row.len() {
                row[j] = i * 10 + j;
            }
        }

        // check values
        for i in 0..buf.num_rows() {
            let row = buf.get_row(i);
            for j in 0..row.len() {
                assert_eq!(i * 10 + j, row[j]);
            }
        }

        let mut buf2 = buf.clone();
        buf.add(&buf2);

        // check values
        for i in 0..buf.num_rows() {
            let row = buf.get_row(i);
            for j in 0..row.len() {
                assert_eq!((i * 10 + j) * 2, row[j]);
            }
        }

        buf.subtract(&buf2);

        // check values
        for i in 0..buf.num_rows() {
            let row = buf.get_row(i);
            for j in 0..row.len() {
                assert_eq!(i * 10 + j, row[j]);
            }
        }

        buf.add_with_multiplier(&buf2, 2);

        // check values
        for i in 0..buf.num_rows() {
            let row = buf.get_row(i);
            for j in 0..row.len() {
                assert_eq!((i * 10 + j) * 3, row[j]);
            }
        }

        buf.copy_into(&mut buf2);

        // check values
        for i in 0..buf.num_rows() {
            let row = buf.get_row(i);
            for j in 0..row.len() {
                assert_eq!((i * 10 + j) * 3, row[j]);
            }
        }

    }

}
