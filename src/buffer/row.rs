use std::ptr;
use std::slice;
use std::fmt;

#[derive(Clone)]
pub struct RowBuffer {
    buffer: Box<[f32]>,
    row_offsets_and_sizes: Box<[(usize, usize)]>,
}

impl fmt::Debug for RowBuffer {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let mut s = &mut f.debug_struct("RowBuffer");
        if self.buffer.len() < 30 {
            s = s.field("buffer", &self.buffer);
        } else {
            s = s.field("buffer_len", &self.buffer.len());
        }
        s.field("row_offsets_and_sizes", &self.row_offsets_and_sizes)
            .finish()
    }
}

#[allow(dead_code)]
impl RowBuffer {

    pub fn new_with_row_sizes(initial_value: f32, row_sizes: impl AsRef<[usize]>) -> Self {
        let row_sizes = row_sizes.as_ref();
        assert!(row_sizes.len() > 0);
        let total_size: usize = row_sizes.iter().sum();
        let buffer: Vec<f32> = vec![initial_value; total_size];
        let mut row_offsets_and_sizes: Vec<(usize, usize)> = Vec::with_capacity(row_sizes.len());
        let mut offset: usize = 0;
        for i in 0..row_sizes.len() {
            row_offsets_and_sizes.push((offset, row_sizes[i]));
            offset +=row_sizes[i];
        }
        RowBuffer {
            buffer: buffer.into_boxed_slice(),
            row_offsets_and_sizes: row_offsets_and_sizes.into_boxed_slice()
        }
    }

    #[inline]
    pub fn get_row(&self, row: usize) -> &[f32] {
        let (offset, size) = self.row_offsets_and_sizes[row];
        //&self.buffer[offset...offset+size]
        unsafe { slice::from_raw_parts(self.buffer.as_ptr().add(offset), size) }
    }

    #[inline]
    pub fn get_row_mut(&mut self, row: usize) -> &mut [f32] {
        let (offset, size) = self.row_offsets_and_sizes[row];
        unsafe { slice::from_raw_parts_mut(self.buffer.as_mut_ptr().add(offset), size) }
    }

    #[inline]
    pub fn split_rows(&mut self, row_first: usize, row_second: usize) -> (&mut [f32], &mut [f32]) {
        assert_ne!(row_first, row_second);
        let (offset_first, size_first) = self.row_offsets_and_sizes[row_first];
        let (offset_second, size_second) = self.row_offsets_and_sizes[row_second];
        let ptr = self.buffer.as_mut_ptr();
        unsafe {
            (slice::from_raw_parts_mut(ptr.add(offset_first), size_first),
             slice::from_raw_parts_mut(ptr.add(offset_second), size_second))
        }
    }

    #[inline]
    pub fn num_rows(&self) -> usize {
        self.row_offsets_and_sizes.len()
    }

    #[inline]
    pub fn get_last_row(&self) -> &[f32] {
        unsafe {
            let (offset, size) = *(self.row_offsets_and_sizes.as_ptr()
                .add(self.row_offsets_and_sizes.len() - 1));
            slice::from_raw_parts(self.buffer.as_ptr().add(offset), size)
        }
    }

    #[inline]
    pub fn get_last_row_mut(&mut self) -> &mut [f32] {
        unsafe {
            let (offset, size) = *(self.row_offsets_and_sizes.as_ptr()
                .add(self.row_offsets_and_sizes.len() - 1));
            slice::from_raw_parts_mut(self.buffer.as_mut_ptr().add(offset), size)
        }
    }

    #[inline]
    pub fn get_first_row(&self) -> &[f32] {
        unsafe {
            let (_offset, size) = *self.row_offsets_and_sizes.as_ptr();
            slice::from_raw_parts(self.buffer.as_ptr(), size)
        }
    }

    #[inline]
    pub fn get_first_row_mut(&mut self) -> &mut [f32] {
        unsafe {
            let (_offset, size) = *self.row_offsets_and_sizes.as_ptr();
            slice::from_raw_parts_mut(self.buffer.as_mut_ptr(), size)
        }
    }

    pub fn reset_to(&mut self, value: f32) {
        let mut ptr = self.buffer.as_mut_ptr();
        unsafe {
            let end = ptr.add(self.buffer.len());
            while ptr < end {
                *ptr = value;
                ptr = ptr.add(1);
            }
        }
    }

    pub fn copy_into(&self, target: &mut RowBuffer) {
        let size = self.buffer.len();
        assert_eq!(size, target.buffer.len());
        unsafe {
            ptr::copy_nonoverlapping(self.buffer.as_ptr(), target.buffer.as_mut_ptr(), size);
        }
    }

    #[inline]
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    #[inline]
    pub fn get_buffer(&self) -> &[f32] {
        &self.buffer
    }

    #[inline]
    pub fn get_buffer_mut(&mut self) -> &mut [f32] {
        &mut self.buffer
    }

    #[allow(dead_code)]
    pub fn add(&mut self, other: &RowBuffer) {
        let size = self.buffer.len();
        assert_eq!(size, other.buffer.len());
        let mut ptr_self = self.buffer.as_mut_ptr();
        let mut ptr_other = other.buffer.as_ptr();
        unsafe {
            let end = ptr_self.add(size);
            while ptr_self < end {
                *ptr_self += *ptr_other;
                ptr_self = ptr_self.add(1);
                ptr_other = ptr_other.add(1);
            }
        }
    }

    pub fn add_with_multiplier(&mut self, other: &RowBuffer, multiplier: f32) {
        let size = self.buffer.len();
        assert_eq!(size, other.buffer.len());
        let mut ptr_self = self.buffer.as_mut_ptr();
        let mut ptr_other = other.buffer.as_ptr();
        unsafe {
            let end = ptr_self.add(size);
            while ptr_self < end {
                *ptr_self += *ptr_other * multiplier;
                ptr_self = ptr_self.add(1);
                ptr_other = ptr_other.add(1);
            }
        }
    }

    pub fn subtract(&mut self, subtract: &RowBuffer) {
        let size = self.buffer.len();
        assert_eq!(size, subtract.buffer.len());
        let mut ptr_self = self.buffer.as_mut_ptr();
        let mut ptr_other = subtract.buffer.as_ptr();
        unsafe {
            let end = ptr_self.add(size);
            while ptr_self < end {
                *ptr_self -= *ptr_other;
                ptr_self = ptr_self.add(1);
                ptr_other = ptr_other.add(1);
            }
        }
    }

}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_row_sizes_types() {
        RowBuffer::new_with_row_sizes(0f32, vec![1, 2, 3]);
        RowBuffer::new_with_row_sizes(0f32, &vec![1, 2, 3]);
        RowBuffer::new_with_row_sizes(0f32, [1, 2, 3]);
        RowBuffer::new_with_row_sizes(0f32, &[1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "assertion failed: row_sizes.len() > 0")]
    fn test_empty_not_allowed() {
        RowBuffer::new_with_row_sizes(0f32, Vec::new());
    }

    #[test]
    fn test_basics() {

        let mut buf = RowBuffer::new_with_row_sizes(0.0, vec![1, 0, 10, 2]);

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
                row[j] = (i * 10 + j) as f32;
            }
        }

        // check values
        for i in 0..buf.num_rows() {
            let row = buf.get_row(i);
            for j in 0..row.len() {
                assert_eq!(row[j], (i * 10 + j) as f32);
            }
        }

        let mut buf2 = buf.clone();
        buf.add(&buf2);

        // check values
        for i in 0..buf.num_rows() {
            let row = buf.get_row(i);
            for j in 0..row.len() {
                assert_eq!(row[j], ((i * 10 + j) * 2) as f32);
            }
        }

        buf.subtract(&buf2);

        // check values
        for i in 0..buf.num_rows() {
            let row = buf.get_row(i);
            for j in 0..row.len() {
                assert_eq!(row[j], (i * 10 + j) as f32);
            }
        }

        buf.add_with_multiplier(&buf2, 2.0);

        // check values
        for i in 0..buf.num_rows() {
            let row = buf.get_row(i);
            for j in 0..row.len() {
                assert_eq!(row[j], ((i * 10 + j) * 3) as f32);
            }
        }

        buf.copy_into(&mut buf2);

        // check values
        for i in 0..buf.num_rows() {
            let row = buf.get_row(i);
            for j in 0..row.len() {
                assert_eq!(row[j], ((i * 10 + j) * 3) as f32);
            }
        }

    }

    #[test]
    fn test_get_first_last_rows() {
        let mut buf = RowBuffer::new_with_row_sizes(0.0, vec![15, 0, 8]);
        for i in 0..buf.buffer_len() {
            buf.get_buffer_mut()[i] = i as f32;
        }
        let first = buf.get_first_row();
        let last = buf.get_last_row();
        assert_eq!(first.len(), 15);
        assert_eq!(last.len(), 8);
        assert_eq!(first, &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.]);
        assert_eq!(last, &[15., 16., 17., 18., 19., 20., 21., 22.]);
    }

}
