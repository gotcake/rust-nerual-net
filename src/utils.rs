use std::u128;
use std::collections::HashSet;
use std::hash::Hash;
use std::slice;

#[inline(always)]
pub fn square_f32(n: f32) -> f32 {
    n * n
}


const SEED_XOR: u128 = 0b10101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010;
pub fn stable_hash_seed(s: &str) -> [u8; 16] {
    let mut val = 17u128;
    for byte in s.as_bytes() {
        val = 31u128.wrapping_mul(val).wrapping_add(*byte as u128);
        val ^= SEED_XOR;
    }
    unsafe {
        std::mem::transmute::<u128, [u8; 16]>(val)
    }
}

pub fn into_string_vec<T, I>(val: T) -> Vec<String> where T: AsRef<[I]>, I: ToString {
    val.as_ref()
        .iter()
        .map(ToString::to_string)
        .collect()
}

pub fn first_duplicate<'a, T, I>(iter: T) -> Option<&'a I> where T: Iterator<Item=&'a I>, I: Eq + Hash + 'a {
    let mut set = HashSet::<&'a I>::new();
    for item in iter {
        if !set.insert(item) {
            return Some(item);
        }
    }
    None
}


pub fn split_slice_mut<T>(slice: &mut [T], left: usize, right: usize) -> (&mut [T], &mut [T]) {
    assert_eq!(slice.len(), left + right);
    let ptr = slice.as_mut_ptr();
    unsafe {
        (slice::from_raw_parts_mut(ptr, left),
         slice::from_raw_parts_mut(ptr.add(left), right))
    }
}

pub fn split_slice<T>(slice: &[T], left: usize, right: usize) -> (&[T], &[T]) {
    assert_eq!(slice.len(), left + right);
    let ptr = slice.as_ptr();
    unsafe {
        (slice::from_raw_parts(ptr, left),
         slice::from_raw_parts(ptr.add(left), right))
    }
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_into_string_vec() {
        assert_eq!(into_string_vec(["foo", "bar"]), vec!["foo".to_string(), "bar".to_string()]);
        assert_eq!(into_string_vec(vec!["foo".to_string(), "bar".to_string()]), vec!["foo".to_string(), "bar".to_string()]);
    }

    #[test]
    fn test_first_duplicate() {
        assert_eq!(
            first_duplicate([1, 2, 3].iter()),
            None
        );
        assert_eq!(
            first_duplicate([1, 2, 3, 2].iter()),
            Some(&2)
        );
        assert_eq!(
            first_duplicate(["foo", "bar"].iter()),
            None
        );
        assert_eq!(
            first_duplicate(["foo", "bar", "foo"].iter()),
            Some(&"foo")
        );
        assert_eq!(
            first_duplicate(into_string_vec(["foo", "bar"]).iter()),
            None
        );
        let mut vec = into_string_vec(["foo", "bar"]);
        let mut s = "fo".to_string();
        s.push('o');
        vec.push(s);
        assert_eq!(
            first_duplicate(vec.iter()),
            Some(&"foo".to_string())
        );
    }

}