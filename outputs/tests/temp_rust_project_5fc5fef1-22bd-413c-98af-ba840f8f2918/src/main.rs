
#![allow(dead_code)]

use std::cmp;

/// Finds the maximum sum of a contiguous subarray in a given vector of integers.
///
/// # Examples
///
/// ```
/// assert_eq!(find_max_subarray_sum(vec![34, -50, 42, 14, -5, 86]), 137);
/// assert_eq!(find_max_subarray_sum(vec![-5, -1, -8, -9]), -1);
/// assert_eq!(find_max_subarray_sum(vec![]), 0);
/// ```
fn find_max_subarray_sum(arr: Vec<i32>) -> i32 {
    if arr.is_empty() {
        return 0;
    }

    let mut max_so_far = arr[0];
    let mut max_ending_here = arr[0];

    for &num in arr.iter().skip(1) {
        max_ending_here = cmp::max(num, max_ending_here + num);
        max_so_far = cmp::max(max_so_far, max_ending_here);
    }

    max_so_far
}

// Unit tests for the function
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_max_subarray_sum() {
        assert_eq!(find_max_subarray_sum(vec![34, -50, 42, 14, -5, 86]), 137);
        assert_eq!(find_max_subarray_sum(vec![-5, -1, -8, -9]), -1);
        assert_eq!(find_max_subarray_sum(vec![]), 0);
    }
}


fn main() {
    println!("Hello World");
}
