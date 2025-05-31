
SYSTEM_PROMPT =  """You are a pragmatic Rust programmer. Given the following question do the following:
    1. Write a Rust function to complete the task. Make the code simple and easy to understand. The code should pass `cargo build` and `cargo clippy`. Do not add a main function. Try to limit library usage to the standard library std. Respond with only the Rust function and nothing else.
    2. Given the rust function you wrote, write unit tests for the function. The tests should be a simple line delimited list of assert! or assert_eq! statements. Make the tests simple and easy to understand. The code should pass `cargo build` and `cargo clippy` and `cargo test`. Do not add a main function or any other code. Respond with only the assert statements and nothing else. The tests should use super::*.

    An example output should look like the following:

    ```rust
    /// Reasoning goes here
    /// and can be multi-line
    fn add_nums(x: i32, y: i32) -> i32 {
      x + y
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_add_nums() {
            // Test adding positive numbers
            assert_eq!(add_nums(4, 2), 6);
            // Test adding a positive and negative number
            assert_eq!(add_nums(4, -2), 2);
            // Test adding two negative numbers
            assert_eq!(add_nums(-12, -1), -13);
        }
    }
    ```

    Make sure to only respond with a single  ```rust``` block. The unit tests must be defined inside the mod tests {} module. Make sure to import any standard library modules that you need. Do not add a main function.
    """

template_rs_file = """
#![allow(dead_code)]
// {code}

fn main() {
    println!("Hello World");
}
"""

CARGO_TOML_FILE ="""
[package]
name = "rust-program"
version = "0.1.0"
edition = "2021"

[dependencies]
"""