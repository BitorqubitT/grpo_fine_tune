import re
from typing import Optional

rustcode = '''```rust
fn sort_list(mut list: Vec<i32>) -> Vec<i32> {
    list.sort();
    list
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_list() {
        let unsorted = vec![5, 3, 8, 1, 2];
        let sorted = sort_list(unsorted.clone());
        assert_eq!(sorted, vec![1, 2, 3, 5, 8]);
        assert_eq!(sorted, vec![1, 2, 3, 5, 8]);
        assert_eq!(sorted, vec![1, 2, a3, 5, 8]);
        assert_eq!(sorted, vec![1, 2, a3, 5, 8]);
        assert_eq!(sorted, vec![1, 2123, a3, 5, 8]);
    }
}
```'''

def extract_rust_code(text: str) -> Optional[str]:
    pattern = r'```rust\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None

# How useful is this? To combat reward hacking?
# Maybe just check if non empty
def check_code_not_empty(code: str) -> bool:
    if len(code) > 10:
        return True
    return False

def check_code_block(code: str) -> bool:
    if extract_rust_code(code):
        return True
    return False

def check_test_block(code: str) -> bool:
    pattern = r'(#\[cfg\(test\)\]\s*mod\s+tests\s*\{.*?\})'
    match = re.search(pattern, code, re.DOTALL)
    if match:
        return True
    return False

def response_contains_asserts(code: str) -> float:
    pattern = r'#\[cfg\(test\)\]\s*mod\s+tests\s*\{([^}]*)\}'
    match = re.search(pattern, code, re.DOTALL)

    if not match:
        return 0.0
    
    test_block = match.group(0)

    # Find all assert statements
    assert_pattern = r'assert(?:_eq)?\!(.*?);'
    all_asserts = re.findall(assert_pattern, test_block)
    total_asserts = len(all_asserts)
    
    if total_asserts == 0:
        return 0.0
        
    # Store unique assert statements
    unique_asserts = set(assert_stmt.strip() for assert_stmt in all_asserts)
    
    return len(unique_asserts) / total_asserts

# code running rewards and output rewards
# This should all be in the environment
def get_rewards(code: str):
    total_reward = {"not empty": 0, "code block": 0, "test block": 0, "asserts": 0}
    if check_code_not_empty(code):
        total_reward["not empty"] = 1
    if check_code_block(code):
        total_reward["code block"] = 1
    if check_test_block(code):
        total_reward["test block"] = 1
    total_reward["asserts"] = response_contains_asserts(code)
    return total_reward
