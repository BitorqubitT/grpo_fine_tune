import re
from typing import Optional
import torch
import numpy as np
import wandb
import torch.nn.functional as F
from transformers import AutoTokenizer

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

def calc_advantages(rewards:list) -> torch.Tensor:
    rewards = torch.tensor(rewards).to('cuda')
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False)
    
    if std_r < 1e-8:
        return torch.zeros_like(rewards)
    
    advantages = (rewards - mean_r) / std_r
    return advantages

def process_batch_rewards(batch_rewards, prompt, actions):
    """Process rewards in batches for better efficiency"""
    rewards_keys = ['not empty',
                'code block', 
                'test block', 
                'asserts', 
                'build', 
                'clippy', 
                'test']
    rows = []
    total_rewards = []

    for i, rewards in enumerate(batch_rewards):
        total_reward = sum(rewards[key] for key in rewards_keys)
        total_rewards.append(total_reward)
        
        # Create a single row with all required columns
        row = [
            prompt,                     # question
            actions[i],                 # generated_code
            total_reward,              # total_rewards
            rewards['test block'],      # test_block
            rewards['asserts'],         # asserts
        ]
        rows.append(row)

    return rows, total_rewards

# batched
def get_logprobs(model, prompts, actions, tokenizer, use_no_grad = True) -> torch.Tensor:
    """
    Compute logprobs for the generated actions, given prompts.
    Assumes prompts and actions are both [B, T] padded sequences.
    We still need the prompts, Because they are used for predicint the actions.
    """
    batch_input_ids = []
    prompt_lengths = []

    for prompt_ids, action_ids in zip(prompts, actions):
        input_ids = torch.cat([prompt_ids, action_ids], dim=0)
        prompt_lengths.append(len(prompt_ids))
        batch_input_ids.append(input_ids)

    batch_input_ids = torch.nn.utils.rnn.pad_sequence(
        batch_input_ids, 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    ).to("cuda")

    # Create attention mask: 1 for non-pad tokens, 0 for pad
    attention_mask = (batch_input_ids != tokenizer.pad_token_id).long()

    if use_no_grad:
        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=attention_mask, return_dict=True)
    else:
        outputs = model(batch_input_ids, attention_mask=attention_mask, return_dict=True)

    logits = outputs.logits  # [B, T, V]

    all_logprobs = []

    for i, (prompt_len, action_ids) in enumerate(zip(prompt_lengths, actions)):
        shift_logits = logits[i, prompt_len - 1:-1, :]
        shift_labels = action_ids

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = torch.gather(log_probs, 1, shift_labels.unsqueeze(-1)).squeeze(-1)
        total_logprob = token_logprobs.sum()
        all_logprobs.append(total_logprob)

    return torch.stack(all_logprobs, dim=0)
