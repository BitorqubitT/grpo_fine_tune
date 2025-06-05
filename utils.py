import re
from typing import Optional
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import List
import os
from torch.cuda.amp import autocast

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
                    'test'
                    ]
    
    rows = []
    total_rewards = []

    for i, rewards in enumerate(batch_rewards):
        total_reward = sum(rewards[key] for key in rewards_keys)
        total_rewards.append(total_reward)
        
        row = [prompt,
               actions[i],
               total_reward,
               rewards['test block'],
               rewards['asserts'],
               ]
        rows.append(row)

    return rows, total_rewards

def get_logprobs(model, input_ids: torch.Tensor, actions: torch.Tensor, tokenizer, use_no_grad=True) -> torch.Tensor:
    """
    Compute logprobs for the generated actions. `input_ids` should be the full prompt + response tokens.
    `actions` should be padded to align with the response positions within input_ids, with -100 elsewhere.
    """
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    if use_no_grad:
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    else:
        with autocast(dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

    logits = outputs.logits  # [B, S, V]
    log_probs = F.log_softmax(logits, dim=-1)
    action_mask = actions != -100  # [B, S]

    # Replace -100 with 0 (or any valid token id) to avoid gather errors
    safe_actions = actions.clone()
    safe_actions[~action_mask] = 0  # safe index to gather
    
    selected = log_probs.gather(2, safe_actions.unsqueeze(-1)).squeeze(-1)  # [B, S]
    selected = selected * action_mask  # Zero out padding

    logprobs = selected.sum(dim=1)  # Sum over sequence
    return logprobs