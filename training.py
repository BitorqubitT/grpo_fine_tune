from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import datasets
import torch.nn.functional as F
from grpo_agent import GRPO_agent
from grpo_agent import Memory
from utils import get_rewards, calc_advantages, process_batch_rewards
from env import env
import wandb
import pandas as pd
import numpy as np
import torch

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

#dataset = datasets.load_dataset("TIGER-Lab/AceCode-87K", split='train')
df = pd.read_parquet("data/cargo_test_passed_train.parquet")
dataset = datasets.Dataset.from_pandas(df)

data_loader = DataLoader(dataset,
                         batch_size = 1,
                         shuffle = True
                        )

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

cargo_toml_file ="""
[package]
name = "rust-program"
version = "0.1.0"
edition = "2021"

[dependencies]
"""


#wandb.init(project = "llm finetune",
#           name = f"experiment 9424",
#           config = {
#                    "gamma": 5,
#                    }
#            )

columns = ['question',
           'generated_code',
           'total_rewards',
           'test_block',
           'asserts'
]

test_table = wandb.Table(columns = columns)
memory = Memory(3, 1, device, (3, 4, 2))
x = GRPO_agent(model, tokenizer, SYSTEM_PROMPT, 3, memory)
env = env(cargo_toml_file, template_rs_file)

def get_logprobs(model, prompts, actions) -> torch.Tensor:
    """
    Compute logprobs for the generated actions, given prompts.
    Assumes prompts and actions are both [B, T] padded sequences.
    We still need the prompts, Because they are used for predicint the actions.
    """
    all_logprobs = []

    for prompt_ids, action_ids in zip(prompts, actions):
        # Combine prompt + action
        input_ids = torch.cat([prompt_ids, action_ids], dim=0).unsqueeze(0).to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, return_dict=True)
            logits = outputs.logits  # [1, seq_len, vocab]

        # We want logprobs over the actions only
        response_start = prompt_ids.shape[0]
        shift_logits = logits[:, response_start - 1:-1, :]  # Predict action tokens
        shift_labels = action_ids.unsqueeze(0)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, action_len]

        total_logprob = token_logprobs.sum(dim=-1)  # [1]
        all_logprobs.append(total_logprob.item())
    return torch.Tensor(all_logprobs)

#TODO: Use the new get_logprobs function to get logprob
# It doesnt use more resources than calculating it right after the model call for the query.
# We giuve model, prompt, action, prompt length to retrieve it

# Should be epochs
for _ in range(1):
    for k, batch in enumerate(data_loader):
        if k == 3:
            break

        prompt = batch["rust_prompt"][0]
        action, prompt_id, generated_full_ids, generated_ids = x.get_action(prompt)
        
        batch_rewards = env.step(action)
 
        table_rows, total_rewards = process_batch_rewards(batch_rewards, prompt, action)

        #wandb.log({
        #"total_reward": np.mean(total_rewards),
        #})
     
        for row in table_rows:
            test_table.add_data(*row)

        with torch.no_grad():
            logprobs = get_logprobs(model, prompt_id, generated_ids)
            advantages = calc_advantages(total_rewards)
        
        memory.update_values(prompt_id, generated_ids, logprobs, advantages)
        x.optimise_network()
        # Copy to ref model

    #wandb.log({"test_table": test_table})
    #wandb.finish() 