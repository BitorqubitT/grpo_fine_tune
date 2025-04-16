from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import datasets
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

x = GRPO_agent(model, tokenizer, SYSTEM_PROMPT, 3)
model = ""
env = env(cargo_toml_file, template_rs_file)
#memory = Memory(3, 1, device)

wandb.init(project = "llm finetune",
           name = f"experiment 9424",
           config = {
                    "gamma": 5,
                    }
            )

columns = ['question',
           'generated_code',
           'total_rewards',
           'test_block',
           'asserts'
]

test_table = wandb.Table(columns = columns)

memory = Memory(3, 1, device, (3, 4, 2))

"""
def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

 # Check how to optimise this
def get_logprobs(logits, prompt, prompt_length):
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    #TODO: With the input should I include the system prompt aswell?
    #TODO: Should the prompt just be the prompt or maybe tokenized?
    input_ids = prompt[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)
    # We remove the tokens from the prompt.
    per_token_logps = per_token_logps[:,prompt_length-1:]
    return per_token_logps

"""

def get_per_token_logps(logits, input_ids):
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


def get_logprobs(model, prompts, actions, prompt_lengths):
    with torch.no_grad():
        outputs = model(input_ids=prompts, return_dict=True)
        logits = outputs.logits[:, :-1, :]
        input_ids = prompts[:, 1:]

    logprobs = get_per_token_logps(logits, input_ids)

    # Trim to just the action portion (i.e., remove prompt logprobs)
    trimmed_logprobs = []
    for lp_row, pl in zip(logprobs, prompt_lengths):
        trimmed_logprobs.append(lp_row[pl-1:])  # pl-1 because we dropped the first token
    return torch.stack(trimmed_logprobs)


#TODO: Use the new get_logprobs function to get logprob
# It doesnt use more resources than calculating it right after the model call for the query.
# We giuve model, prompt, action, prompt length to retrieve it


# Should be epochs
for _ in range(1):
    for k, batch in enumerate(data_loader):
        if k == 3:
            break

        prompt = batch["rust_prompt"][0]
        print(k)
        
        #action, logits, generated_ids, prompt_length = x.get_action(prompt)
        action = x.get_action(prompt)
        
        batch_rewards = env.step(action)
        #memory.clear(logits.shape)
 
        # TODO: Split this into two functions, one for the rewards and one for the table rows.
        table_rows, total_rewards = process_batch_rewards(batch_rewards, prompt, action)

        for row in table_rows:
            test_table.add_data(*row)

        with torch.no_grad():
            
            logprobs = get_logprobs(logits, generated_ids, prompt_length)
            advantages = calc_advantages(total_rewards)

        #memory.update_values(i, prompt, logits[i], advantages)

        #x.optimise_network()
        # Copy to ref model

    wandb.log({"test_table": test_table})
    wandb.finish()