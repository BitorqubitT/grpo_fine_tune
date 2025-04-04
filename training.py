from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import datasets
from grpo_agent import GRPO_agent
from grpo_agent import Memory
from utils import get_rewards
from env import env
import wandb
import pandas as pd
import numpy as np

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
env = env(cargo_toml_file, template_rs_file)
memory = Memory(3, 1, device)


wandb.init(project = "llm finetune",
           name = f"experiment 9424",
           config = {
                    "gamma": 5,
                    }
            )

columns = ['not empty', 
           'code block',
           'test block', 
           'asserts', 
           'build', 
           'clippy', 
           'test', 
           'test_time', 
           'result_output', 
           'question', 
           'generated_code'
           ]

test_table = wandb.Table(columns = columns)

# Should be epochs
for _ in range(10):
    for k, batch in enumerate(data_loader):
        if k == 200:
            break

        prompt = batch["rust_prompt"][0]

        # Retrieve multiple observations
        action = x.get_action(prompt)

        # Send observations to the environment to get rewards
        batch_rewards = env.step(action)

        all_rewards = []
        for i, rewards in enumerate(batch_rewards):
            print(i)
            rewards.update({
                'question': prompt,
                'generated_code': action[i]
            })

            # Log rewards
            test_table.add_data(*rewards.values())
            rewards_keys = ['not empty', 'code block', 'test block', 'asserts', 'build', 'clippy', 'test']
            total_reward = sum(rewards[key] for key in rewards_keys)
            all_rewards.append(total_reward)
            wandb.log({"total_reward": total_reward, 
                        "test_block": rewards["test block"], 
                        "asserts": rewards["asserts"]}
                        )

            # TODO: should we send the tokens to action?
            memory.update_values(i, prompt, action[i], None, total_reward)

        print(memory.get_values())
        #print(all_rewards)
        # calculate advantages
        # r - mean(r)/std(r)
        advantages = (all_rewards - np.mean(all_rewards)) / np.std(all_rewards)
        #print(advantages)

        # Use advantage and kl to compute the loss (for this we need logprobs)

        # Copy new policy model 


        # Update network
        #x.optimise_network()

        # Every x steps update the frozen model


        # send to weights and biases
        # q, actions
    wandb.log({"test_table": test_table})
    wandb.finish()