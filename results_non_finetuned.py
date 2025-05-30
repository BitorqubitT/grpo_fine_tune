from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import datasets
from utils import get_rewards, calc_advantages, process_batch_rewards
from env import env
import wandb
import pandas as pd

# This script is used to run the non-finetuned models on the AceCode dataset.
# This is used as a baseline to compare against the finetuned models.

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

def get_answer(prompt, chat_template, tokenizer, model, amount=5, device="cuda"):
    messages = [
        {"role": "system", "content": chat_template},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text] * amount, return_tensors="pt", padding=True).to(device)

    generated_full_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens = 512,
        num_return_sequences = amount
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_full_ids)
    ]

    answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    return answers

if __name__ == "__main__":

    device = "cuda" # the device to load the model onto

    model_name = "Qwen/Qwen2-1.5B-Instruct"
    #model_name = "Qwen/Qwen3-1.7B"
    #model_name = "Qwen/Qwen3-0.6b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
    )

    #dataset = datasets.load_dataset("TIGER-Lab/AceCode-87K", split='train')
    df = pd.read_parquet("data/cargo_test_passed_train.parquet")
    dataset = datasets.Dataset.from_pandas(df)

    data_loader = DataLoader(dataset,
                            batch_size = 1,
                            shuffle = True
                            )

    env = env(cargo_toml_file, template_rs_file)
    print(" test")
    for k, batch in enumerate(data_loader):

        prompt = batch['rust_prompt'][0]
        #print("prompt:", prompt)

        action = get_answer(prompt, SYSTEM_PROMPT, tokenizer, model, amount=5, device=device)

        batch_rewards = env.step(action)
        table_rows, total_rewards = process_batch_rewards(batch_rewards, prompt, action)
        print(total_rewards)
        
        #TODO: Check if the reward structure that I am using is correct.
        