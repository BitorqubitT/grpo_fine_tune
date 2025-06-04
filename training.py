from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import datasets
import torch.nn.functional as F
from grpo_agent import GRPO_agent
from memory import Memory
from utils import get_rewards, calc_advantages, process_batch_rewards
from env import env
import wandb
import pandas as pd
import numpy as np
import torch
from utils import get_logprobs
import os
from templates import SYSTEM_PROMPT, template_rs_file, CARGO_TOML_FILE

device = "cuda"
#model_name = "Qwen/Qwen3-0.6b"
#model_name = "Qwen/Qwen3-1.8b"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, extra_vocab_file="qwen_extra.tiktoken")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16).to(device)

# Models are misaligned on purpose.
#dataset = datasets.load_dataset("TIGER-Lab/AceCode-87K", split='train')
df = pd.read_parquet("data/cargo_test_passed_train.parquet")
dataset = datasets.Dataset.from_pandas(df)

data_loader = DataLoader(dataset,
                         batch_size = 2,
                         shuffle = True
                        )

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
memory = Memory(tokenizer, device)
grpo_agent = GRPO_agent(model, tokenizer, SYSTEM_PROMPT, 3, memory)
env = env(CARGO_TOML_FILE, template_rs_file)

for k, batch in enumerate(data_loader):
    memory.clear()

    if k == 30:
        break

    # TODO: Give normal ids to prompts so we can reuse or search them later.

    # Get a dictionary with, task id, rust_prompt.
    batch_prompts = batch["rust_prompt"]

    for prompt in batch_prompts:
        # str answer, prompt id, prompt+answerids, answer_ids
        action, prompt_id, generated_full_ids, generated_ids = grpo_agent.get_action(prompt)
        batch_rewards = env.step(action)
        table_rows, total_rewards = process_batch_rewards(batch_rewards, prompt, action)
        advantages = calc_advantages(total_rewards)
        for i in range(3): # sample size
            full_input_ids = generated_full_ids[i]
            generated_id = generated_ids[i]
            memory.add_sample(full_input_ids, generated_id, advantages)

        #threshold = 1e-3
        #if advantages.abs().max().item() < threshold:
        #    print("All samples have low advantage, skipping this step.")
        #    print("advantages", advantages)
        #    memory.clear()
        #    continue 
        
    print('------------------------------------go optimise')
    grpo_agent.optimise_network()
    # Copy to ref model

    #TODO: Fix this later
    #for row in table_rows:
    #    test_table.add_data(*row)
    #wandb.log({"total_reward": np.mean(total_rewards)})
    
    # TODO: When do we update the reference model? Every x steps?
    if k == 15:
        grpo_agent.update_reference_model()

wandb.log({"test_table": test_table})
wandb.finish()