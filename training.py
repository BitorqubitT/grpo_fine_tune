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
from utils import get_logprobs
import os
from templates import SYSTEM_PROMPT, template_rs_file, CARGO_TOML_FILE

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = "cuda" # the device to load the model onto
#model_name = "Qwen/Qwen3-0.6b"
#model_name = "Qwen/Qwen3-1.8b"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, extra_vocab_file="qwen_extra.tiktoken")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Models are misaligned on purpose.
print("Tokenizer vocab size:", tokenizer.vocab_size)
print("Model embedding size:", model.get_input_embeddings().num_embeddings)

#dataset = datasets.load_dataset("TIGER-Lab/AceCode-87K", split='train')
df = pd.read_parquet("data/cargo_test_passed_train.parquet")
dataset = datasets.Dataset.from_pandas(df)

data_loader = DataLoader(dataset,
                         batch_size = 1,
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
memory = Memory(3, 1, device, (3, 4, 2))
x = GRPO_agent(model, tokenizer, SYSTEM_PROMPT, 3, memory)
env = env(CARGO_TOML_FILE, template_rs_file)

#TODO: Use the new get_logprobs function to get logprob
# It doesnt use more resources than calculating it right after the model call for the query.
# We giuve model, prompt, action, prompt length to retrieve it

# Should be epochs
for _ in range(1):
    for k, batch in enumerate(data_loader):
        if k == 30:
            break
        print("################################################")
        prompt = batch["rust_prompt"][0]
        action, prompt_id, generated_full_ids, generated_ids = x.get_action(prompt)
        
        batch_rewards = env.step(action)
 
        table_rows, total_rewards = process_batch_rewards(batch_rewards, prompt, action)

        wandb.log({"total_reward": np.mean(total_rewards)})
     
        #answers:  human-readable text (useful for logging or reward computation)
        #model_inputs.input_ids: prompt
        #generated_ids: what the model did aka answer
        #generated_full_ids: full sequence, useful for recovering the original generation context
        
        for row in table_rows:
            test_table.add_data(*row)

        #TODO: CHECK, I think we somehow reload the model with get_logprobs

        print("total_rewards", total_rewards)
        advantages = calc_advantages(total_rewards)
        threshold = 1e-3
        if advantages.abs().max().item() < threshold:
            print("All samples have low advantage, skipping this step.")
            print("advantages", advantages)
            memory.clear()
            continue  # Skip memory update and optimization
        
        logprobs = get_logprobs(model, prompt_id, generated_ids, tokenizer, use_no_grad=True)

        print("advantages", advantages)
        memory.update_values(prompt_id, generated_ids, logprobs, advantages)
        x.optimise_network()
        # Copy to ref model
        memory.clear()

    wandb.log({"test_table": test_table})
    wandb.finish() 