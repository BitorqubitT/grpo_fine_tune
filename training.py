import os
import datasets
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from grpo_agent import GRPO_agent
from memory import Memory
from utils import get_rewards, calc_advantages, process_batch_rewards
from env import env
from templates import SYSTEM_PROMPT, template_rs_file, CARGO_TOML_FILE
import pandas as pd

device = "cuda"
#model_name = "Qwen/Qwen3-0.6b"
#model_name = "Qwen/Qwen3-1.8b"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

AMOUNT_OF_SAMPLES = 4
AMOUNT_OF_PROMPTS = 2

tokenizer = AutoTokenizer.from_pretrained(model_name, extra_vocab_file="qwen_extra.tiktoken")

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16).to(device)

model = get_peft_model(base_model, lora_config)
reference_model = get_peft_model(base_model, lora_config)

# Models are misaligned on purpose.
#dataset = datasets.load_dataset("TIGER-Lab/AceCode-87K", split='train')
#df = pd.read_parquet("data/cargo_test_passed_train.parquet")
df = pd.read_parquet("data/cargo_test_passed_train.parquet")
dataset = datasets.Dataset.from_pandas(df)
dataset = dataset.shuffle(seed=1337)

train_dataset = dataset.select(range(500, len(dataset)))

train_dataset.save_to_disk("data/train_split")

#train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
print(len(train_dataset))
data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

wandb.init(project = "llm finetune 2983129",
           name = f"experiment 9426123"
            )

memory = Memory(tokenizer, device, AMOUNT_OF_SAMPLES, AMOUNT_OF_PROMPTS)
grpo_agent = GRPO_agent(model, reference_model, tokenizer, SYSTEM_PROMPT, AMOUNT_OF_SAMPLES, memory)
env = env(CARGO_TOML_FILE, template_rs_file)

skipped_prompts = []

for k, batch in enumerate(data_loader):
    print(k)

    if k == 3000:
        break

    #TODO: change this so we also have 
    for prompt, task_id in zip(batch["rust_prompt"], batch["task_id"]):
        # str answer, prompt id, prompt+answerids, answer_ids
        action, prompt_id, generated_full_ids, generated_ids = grpo_agent.get_action(prompt)
        batch_rewards = env.step(action)
        table_rows, total_rewards = process_batch_rewards(batch_rewards, prompt, action)
        if sum(total_rewards)/len(total_rewards) == 1:
            print(total_rewards)
            skipped_prompts.append(task_id)
            continue
        
        advantages = calc_advantages(total_rewards)
        if sum(advantages)/advantages.shape[0] == advantages[0]:
            print(" no bueno")
            print(total_rewards)
            skipped_prompts.append(task_id)
            continue

        print(f"Prompt ID: {task_id}, total_rewards: {total_rewards}, advantages: {advantages}")

        for i in range(4): # sample size
            full_input_ids = generated_full_ids[i]
            generated_id = generated_ids[i]
            memory.add_sample(full_input_ids, generated_id, advantages)

    if len(memory.buffer) < 8:
        continue

    else:

        logging = grpo_agent.optimise_network()

        for row in logging:
            wandb.log({"step": row[0],
                    "loss": row[1],
                    "kl_loss": row[2],
                    "mean_advantage:": sum(advantages)/advantages.shape[0],
                    # TODO: change this to moving average?
                    "average_loss": row[3]})
        if k == 50:
            grpo_agent.update_reference_model()

        memory.clear()

print(len(skipped_prompts))
wandb.finish()