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
from utils import get_rewards, calc_advantages, process_batch_rewards, check_loss_logging
from env import env
from templates import SYSTEM_PROMPT, template_rs_file, CARGO_TOML_FILE
import pandas as pd
import time

device = "cuda"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
#model_name = "Qwen/Qwen2.5-0.5B-Instruct"

AMOUNT_OF_SAMPLES = 4
AMOUNT_OF_PROMPTS = 2

tokenizer = AutoTokenizer.from_pretrained(model_name, extra_vocab_file="qwen_extra.tiktoken")

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    #TODO: Use in cloud build
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16).to(device)

model = get_peft_model(base_model, lora_config)

# Use this otherwise we get warning about stacking lora
reference_base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    #TODO: Use in cloud build
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16).to(device)

reference_model = get_peft_model(reference_base_model, lora_config)

#dataset = datasets.load_dataset("TIGER-Lab/AceCode-87K", split='train')
#df = pd.read_parquet("data/cargo_test_passed_train.parquet")
df = pd.read_parquet("data/cargo_test_passed_train.parquet")
dataset = datasets.Dataset.from_pandas(df)
dataset = dataset.shuffle(seed=1337)

train_dataset = dataset.select(range(500, len(dataset)))

train_dataset.save_to_disk("data/train_split")

#train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

wandb.init(project = "llm finetune 2983129test",
           name = f"experiment 9426123"
            )

memory = Memory(tokenizer, device, AMOUNT_OF_SAMPLES, AMOUNT_OF_PROMPTS)
grpo_agent = GRPO_agent(model, reference_model, tokenizer, SYSTEM_PROMPT, AMOUNT_OF_SAMPLES, memory)
env = env(CARGO_TOML_FILE, template_rs_file)

skipped_prompts = []
average_total_loss = []
updates = 0


total_loop_start = time.time()
for k, batch in enumerate(data_loader):
    print("Batch number:", k)
    loop_start = time.time()
    for prompt, task_id in zip(batch["rust_prompt"], batch["task_id"]):
        t0 = time.time()
        # str answer, prompt id, prompt+answerids, answer_ids
        action, prompt_id, generated_full_ids, generated_ids = grpo_agent.get_action(prompt)
        t1 = time.time()
        batch_rewards = env.step(action)
        t2 = time.time()
        table_rows, total_rewards = process_batch_rewards(batch_rewards, prompt, action)
        if sum(total_rewards)/len(total_rewards) == 1:
            skipped_prompts.append(task_id)
            continue
        
        advantages = calc_advantages(total_rewards)
        if sum(advantages)/advantages.shape[0] == advantages[0]:
            skipped_prompts.append(task_id)
            continue

        print(f"Prompt ID: {task_id}, total_rewards: {total_rewards}, advantages: {advantages}")
        t3 = time.time()
        for i in range(AMOUNT_OF_SAMPLES): # sample size
            full_input_ids = generated_full_ids[i]
            generated_id = generated_ids[i]
            memory.add_sample(full_input_ids, generated_id, advantages)
        t4 = time.time()
        print(f"[TIMING] get_action: {t1 - t0:.2f}s | env.step: {t2 - t1:.2f}s | reward processing: {t3 - t2:.2f}s | add_sample: {t4 - t3:.2f}s")
    if len(memory.buffer) < 8:
        continue

    else:
        updates += 1
        logging = grpo_agent.optimise_network()
        #print("Logging metrics:", logging)
        # Average total loss of that episode

        #logging_metrics.append([total_loss, kl_loss, average_loss])
        if len(average_total_loss) < 100:
            average_total_loss.append(logging[0])
        else:
            average_total_loss.pop(0)
            average_total_loss.append(logging[0])

        wandb.log({"Average total loss over last 100 runs": check_loss_logging(average_total_loss),
                   "policy_loss": logging[1],
                    "kl_loss": logging[2],
                    "mean_rewards:": sum(total_rewards)/len(total_rewards),
                    # TODO: change this to moving average?
                    "mean_policy_loss": logging[3],
                    "mean_kl_loss": logging[4]})
        
        if updates == 100:
            print("Updating reference model")
            grpo_agent.update_reference_model()
            updates = 0

        memory.clear()

    if k % 1000 == 0:
        print("Saving model")
        grpo_agent.save(str(k))

grpo_agent.save("final")
    
print(len(skipped_prompts))
wandb.finish()