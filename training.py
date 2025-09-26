import os
import time
import tempfile
import torch
import wandb
import datasets
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from grpo_agent import GRPO_agent
from memory import Memory
from utils import (
    get_rewards, calc_advantages, process_batch_rewards,
    check_loss_logging, MovingAverage
)
from env import env as environment
from templates import SYSTEM_PROMPT, template_rs_file, CARGO_TOML_FILE
from config_class import TrainingConfig

def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, extra_vocab_file="qwen_extra.tiktoken")

def load_model(model_name: str, lora_config: LoraConfig, device):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
        #attn_implementation="flash_attention_2",
    ).to(device)
    return get_peft_model(base_model, lora_config)

#def get_lora_config(r, alpha, dropout):
#    return LoraConfig(
#        r=r,
#        lora_alpha=alpha,
#        #target_modules="all-linear",
#        #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
#        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#        lora_dropout=dropout,
#        bias="none",
#        task_type="CAUSAL_LM",
#    )

def load_dataset(path: str, location: str):
    df = pd.read_parquet(path)
    dataset = datasets.Dataset.from_pandas(df).shuffle(seed=1337)
    train_dataset = dataset.select(range(500, len(dataset)))
    train_dataset.save_to_disk(location)
    return train_dataset

def train():

    cfg = TrainingConfig()
    # TODO: check how to access

    tokenizer = load_tokenizer(cfg.model_name)
    
    model = load_model(cfg.model_name, cfg.lora.create_lora_config(), cfg.device)
    reference_model = load_model(cfg.model_name, cfg.lora.create_lora_config(), cfg.device)

    dataset = load_dataset(cfg.data_path, cfg.train_split_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name)

    memory = Memory(tokenizer, cfg.device, cfg.amount_of_samples, cfg.amount_of_prompts)
    grpo_agent = GRPO_agent(model, reference_model, tokenizer, SYSTEM_PROMPT, cfg, memory)
    env = environment(CARGO_TOML_FILE, template_rs_file)

    skipped_prompts = []
    average_total_loss = []
    updates = 0

    # Moving averages
    ma_builds = MovingAverage(1000)
    ma_tests = MovingAverage(1000)
    ma_rewards = MovingAverage(1000)
    ma_loss = MovingAverage(1000)

    for k, batch in enumerate(data_loader):
        print("Batch number:", k)
        for prompt, task_id in zip(batch["rust_prompt"], batch["task_id"]):
            # str answer, prompt id, prompt+answerids, answer_ids
            if len(prompt) > 5000:
                continue

            action, prompt_id, full_ids, generated_ids = grpo_agent.get_action(prompt)
            batch_rewards = env.step(action)
            table_rows, total_rewards = process_batch_rewards(batch_rewards, prompt, action)

           #TODO: Logical to skip similar advantages? 
            advantages = calc_advantages(total_rewards)
            if sum(advantages)/advantages.shape[0] == advantages[0]:
                skipped_prompts.append(task_id)
                continue

            print(f"Prompt ID: {task_id}, total_rewards: {total_rewards}, advantages: {advantages}")

            for i in range(cfg.amount_of_samples): # sample size
                memory.add_sample(full_ids[i], generated_ids[i], advantages)

        if len(memory.buffer) < 4:
            continue

        updates += 1
        logging = grpo_agent.optimise_network()

        # Update averages
        avg_build = sum(row[7] for row in table_rows) / cfg.amount_of_samples
        avg_test = sum(row[9] for row in table_rows) / cfg.amount_of_samples

        ma_builds.update(avg_build)
        ma_tests.update(avg_test)
        ma_rewards.update(sum(total_rewards) / cfg.amount_of_samples)
        #check if this is working well
        ma_loss.update(logging[0])

        #average_total_loss = (average_total_loss[-99:] + [logging[0]]) if average_total_loss else [logging[0]]

        if len(average_total_loss) < 100:
            average_total_loss.append(logging[0])
        else:
            average_total_loss.pop(0)
            average_total_loss.append(logging[0])

        wandb.log({
            "avg_loss_100": check_loss_logging(average_total_loss),
            "policy_loss": logging[1],
            "kl_loss": logging[2],
            "mean_rewards:": ma_rewards.average(),
            "mean_policy_loss": logging[3],
            "mean_kl_loss": logging[4],
            "average_build_succes": ma_builds.average(),
            "average_test_succes": ma_tests.average(),
            "moving loss average": ma_loss.average(),
        })
    
        if updates == 2000:
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

    #TODO: Write the whole config to csv file


if __name__ == "__main__":
    train()