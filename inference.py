import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from peft import PeftModel, PeftConfig
import datasets
import wandb
from utils import process_batch_rewards
from env import env
from templates import SYSTEM_PROMPT, template_rs_file, CARGO_TOML_FILE
from peft import PeftModel, LoraConfig

def get_answer(prompt, chat_template, tokenizer, model, amount=1, device="cuda"):
    messages = [
        {"role": "system", "content": chat_template},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    ).to(device)

    with torch.no_grad():
        generated_full_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.9,
            temperature=0.2,
        )

    prompt_len = model_inputs.input_ids.shape[1]
    generated_ids = [output_ids[prompt_len:] for output_ids in generated_full_ids]
    answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return answers


if __name__ == "__main__":
    device = "cuda"
    base_model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    adapter_path = "saved_models/grpo_3000_maybe2k.pth"

    columns = [
        'question', 'generated_code', 'total_reward', 'not empty',
        'code block', 'test block', 'asserts', 'build', 'clippy', 'test',
        'output_build', 'output_clippy', 'output_test'
    ]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    # === Build a PEFT model shell and load LoRA weights ===
    print(f"🔹 Loading LoRA adapter weights from {adapter_path} ...")

    # Create a dummy PEFT config (needed to reconstruct the PEFT wrapper)
    lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")
    bias="none",
    task_type="CAUSAL_LM",
)

    model = PeftModel(base_model, lora_config)

    adapter_state_dict = torch.load(adapter_path, map_location=device)
    model.load_state_dict(adapter_state_dict, strict=False)

    model = model.merge_and_unload()
    model.eval()
    print("✅ LoRA model loaded and merged successfully.")

    df = pd.read_parquet("data/cargo_test_passed_train.parquet")
    dataset = datasets.Dataset.from_pandas(df).shuffle(seed=1337)

    eval_dataset = dataset.select(range(500))
    train_dataset = dataset.select(range(500, len(dataset)))

    train_dataset.save_to_disk("data/train_split")
    eval_dataset.save_to_disk("data/eval_split")

    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    wandb.init(
        project="llm-finetuned-eval-2k",
        name="eval-finetuned-qwen-code-1.5b-lora2",
        config={}
    )

    env_instance = env(CARGO_TOML_FILE, template_rs_file)
    test_table = wandb.Table(columns=columns)

    for k, batch in enumerate(eval_loader):
        prompt = batch["rust_prompt"][0]
        action = get_answer(prompt, SYSTEM_PROMPT, tokenizer, model, amount=1, device=device)
        batch_rewards = env_instance.step(action)
        table_rows, total_rewards = process_batch_rewards(batch_rewards, prompt, action)

        for row in table_rows:
            row.append(batch_rewards[0]["result_outputbuild"])
            row.append(batch_rewards[0]["result_outputclippy"])
            row.append(batch_rewards[0]["result_outputtest"])
            test_table.add_data(*row)
            wandb.log({"total_reward": np.mean(total_rewards)})

    wandb.log({"test_table": test_table})
    wandb.finish()
