import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import get_rewards, calc_advantages, process_batch_rewards
from env import env
from templates import SYSTEM_PROMPT, template_rs_file, CARGO_TOML_FILE
import pandas as pd
import wandb
import numpy as np
# This script is used to run the non-finetuned models on the dataset.
# This is used as a baseline to compare against the finetuned models.

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

    model_inputs = tokenizer([text], return_tensors="pt", padding=True, return_attention_mask=True).to(device)

    generated_full_ids = model.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens = 1024,
        do_sample=True,
        top_p=0.90,
        temperature=0.2,
    )

    prompt_len = model_inputs.input_ids.shape[1]
    generated_ids = [output_ids[prompt_len:] for output_ids in generated_full_ids]
    
    answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return answers

if __name__ == "__main__":

    columns = ['question',
            'generated_code',
            'total_reward',
            'not empty',
            'code block',
            'test block',
            'asserts',
            'build',
            'clippy',
            'test'
            ]

    device = "cuda" # the device to load the model onto
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    test_table = wandb.Table(columns = columns)

    df = pd.read_parquet("data/results_code_and_tests.parquet")
    dataset = datasets.Dataset.from_pandas(df)
    dataset = dataset.shuffle(seed=1337)
    
    eval_dataset = dataset.select(range(500))
    train_dataset = dataset.select(range(500, len(dataset)))

    train_dataset.save_to_disk("data/train_split")
    eval_dataset.save_to_disk("data/eval_split")

    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    wandb.init(project = "llm finetune eval 344512",
        name = f"eval set non-finetuned 3445123",
        config = {
                "gamma": 5,
                })

    print(len(eval_dataset))
    env = env(CARGO_TOML_FILE, template_rs_file)

    for k, batch in enumerate(eval_loader):
        prompt = batch['rust_prompt'][0]
        action = get_answer(prompt, SYSTEM_PROMPT, tokenizer, model, amount=1, device=device)
        batch_rewards = env.step(action)
        table_rows, total_rewards = process_batch_rewards(batch_rewards, prompt, action)

        for row in table_rows:
            test_table.add_data(*row)
            wandb.log({"total_reward": np.mean(total_rewards)})

    wandb.log({"test_table": test_table})
    wandb.finish()