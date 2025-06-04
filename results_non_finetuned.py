from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import datasets
from utils import process_batch_rewards
from env import env
import wandb
import pandas as pd
from templates import SYSTEM_PROMPT, template_rs_file, CARGO_TOML_FILE

# This script is used to run the non-finetuned models on the dataset.
# This is used as a baseline to compare against the finetuned models.

def get_answer(prompt, chat_template, tokenizer, model, amount=3, device="cuda"):
    messages = [
        {"role": "system", "content": chat_template},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)

    generated_full_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens = 512,
        num_return_sequences = amount,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
    )

    prompt_len = model_inputs.input_ids.shape[1]
    generated_ids = [output_ids[prompt_len:] for output_ids in generated_full_ids]
    
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

    env = env(CARGO_TOML_FILE, template_rs_file)
    for k, batch in enumerate(data_loader):

        prompt = batch['rust_prompt'][0]

        action = get_answer(prompt, SYSTEM_PROMPT, tokenizer, model, amount=3, device=device)

        batch_rewards = env.step(action)
        table_rows, total_rewards = process_batch_rewards(batch_rewards, prompt, action)
        print(total_rewards)
        
        #TODO: Check if the reward structure that I am using is correct.
        #TODO: logging to wandb 