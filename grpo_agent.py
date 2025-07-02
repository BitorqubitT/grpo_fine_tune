import torch
import torch.optim as optim
from torch.nn.functional import kl_div, log_softmax
from utils import get_logprobs
from torch.nn.utils.rnn import pad_sequence
import gc
import copy
from collections import deque
from transformers import get_linear_schedule_with_warmup

class GRPO_agent():

    def __init__(self, model, reference_model, tokenizer, chat_template: str, amount_of_answers: int = 5, memory=None, lr=5e-6):
        self.model = model
        self.reference_model = reference_model.eval()
        #self.reference_model = reference_model
        self.memory = memory
        self.chat_template = chat_template
        self.tokenizer = tokenizer
        self.amount = amount_of_answers
        self.device = "cuda"
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=lr,
                                     betas=(0.9, 0.99),
                                     weight_decay=0.01)
        self.kl_clip = 0.1
        self.clip_eps = 0.2
        self.kl_coef = 0.1
        self.num_steps = 4
        self.losses = deque(maxlen=100)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=1500,  # Adjust as needed
            #TODO: calculate the real steps
            num_training_steps=15000  # Adjust as needed
        )
        self.accumulation_steps = 1

    def get_action(self, prompt) -> tuple:
        """
        answers:  human-readable text (useful for logging or reward computation)
        model_inputs.input_ids: prompt
        generated_ids: what the model did aka answer
        generated_full_ids: full sequence, useful for recovering the original generation context
        """

        messages = [
            {"role": "system", "content": self.chat_template},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(self.device)
        attention_mask = (model_inputs.input_ids != self.tokenizer.pad_token_id).long()

        generated_full_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.90,
            temperature=0.2,
            num_return_sequences=self.amount
        )

        prompt_len = model_inputs.input_ids.shape[1]
        generated_ids = [output_ids[prompt_len:] for output_ids in generated_full_ids]

        answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return answers, model_inputs.input_ids, generated_full_ids, generated_ids

    def update_reference_model(self):
        self.reference_model.load_state_dict(self.model.state_dict())
        self.reference_model.eval()

    def update_loss(self, new_loss):
        self.losses.append(new_loss)
        avg_loss = sum(self.losses) / len(self.losses)
        return avg_loss

    def optimise_network(self):

        for i in range(2):

            input_ids, actions, advantages = self.memory.get_value_per_batch(i)
            old_logprobs = get_logprobs(self.model, input_ids, actions, self.tokenizer, use_no_grad=True)
            policy_loss = 0.0

            selected_rows = torch.arange(0, advantages.size(0), self.amount, device=advantages.device)
            filtered = advantages[selected_rows]
            
            advantages = filtered.view(-1)
            action_mask = (actions != -100).float()
            # We do this because we have logprobs per token
            advantages = advantages.unsqueeze(1) * action_mask  # [B, S] -> [B, S] with actions masked 
            # 4 to 8
            logging_metrics = []


            print(len(self.memory.buffer), "samples in memory")
            
            for step in range(1):
                new_logprobs = get_logprobs(self.model, input_ids, actions, self.tokenizer, False)
                
                ratio = torch.exp(new_logprobs - old_logprobs)

                # PPO-style clipped loss
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                loss_unclipped = ratio * advantages
                loss_clipped = clipped_ratio * advantages

                per_token_loss = -torch.min(loss_unclipped, loss_clipped)  # [B, S]

                # Normalize: mean over non-masked tokens
                policy_loss = per_token_loss.sum(dim=1).mean()

                #TODO: Put in ref mode
                with torch.no_grad():
                    ref_logprobs = get_logprobs(self.reference_model, input_ids, actions, self.tokenizer, False)

                kl_div = new_logprobs - ref_logprobs
                kl_loss = torch.mean(kl_div)
                #print("kl_loss:", kl_loss.item())
                total_loss = policy_loss + self.kl_coef * kl_loss
                total_loss = total_loss / self.accumulation_steps
                #print("total_loss:", total_loss.item())

                total_loss.backward()
                

                if i + 1 == self.accumulation_steps:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    average_loss = self.update_loss(total_loss.item())
                    print(average_loss, "average loss")
                    logging_metrics.append([step, total_loss, kl_loss, average_loss])

                
                    with torch.no_grad():
                        del new_logprobs, ratio, clipped_ratio, policy_loss, total_loss, ref_logprobs, kl_div, kl_loss, average_loss, per_token_loss
                    
                    gc.collect()
                    torch.cuda.empty_cache()

        return logging_metrics