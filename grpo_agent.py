import torch
import torch.optim as optim
from utils import get_logprobs
import gc
from collections import deque
from transformers import get_linear_schedule_with_warmup
import time
import copy
from peft import PeftModel

class GRPO_agent():

    def __init__(self, model, reference_model, tokenizer, chat_template: str, amount_of_answers: int = 5, memory=None, lr=5e-6):
        self.model = model
        self.reference_model = reference_model.eval()
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
        self.backwards_steps_per_update = 4

    def get_action(self, prompt) -> tuple:
        """
        answers:  human-readable text (useful for logging or reward computation)
        model_inputs.input_ids: prompt
        generated_ids: what the model did aka answer
        generated_full_ids: full sequence, useful for recovering the original generation context
        """
        t0 = time.time()
        messages = [
            {"role": "system", "content": self.chat_template},
            {"role": "user", "content": prompt}
        ]

        t1 = time.time()
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        t2 = time.time()
        model_inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(self.device)
        #attention_mask = (model_inputs.input_ids != self.tokenizer.pad_token_id).long()

        t2 = time.time()
        with torch.inference_mode():
            generated_full_ids = self.model.generate(
                **model_inputs,
                #model_inputs.input_ids,
                #attention_mask=attention_mask,
                max_new_tokens=1024,
                do_sample=True,
                top_p=0.90,
                temperature=0.9,
                num_return_sequences=self.amount
            )

        t3 = time.time()
        prompt_len = model_inputs.input_ids.shape[1]
        generated_ids = [output_ids[prompt_len:] for output_ids in generated_full_ids]

        t4 = time.time()
        answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        t5 = time.time()

        print(
        f"[get_action timings] message_prep={t1 - t0:.2f}s | "
        f"template_apply={t2 - t1:.2f}s | "
        f"generation={t3 - t2:.2f}s | "
        f"postproc_decode={t5 - t4:.2f}s"
        )
        return answers, model_inputs.input_ids, generated_full_ids, generated_ids

    def get_action_bish(self, prompt) -> tuple:
        """
        answers:  human-readable text (useful for logging or reward computation)
        model_inputs.input_ids: prompt
        generated_ids: what the model did aka answer
        generated_full_ids: full sequence, useful for recovering the original generation context
        """
        t0 = time.time()
        # Deep copy (to keep training model intact)
        model_for_inference = copy.deepcopy(self.model)

        # Merge LoRA weights into base model
        model_for_inference = model_for_inference.merge_and_unload()
        model_for_inference.eval()  # important for dropout etc.
        model = torch.compile(model_for_inference)

        messages = [
            {"role": "system", "content": self.chat_template},
            {"role": "user", "content": prompt}
        ]

        t1 = time.time()
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        t2 = time.time()
        model_inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(self.device)
        #attention_mask = (model_inputs.input_ids != self.tokenizer.pad_token_id).long()

        t2 = time.time()
        with torch.inference_mode():
            generated_full_ids = model.generate(
                **model_inputs,
                #model_inputs.input_ids,
                #attention_mask=attention_mask,
                max_new_tokens=768,
                do_sample=True,
                top_p=0.90,
                temperature=0.9,
                num_return_sequences=self.amount
            )

        t3 = time.time()
        prompt_len = model_inputs.input_ids.shape[1]
        generated_ids = [output_ids[prompt_len:] for output_ids in generated_full_ids]

        # Token lengths of the generated outputs
        output_lengths = [len(g_ids) for g_ids in generated_ids]

        # Full sequence token lengths and prompt length
        full_lengths = [len(full) for full in generated_full_ids]


        t4 = time.time()
        answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        t5 = time.time()

        print(
        f"[get_action timings] message_prep={t1 - t0:.2f}s | "
        f"template_apply={t2 - t1:.2f}s | "
        f"generation={t3 - t2:.2f}s | "
        f"postproc_decode={t5 - t4:.2f}s"
        )
        return answers, model_inputs.input_ids, generated_full_ids, generated_ids



    def update_reference_model(self):
        self.reference_model.load_state_dict(self.model.state_dict())
        self.reference_model.eval()

    def update_loss(self, new_loss):
        self.losses.append(new_loss)
        avg_loss = sum(self.losses) / len(self.losses)
        return avg_loss

    def save(self, iteration: str):
        torch.save(self.model.state_dict(), "saved_models/grpo_" + iteration + ".pth")

    def optimise_network(self):
        accumulated_steps = 0
        num_batches = 1
        total_loss_history = []
        kl_loss_history = []
        for i in range(num_batches):

            input_ids, actions, advantages = self.memory.get_value_per_batch(i)
            # TODO: SHOULD THIS BE REFERENCE MODEL?
            with torch.no_grad():
                old_logprobs = get_logprobs(self.reference_model, input_ids, actions, self.tokenizer, use_no_grad=True)

            # WATCH the shapes when we use different batching sizes.
            advantages = advantages[0].view(-1, 1)
            action_mask = (actions != -100).float()
            advantages = advantages * action_mask

            #print("max gpu used", torch.cuda.max_memory_allocated() / 1024**3, "GB")

            for _ in range(4):
                new_logprobs = get_logprobs(self.model, input_ids, actions, self.tokenizer, use_no_grad = False)
                ratio = torch.exp(new_logprobs - old_logprobs)
                # PPO-style clipped loss
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                loss_unclipped = ratio * advantages
                loss_clipped = clipped_ratio * advantages
                per_token_loss = -torch.min(loss_unclipped, loss_clipped)  # [B, S]

                policy_loss = per_token_loss.sum(dim=1).mean()
                with torch.no_grad():
                    #ref_logprobs = get_logprobs(self.reference_model, input_ids, actions, self.tokenizer, False)
                    #TODO: Is this correct?
                    ref_logprobs = old_logprobs

                kl_div = new_logprobs - ref_logprobs
                kl_loss = torch.mean(kl_div)
                #print("kl_loss:", kl_loss.float().item())
                total_loss = policy_loss + self.kl_coef * kl_loss
                #print("policy loss", policy_loss.float().item())
                total_loss = total_loss / self.backwards_steps_per_update
                #print("total_loss:", total_loss.float().item())
                
                total_loss_history.append(total_loss.item())
                kl_loss_history.append(kl_loss.item())
                
                total_loss.backward()
                
                accumulated_steps += 1
                print("Current GPU memory allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
                if accumulated_steps % (self.backwards_steps_per_update) == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Logging total and kl loss like this is useless.
                    mean_policy_loss = sum(total_loss_history) / len(total_loss_history)
                    mean_kl_loss = sum(kl_loss_history) / len(kl_loss_history)
                    logging_metrics = [total_loss.float().item(),
                                       policy_loss.float().item(),
                                       kl_loss.float().item(),
                                       mean_policy_loss,
                                       mean_kl_loss]

                    del new_logprobs, ratio, clipped_ratio, policy_loss, total_loss
                    del ref_logprobs, kl_div, kl_loss, per_token_loss, loss_unclipped, loss_clipped
                    #del new_logprobs, ratio, clipped_ratio, policy_loss, total_loss, ref_logprobs, kl_div, kl_loss, per_token_loss
                    gc.collect()
                    torch.cuda.empty_cache()

        #print("out this function")
        return logging_metrics