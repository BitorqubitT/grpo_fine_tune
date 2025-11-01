import torch
import torch.optim as optim
from utils import get_logprobs
import gc
from collections import deque
from transformers import get_cosine_schedule_with_warmup
import copy

class GRPO_agent():

    def __init__(self, model, reference_model, tokenizer, chat_template: str, cfg, memory=None):
        self.model = model
        self.inference_model = None
        self.reference_model = reference_model.eval()
        self.memory = memory
        self.chat_template = chat_template
        self.tokenizer = tokenizer
        self.amount = cfg.amount_of_samples
        self.cfg = cfg
        self.device = "cuda"
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=cfg.optimizer.learning_rate,
                                     betas=(cfg.optimizer.betas_0, cfg.optimizer.betas_1),
                                     weight_decay=cfg.optimizer.weight_decay)
        self.kl_clip = cfg.grpo.kl_clip
        self.clip_eps = cfg.grpo.clips_eps
        self.kl_coef = cfg.grpo.kl_coef
        self.num_steps = cfg.grpo.num_steps
        self.losses = deque(maxlen=100)

        #Warmup for 750 steps (LR ramps up linearly from 0 → lr)
        #Then cosine decay for the remaining steps, smoothly going toward 0
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps= cfg.scheduler.warmup_steps,
            num_training_steps= cfg.scheduler.total_training_steps
        )
        self.backwards_steps_per_update = cfg.scheduler.backwards_steps_per_update

    def update_inference_model(self):
        model_copy = copy.deepcopy(self.model)
        merged = model_copy.merge_and_unload()
        self.inference_model = merged.eval().to(self.device)

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

        with torch.inference_mode():
            generated_full_ids = self.inference_model.generate(
                **model_inputs,
                #model_inputs.input_ids,
                #attention_mask=attention_mask,
                max_new_tokens=self.cfg.inference.max_new_tokens,
                do_sample=self.cfg.inference.do_sample,
                top_p=self.cfg.inference.top_p,
                temperature=self.cfg.inference.temperature,
                num_return_sequences=self.cfg.inference.num_return_sequences
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

    def save(self, iteration: str):
        #torch.save(self.model.state_dict(), "saved_models/grpo_" + iteration + ".pth")
        torch.save(self.model.state_dict(), "saved_models/grpo_rust.pth")

    def optimise_network(self):
        accumulated_steps = 0
        num_batches = 1
        total_loss_history = []
        kl_loss_history = []
        torch.cuda.reset_peak_memory_stats(device=self.model.device)
        for i in range(num_batches):
            input_ids, actions, advantages = self.memory.get_value_per_batch(i)
            with torch.no_grad():
                old_logprobs = get_logprobs(self.reference_model, input_ids, actions, self.tokenizer, use_no_grad=True)

            # WATCH the shapes when we use different batching sizes.
            advantages = advantages[0].view(-1, 1)
            action_mask = (actions != -100).float()
            advantages = advantages * action_mask

            print(input_ids.shape, actions.shape, advantages.shape)

            for _ in range(4):
                new_logprobs = get_logprobs(self.model, input_ids, actions, self.tokenizer, use_no_grad = False)
                ratio = torch.exp(new_logprobs - old_logprobs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                loss_unclipped = ratio * advantages
                loss_clipped = clipped_ratio * advantages
                per_token_loss = -torch.min(loss_unclipped, loss_clipped)  # [B, S]

                policy_loss = (per_token_loss.sum(dim=1) / action_mask.sum(dim=1)).mean()
                with torch.no_grad():
                    ref_logprobs = old_logprobs

                kl_div = new_logprobs - ref_logprobs
                kl_loss = torch.mean(kl_div)
                total_loss = policy_loss + self.kl_coef * kl_loss
                total_loss = total_loss / self.backwards_steps_per_update
                
                total_loss_history.append(total_loss.item())
                kl_loss_history.append(kl_loss.item())
                
                total_loss.backward()
                accumulated_steps += 1

                if accumulated_steps % (self.backwards_steps_per_update) == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    mean_policy_loss = sum(total_loss_history) / len(total_loss_history)
                    mean_kl_loss = sum(kl_loss_history) / len(kl_loss_history)
                    logging_metrics = [total_loss.float().item(),
                                       policy_loss.float().item(),
                                       kl_loss.float().item(),
                                       mean_policy_loss,
                                       mean_kl_loss]

                    del new_logprobs, ratio, clipped_ratio, policy_loss, total_loss
                    del kl_div, kl_loss, per_token_loss, loss_unclipped, loss_clipped
                    gc.collect()
                    torch.cuda.empty_cache()

        return logging_metrics