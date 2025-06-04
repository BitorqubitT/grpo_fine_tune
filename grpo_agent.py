import torch
import torch.optim as optim
from torch.nn.functional import kl_div, log_softmax
from utils import get_logprobs
from torch.nn.utils.rnn import pad_sequence
import gc

class GRPO_agent():

    def __init__(self, model, tokenizer, chat_template: str, amount_of_answers: int = 5, memory=None, lr=1e-5):
        self.model = model
        self.reference_model = model.clone().eval()
        self.memory = memory
        self.chat_template = chat_template
        self.tokenizer = tokenizer
        self.amount = amount_of_answers
        self.device = "cuda"
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.kl_clip = 0.1
        self.clip_eps = 0.2   #Used in deepseek
        self.kl_coef = 0.1 #Used in deepseek
        self.num_steps = 3

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
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            num_return_sequences=self.amount
        )

        prompt_len = model_inputs.input_ids.shape[1]
        generated_ids = [output_ids[prompt_len:] for output_ids in generated_full_ids]

        answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return answers, model_inputs.input_ids, generated_full_ids, generated_ids

    def update_reference_model(self):
        self.reference_model.load_state_dict(self.model.state_dict())
        self.reference_model.eval()

    def optimise_network(self):
        # We dont want the raw prompt, we want the input_ids
        # And actions are the models answers

        # (prompt+answer), answer adv
        input_ids, actions, advantages = self.memory.get_values()
        print("input_ids", input_ids.shape)
        print("actions_ids", actions.shape)
        print("advantages", advantages)
        old_logprobs = get_logprobs(self.model, input_ids, actions, self.tokenizer, use_no_grad=True)
        #total_loss = 0.0
        policy_loss = 0.0

        #TODO: better naming
        selected_rows = torch.arange(0, advantages.size(0), self.amount, device=advantages.device)
        filtered = advantages[selected_rows]
        print(filtered)
        advantages = filtered.view(-1)
        print("advantages", advantages)

        # 4 to 8
        for _ in range(self.num_steps):
            new_logprobs = get_logprobs(self.model, input_ids, actions, self.tokenizer, False)
            
            # We currently use total logprobs, so for the whole sequence.
            # Look at advanatage of doing it per token

            #Values are toooooo big 
            #Lets use logspace operatoins:
            ratio_log = new_logprobs - old_logprobs
            ratio = torch.exp(torch.clamp(ratio_log, -10, 10))

            print("new_logprobs", new_logprobs)
            print("old_logprobs", old_logprobs)
            print("ratio", ratio)
            # PPO-style clipped loss
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            loss_unclipped = ratio * advantages
            loss_clipped = clipped_ratio * advantages
            policy_loss = -torch.mean(torch.min(loss_unclipped, loss_clipped))

            #TODO: Put in ref mode
            ref_logprobs = get_logprobs(self.reference_model, input_ids, actions, self.tokenizer, False)
            #ref_logprobs = new_logprobs

            kl_div = ref_logprobs - new_logprobs
            kl_loss = torch.mean(torch.clamp(kl_div, max=self.kl_clip))
            total_loss = policy_loss + self.kl_coef * kl_loss
            #print("policy_loss:", policy_loss)
            #print("kl_loss:", kl_loss)
            #print("total_loss:", total_loss)

            print("policy_loss:", total_loss)

            self.optimizer.zero_grad()
            total_loss.backward()
            #policy_loss.backward()
            self.optimizer.step() 

            del new_logprobs, ref_logprobs, ratio, clipped_ratio, policy_loss, kl_loss, total_loss
            #del new_logprobs, policy_loss
            gc.collect()
            torch.cuda.empty_cache()