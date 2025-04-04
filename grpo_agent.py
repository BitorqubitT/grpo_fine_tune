import torch
import torch.optim as optim

class GRPO_agent():

    def __init__(self, model, tokenizer, chat_template: str, amount_of_answers: int = 5, memory=None, lr=1e-5):
        self.model = model
        self.reference_model = None #model
        self.memory = memory
        self.chat_template = chat_template
        self.tokenizer = tokenizer
        self.amount = amount_of_answers
        self.device = "cuda"
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

    def get_action(self, prompt):
        # Do I only sample from the value model or also from reference model?
        # Maybe make it optionaL?
        # More effictient way of getting value + better naming?

        messages = [
            {"role": "system", "content": self.chat_template},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text] * self.amount, return_tensors="pt", padding=True).to(self.device)
        #model_inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            num_return_sequences= self.amount
      
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return answers

    def sample_outputs(self, queries):
        outputs = []
        for query in queries:
            inputs = self.tokenizer(query, return_tensors="pt").input_ids
            samples = [
                self.model.generate(inputs, max_length=100, do_sample=True) for _ in range(self.G)
            ]
            decoded_samples = [self.tokenizer.decode(s[0], skip_special_tokens=True) for s in samples]
            outputs.append(decoded_samples)
        return outputs

    def compute_rewards(self, outputs, reward_fn):
        return [[reward_fn(o) for o in group] for group in outputs]

    def compute_advantages(self, rewards):
        advantages = []
        for group in rewards:
            mean_reward = sum(group) / len(group)
            advantages.append([r - mean_reward for r in group])
        return advantages

    def optimise_network(self, queries, outputs, advantages):
        loss = 0
        # Pretty sure we want
        # get memory class
        # Multiple epochs ???? not sure about this
        # Calculate old and new logits to calc kl_loss
        # Update model
        # Do we put these back in memory so we can iterate over them a few times?

        # Update ref ever?

        # Get from memory
        #input = dataset.sample()
        #outputs = self.sample_outputs(batch)
        #rewards = self.compute_rewards(outputs, reward_fn)
        #advantages = self.compute_advantages(rewards)
        
        obs, actions, logprobs, rewards = self.memory.get_values()

        for i in range(0, 5):
            ref_logits = self.reference_model(input)
            ref_output = self.get_model(input)

            kl_loss = kl_div(log_softmax(logits, dim=-1),
                             log_softmax(ref_logits, dim=-1),
                             reduction="batchmean"
            )

            kl_loss = torch.clamp(kl_loss, max=self.kl_clip)
            # Do we use old advantage?
            loss += -(advantage * kl_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Memory():
    """ 
        Class that holds memory for ppoagent
    """
    def __init__(self, num_steps: int, num_envs: int, device: torch.device):
        self.obs = torch.zeros((num_steps, num_envs) + (9,)).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + ()).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        #TODO: add advantage, attention mask, kl?


    def update_values(self, step: int, obs, actions, logprobs, rewards):
        self.obs[step] = obs
        self.actions[step] = actions
        self.logprobs[step] = logprobs
        self.rewards[step] = rewards

    
    def get_values(self):
        return self.obs, self.actions, self.logprobs, self.rewards