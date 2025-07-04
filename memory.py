import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple

class Memory:
    # Better at handling batches
    def __init__(self, tokenizer, device, amount_of_samples: int, amount_of_prompts: int):
        self.tokenizer = tokenizer
        self.device = device
        self.buffer = []
        self.amount_of_samples = amount_of_samples
        self.amount_of_prompts = amount_of_prompts

    def add_sample(self, input_ids: torch.Tensor, actions: torch.Tensor, advantage: float):
        """
        Store one sample:
        - input_ids: full sequence (prompt + generated response), 1D LongTensor
        - actions: generated tokens only (response), 1D LongTensor
        - advantage: scalar float advantage for this sample
        """
        self.buffer.append({
            "input_ids": input_ids.detach().cpu(),
            "actions": actions.detach().cpu(),
            "advantage": advantage.clone().detach().to(dtype=torch.bfloat16, device=self.device),
        })

    def clear(self):
        self.buffer = []

    def get_value_per_batch(self, batch_number):
        """
        Returns padded tensors:
        - input_ids: LongTensor [B, max_seq_len]
        - actions: LongTensor [B, max_seq_len], aligned with input_ids (response tokens only, rest = -100)
        - advantages: FloatTensor [B]
        """
        if len(self.buffer) == 0:
            return None, None, None

        input_ids_list = [sample["input_ids"] for sample in self.buffer[self.amount_of_samples*batch_number:((batch_number+1)*self.amount_of_samples)]]
        response_ids_list = [sample["actions"] for sample in self.buffer[self.amount_of_samples*batch_number:((batch_number+1)*self.amount_of_samples)]]
        advantages = torch.stack([sample["advantage"] for sample in self.buffer[self.amount_of_samples*batch_number:((batch_number+1)*self.amount_of_samples)]])

        #TODO: FiX UGLy codE
        input_ids_list = [sample["input_ids"] for sample in self.buffer[4*batch_number:((batch_number+1)*4)]]
        response_ids_list = [sample["actions"] for sample in self.buffer[4*batch_number:((batch_number+1)*4)]]
        advantages = torch.stack([sample["advantage"] for sample in self.buffer[4*batch_number:((batch_number+1)*4)]])

        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        max_seq_len = input_ids_padded.size(1)
        actions_aligned = torch.full((len(input_ids_list), max_seq_len), -100, dtype=torch.long)

        for i, (full_ids, response_ids) in enumerate(zip(input_ids_list, response_ids_list)):
            prompt_len = full_ids.size(0) - response_ids.size(0)
            actions_aligned[i, prompt_len:prompt_len + response_ids.size(0)] = response_ids

        return (input_ids_padded.to(self.device),
                actions_aligned.to(self.device),
                advantages.to(self.device))
    