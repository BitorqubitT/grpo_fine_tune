#https://github.com/Oxen-AI/GRPO-With-Cargo-Feedback/blob/main/train.py
#https://www.oxen.ai/blog/training-a-rust-1-5b-coder-lm-with-reinforcement-learning-grpo

import torch

# Check if CUDA is available
print("CUDA Available:", torch.cuda.is_available())

# Check the CUDA version PyTorch was built with
print("PyTorch Built with CUDA:", torch.version.cuda)

# Check the GPU being used
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Count:", torch.cuda.device_count())
else:
    print("No GPU detected, running on CPU.")
