Finetuning Qwen2-1.5B for Rust.# GRPO Fine-Tuning for Code Generation

A Python implementation of Group Relative Policy Optimization (GRPO) for fine-tuning large language models on code generation tasks, specifically for Rust.

## Overview

This project implements GRPO (Group Relative Policy Optimization), based on DeepSeek's approach. It is specifically designed for the Qwen2.5-Coder-1.5B-Instruct model with LoRA adaptation.

The current implementation utilizes a single H100 GPU, with memory optimization techniques including gradient accumulation.

## Key Features

- GRPO implementation with KL divergence control
- LoRA fine-tuning support
- Integrated Rust code evaluation environment
- Wandb logging and monitoring
- Batch processing with memory management
- Configurable training parameters

## Project Structure

- `training.py`: Main training loop and setup
- `grpo_agent.py`: GRPO agent implementation
- `config_class.py`: Configuration management
- `env.py`: Rust code evaluation environment
- `inference.py`: Model inference and evaluation

## Usage

1. Setup environment:
```bash
pip install -r requirements.txt

# On Linux, enable Flash Attention for better performance
pip install flash-attn --no-build-isolation

#Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Configure parameters in `config_class.py`

3. Start training:
```bash
python training.py
```

4. For inference:
```bash
python inference.py
```

## Monitoring

Training progress is monitored through Wandb, tracking:
- Policy loss
- KL divergence
- Build success rate
- Test success rate
- Moving averages of rewards

## Model Checkpoints

Models are saved periodically and automatically uploaded to Hugging Face Hub using the HuggingFace API. Checkpoints include:
- LoRA weights
- Full model state
- Training metadata

The model checkpoints are saved every 1000 steps and can be accessed via the specified HuggingFace repository in `training.py`. To use your own repository, update the `repo_id` in the training script.

## Hardware Requirements

This project was developed and tested on a single NVIDIA H100 GPU. The implementation is optimized for this hardware configuration, taking advantage of:
- Flash Attention (on Linux)
- Gradient accumulation for memory efficiency
- Optimized batch sizes for H100's memory capacity

For different hardware configurations, you will need to adjust the batch sizes and memory management parameters in `config_class.py`.

## Future Improvements

While the current implementation effectively utilizes a single H100 GPU, potential improvements include:
- Multi-GPU support for parallel batch processing
- Implementation of more sophisticated reward modeling

## Acknowledgments

- This project builds upon the Qwen2.5-Coder-1.5B-Instruct model
- Implementation inspired by DeepSeek's GRPO approach and PPO papers

## License

MIT License

Copyright (c) 2025 BitorqubitT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
