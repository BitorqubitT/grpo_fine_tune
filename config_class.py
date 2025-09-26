from dataclasses import dataclass, field
from peft import LoraConfig

# --- Sub-configs ---
@dataclass
class OptimizerConfig:
    learning_rate: float = 5e-5
    betas_0: float = 0.9
    betas_1: float = 0.99
    weight_decay: float = 0.1

@dataclass
class SchedulerConfig:
    scheduler: str = "COSINE"
    warmup_steps: int = 750
    total_training_steps: int = 15000
    backwards_steps_per_update: int = 4

@dataclass
class GRPOConfig:
    kl_clip: float = 0.1
    clips_eps: float = 0.2
    kl_coef: float = 0.1
    num_steps: int = 4

@dataclass
class InferenceConfig:
    max_new_tokens: int = 768
    do_sample: bool = True
    top_p: float = 0.90
    temperature: float = 0.9
    num_return_sequences: int = 4

@dataclass
class LoRAConfig:
    lora_dropout: float = 0.05
    r: int = 8
    lora_alpha: int = 32
    target_modules="all-linear"
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    def create_lora_config(self):
        return LoraConfig(
                r=self.r,
                lora_alpha=self.lora_alpha,
                #TODO: Maybe wrap in list?
                target_modules=list(self.target_modules),
                lora_dropout=self.lora_dropout,
                bias= self.bias,
                task_type=self.task_type,
            )

# --- Master config ---
@dataclass
class TrainingConfig:
    device: str = "cuda"
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    data_path: str = "data/cargo_test_passed_train.parquet"
    train_split_path: str = "data/train_split"
    #data_path = "data/cargo_test_passed_train.parquet"

    # Experiment setup
    amount_of_samples: int = 4
    amount_of_prompts: int = 1
    wandb_project: str = "llm_finetune"
    wandb_run_name: str = "experiment_91232"
    
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)