import torch
import warnings
import os
import sys

# 1. Silence the legacy TF32 warning
warnings.filterwarnings("ignore", message=".*use the new API settings to control TF32 behavior.*")
warnings.filterwarnings("ignore", message=".*TensorFloat32 tensor cores.*available but not enabled.*")

# 2. Assert GPU Availability immediately
if not torch.cuda.is_available():
    print("CRITICAL ERROR: CUDA is NOT available to PyTorch.")
    print("Training on CPU will be extremely slow. Please re-run setup.ps1 and ensure NVIDIA drivers are installed.")
    print(f"PyTorch version: {torch.__version__}")
    sys.exit(1)

print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM detected: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 3. Set high-level matmul precision for RTX 5090
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer, SFTConfig

# Optimized DataCollator
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(self, response_template, tokenizer, *args, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=False, *args, **kwargs)
        self.response_template = response_template
        self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            input_ids = batch["input_ids"][i]
            ids_list = input_ids.tolist()
            for j in range(len(ids_list) - len(self.response_token_ids) + 1):
                if ids_list[j : j + len(self.response_token_ids)] == self.response_token_ids:
                    mask_until = j + len(self.response_token_ids)
                    batch["labels"][i, :mask_until] = -100
                    break
        return batch

# Model configuration
MODEL_ID = "LiquidAI/LFM2-350M"
DATASET_PATH = "training data/dataset.jsonl"
OUTPUT_DIR = "./lfm2-350m-zimage-turbo"

def formatting_prompts_func(example):
    if isinstance(example.get("instruction"), list):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"### Instruction:\n{example['instruction'][i]}\n\n### Input:\n{example['input'][i]}\n\n### Response:\n{example['output'][i]}"
            output_texts.append(text)
        return output_texts
    else:
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"

def train():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load dataset
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # Load model with bfloat16 and force onto CUDA
    print("Loading model to GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")

    # Compile with Triton
    print("Compiling model (CPU spike expected during optimization)...")
    model = torch.compile(model)

    # Optimized SFTConfig for RTX 5090 with 32GB VRAM
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_steps=3000, 
        per_device_train_batch_size=64, 
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500, 
        bf16=True,
        tf32=True, 
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_steps=300, 
        gradient_checkpointing=False,
        optim="adamw_8bit",
        max_length=512,
        dataset_text_field="text",
        dataloader_num_workers=0, 
        dataloader_pin_memory=True,
    )

    response_template = "### Response:\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        args=training_args,
    )

    print("Starting training on GPU...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()
