"""SFT baseline with TRL SFTTrainer (QLoRA, Qwen2.5-3B-Instruct).

Mirror of sft_unsloth.py: same data slice, LoRA settings and schedule,
so the two backends can be benchmarked against each other.

Usage:
    python src/sft_trl.py
"""

import json
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
DATASET = "yahma/alpaca-cleaned"
N_EXAMPLES = 5000
MAX_STEPS = 200
SEQ_LEN = 1024
SEED = 42


def to_text(example: dict) -> dict:
    prompt = example["instruction"]
    if example.get("input", "").strip():
        prompt += "\n\n" + example["input"]
    return {
        "text": f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>\n"
    }


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    dataset = load_dataset(DATASET, split=f"train[:{N_EXAMPLES}]")
    dataset = dataset.map(to_text, remove_columns=dataset.column_names)

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir="outputs/trl",
            max_steps=MAX_STEPS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            max_length=SEQ_LEN,
            bf16=True,
            gradient_checkpointing=True,
            seed=SEED,
            report_to="none",
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            task_type="CAUSAL_LM",
        ),
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - start

    metrics = {
        "backend": "trl",
        "seconds_per_step": elapsed / MAX_STEPS,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "final_loss": result.training_loss,
    }
    with open("metrics_trl.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)

    trainer.save_model("outputs/trl")


if __name__ == "__main__":
    main()
