"""SFT with Unsloth (QLoRA, Qwen2.5-3B-Instruct).

Mirror of sft_trl.py: same data slice, LoRA settings and schedule.
Unsloth MUST be imported before transformers/trl — its import hook
patches them with Triton kernels.

Usage:
    python src/sft_unsloth.py
"""

import json
import time

from unsloth import FastLanguageModel  # noqa: I001  (must come first)

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
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
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=SEQ_LEN,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    dataset = load_dataset(DATASET, split=f"train[:{N_EXAMPLES}]")
    dataset = dataset.map(to_text, remove_columns=dataset.column_names)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir="outputs/unsloth",
            max_steps=MAX_STEPS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            max_length=SEQ_LEN,
            bf16=True,
            seed=SEED,
            report_to="none",
        ),
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - start

    metrics = {
        "backend": "unsloth",
        "seconds_per_step": elapsed / MAX_STEPS,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "final_loss": result.training_loss,
    }
    with open("metrics_unsloth.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)

    model.save_pretrained("outputs/unsloth")


if __name__ == "__main__":
    main()
