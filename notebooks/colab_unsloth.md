# Running the Unsloth path on free Colab (T4)

The T4 has 16 GB of VRAM and no bf16 support — a few adjustments are needed.

## Setup cell

```python
!pip install unsloth
```

Unsloth pins compatible versions of transformers/trl/peft itself. Do not
pre-install them, and restart the runtime if Colab already imported an
incompatible transformers.

## Adjustments vs `src/sft_unsloth.py`

- T4 is pre-Ampere: replace `bf16=True` with `fp16=True` in `SFTConfig`.
- Keep `load_in_4bit=True` — a 3B model in 4-bit + LoRA uses ~5 GB, well within budget.
- If you hit OOM at `SEQ_LEN=1024`, drop `per_device_train_batch_size` to 1
  before reducing sequence length; batch size costs more memory than length here.

## Expected numbers (T4, 200 steps)

| metric | value |
|---|---|
| s / step | ~4-5 |
| peak VRAM | ~6 GB |
| total wall time | ~15-18 min |

## Exporting to GGUF for Ollama

```python
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method="q4_k_m")
```

Then download `model_gguf/*.gguf` and point an Ollama `Modelfile` at it.
