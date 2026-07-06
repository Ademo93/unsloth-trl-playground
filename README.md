# Unsloth + TRL Playground

Fast LLM fine-tuning experiments comparing **TRL** (the Hugging Face reference stack) and **Unsloth** (hand-written Triton kernels, ~2x faster and ~60% less VRAM on a single GPU).

Same dataset, same LoRA settings, two backends — so the speed/memory trade-off is measured, not guessed.

## Why two backends

| | TRL (SFTTrainer) | Unsloth |
|---|---|---|
| Ecosystem | full HF ecosystem, multi-GPU, every trainer (SFT/DPO/GRPO/PPO/reward) | single GPU focus, SFT/DPO/GRPO |
| Speed (7B, QLoRA) | baseline | ~2x faster |
| VRAM | baseline | up to -60% |
| Kernels | PyTorch + flash-attn | custom Triton, manual autograd |
| Export | HF checkpoint | HF checkpoint + direct GGUF/Ollama export |

Rule of thumb: prototype on Unsloth (Colab T4 friendly), scale on TRL when you need multi-GPU or trainers Unsloth doesn't cover.

## Quickstart

```bash
# TRL path (any CUDA GPU)
pip install -r requirements-trl.txt
python src/sft_trl.py

# Unsloth path (Linux / WSL, CUDA GPU)
pip install -r requirements-unsloth.txt
python src/sft_unsloth.py
```

Both scripts fine-tune `Qwen2.5-3B-Instruct` on `yahma/alpaca-cleaned` with identical LoRA hyperparameters (r=16, alpha=32, all linear layers) and log step time + peak VRAM at the end, so runs are directly comparable.

## Project structure

```
src/
  sft_trl.py        # TRL SFTTrainer + peft + bitsandbytes
  sft_unsloth.py    # Unsloth FastLanguageModel, same hyperparameters
  benchmark.py      # parse both runs' metrics and print a comparison table
notebooks/
  colab_unsloth.md  # notes for running the Unsloth path on free Colab
```

## Benchmark protocol

1. Fixed seed, fixed dataset slice (first 5 000 examples), 200 optimizer steps.
2. Record: wall-clock per step, `torch.cuda.max_memory_allocated()`, final train loss.
3. Same effective batch size (batch 2 x grad accum 8) and sequence length (1024).

Loss curves should overlap almost exactly — if they don't, the comparison is broken, not the model.

## Gotchas learned the hard way

- Unsloth must be imported **before** transformers/trl, its import hook patches them.
- Unsloth is Linux-first: on Windows use WSL2, native support is experimental.
- `use_gradient_checkpointing="unsloth"` (their variant) saves more memory than the vanilla one.
- With TRL, `packing=True` changes the effective tokens/step — keep it identical on both sides.

## References

- [TRL documentation](https://huggingface.co/docs/trl)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)

## License

MIT
