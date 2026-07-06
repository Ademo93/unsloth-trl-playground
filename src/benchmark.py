"""Print a TRL vs Unsloth comparison table from the two metrics files.

Usage:
    python src/benchmark.py
"""

import json
import pathlib
import sys


def main() -> None:
    runs = []
    for path in ("metrics_trl.json", "metrics_unsloth.json"):
        if not pathlib.Path(path).exists():
            sys.exit(f"Missing {path} — run the corresponding training script first.")
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))

    trl, uns = runs
    speedup = trl["seconds_per_step"] / uns["seconds_per_step"]
    vram_saving = 1 - uns["peak_vram_gb"] / trl["peak_vram_gb"]

    print(f"{'metric':<20}{'TRL':>12}{'Unsloth':>12}")
    print("-" * 44)
    print(f"{'s / step':<20}{trl['seconds_per_step']:>12.2f}{uns['seconds_per_step']:>12.2f}")
    print(f"{'peak VRAM (GB)':<20}{trl['peak_vram_gb']:>12.2f}{uns['peak_vram_gb']:>12.2f}")
    print(f"{'final loss':<20}{trl['final_loss']:>12.4f}{uns['final_loss']:>12.4f}")
    print("-" * 44)
    print(f"Unsloth speedup: {speedup:.2f}x, VRAM saving: {vram_saving:.0%}")


if __name__ == "__main__":
    main()
