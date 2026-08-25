#!/usr/bin/env bash
# Full reproduction. Each config is a separate process so that no in-memory
# state carries between configurations -- see README, "Clean ablation".
set -euo pipefail
mkdir -p results checkpoints
export PYTHONPATH=src

bash scripts/pin_versions.sh

python src/run_eval.py --config fp16 --benchmark all
python src/run_eval.py --config int4 --benchmark all

# Control: preparation dtype changes WITHOUT trained adapter weights.
python src/run_eval.py --config int4_zero --benchmark all

python src/train_qlora.py --out checkpoints/lora-adapters --seed 42
python src/run_eval.py --config int4_qlora --benchmark all \
    --adapter-path checkpoints/lora-adapters

# Seed variance for the fine-tuning result.
for s in 0 1 2; do
    python src/train_qlora.py --out "checkpoints/lora-seed$s" --seed "$s"
    python src/run_eval.py --config int4_qlora --benchmark hotpotqa \
        --adapter-path "checkpoints/lora-seed$s" --outdir "results/seed$s"
done

python analysis/stats.py --results results --out analysis/stats_output.md
