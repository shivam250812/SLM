"""Unified evaluation harness.

Every configuration runs through this exact file. The model is the only
variable. Per-task results are written to results/<benchmark>_<config>.csv
with the columns the analysis scripts expect, including tool_calls, which is
the primary evidence for the fine-tuning result in the paper.

Usage
-----
    python src/run_eval.py --config fp16       --benchmark all
    python src/run_eval.py --config int4       --benchmark all
    python src/run_eval.py --config int4_qlora --benchmark all \
        --adapter-path checkpoints/lora-adapters
    python src/run_eval.py --config int4_zero  --benchmark hotpotqa
"""
import argparse
import json
import os
import time

import pandas as pd
import torch

from agent import run_agent
from benchmarks import (BENCHMARK_50, load_gsm8k, load_hotpotqa, score_custom,
                        score_gsm8k, score_hotpot)
from models import load_model, peak_memory_gb
from prompts import CUSTOM_SYSTEM_PROMPT, GSM8K_SYSTEM_PROMPT, HOTPOT_SYSTEM_PROMPT
from tools import make_hotpot_search, tool_calculator, tool_search

CUSTOM_TOOLS = {"search": tool_search, "calculator": tool_calculator}
GSM8K_TOOLS = {"calculator": tool_calculator}


def evaluate(benchmark, config, items, agent_fn, scorer, outdir, save_every=10):
    rows = []
    path = os.path.join(outdir, f"{benchmark}_{config}.csv")
    print("=" * 70)
    print(f"RUNNING: {benchmark} | {config} | {len(items)} tasks")
    print("=" * 70)
    for i, item in enumerate(items, 1):
        try:
            ans, steps, calls, lat, trace = agent_fn(item)
        except Exception as e:  # keep the run alive; the row records the failure
            ans, steps, calls, lat, trace = f"AGENT ERROR: {e}", 0, 0, 0.0, ""
        correct = scorer(item, ans)
        rows.append({
            "id": item["id"],
            "category": item.get("category", benchmark),
            "task": item["task"][:200],
            "gold": str(item.get("expected", item.get("gold")))[:100],
            "answer": str(ans)[:200],
            "correct": bool(correct),
            "steps": steps,
            "tool_calls": calls,
            "latency_s": round(lat, 3),
            "trace": trace,          # full trace: needed to audit trace depth
        })
        print(f"{'PASS' if correct else 'FAIL'} [{i:03d}/{len(items)}] "
              f"{lat:5.1f}s  calls={calls}  {str(ans)[:55]}")
        if i % save_every == 0 or i == len(items):
            pd.DataFrame(rows).to_csv(path, index=False)
    df = pd.DataFrame(rows)
    print(f"\n>>> {benchmark} | {config}: {100 * df['correct'].mean():.1f}% "
          f"({df['correct'].sum()}/{len(df)}), "
          f"mean latency {df['latency_s'].mean():.2f}s, "
          f"mean tool calls {df['tool_calls'].mean():.2f}")
    print(f">>> saved to {path}\n")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    choices=["fp16", "int4", "int4_qlora", "int4_zero"])
    ap.add_argument("--benchmark", default="all",
                    choices=["all", "custom50", "gsm8k", "hotpotqa"])
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=100,
                    help="tasks sampled from GSM8K / HotpotQA")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    model, tokenizer, weight_gb = load_model(
        args.config, adapter_path=args.adapter_path, seed=args.seed)
    print(f"[{args.config}] weight memory: {weight_gb:.2f} GB")

    def agent_custom(item):
        return run_agent(item["task"], CUSTOM_TOOLS, CUSTOM_SYSTEM_PROMPT,
                         model, tokenizer, max_steps=6, max_new_tokens=200)

    def agent_gsm8k(item):
        return run_agent(item["task"], GSM8K_TOOLS, GSM8K_SYSTEM_PROMPT,
                         model, tokenizer, max_steps=8, max_new_tokens=260)

    def agent_hotpot(item):
        return run_agent(item["task"], {"search": make_hotpot_search(item)},
                         HOTPOT_SYSTEM_PROMPT, model, tokenizer,
                         max_steps=6, max_new_tokens=220)

    jobs = {
        "custom50": (BENCHMARK_50, agent_custom, score_custom),
        "gsm8k": (lambda: load_gsm8k(args.n, args.seed), agent_gsm8k, score_gsm8k),
        "hotpotqa": (lambda: load_hotpotqa(args.n, args.seed), agent_hotpot, score_hotpot),
    }
    selected = jobs if args.benchmark == "all" else {args.benchmark: jobs[args.benchmark]}

    for name, (items, fn, scorer) in selected.items():
        if callable(items):
            items = items()
        evaluate(name, args.config, items, fn, scorer, args.outdir)

    meta = {
        "config": args.config,
        "weight_memory_gb": round(weight_gb, 3),
        "peak_memory_gb": round(peak_memory_gb(), 3),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        import bitsandbytes
        meta["bitsandbytes"] = bitsandbytes.__version__
    except ImportError:
        meta["bitsandbytes"] = None
    try:
        import transformers
        meta["transformers"] = transformers.__version__
    except ImportError:
        pass

    with open(os.path.join(args.outdir, f"meta_{args.config}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
