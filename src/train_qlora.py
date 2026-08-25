"""QLoRA fine-tuning on 15 generic ReAct demonstrations.

The demonstrations live in data/demonstrations.json and are disjoint from all
evaluation tasks in entities, numbers and wording. No GSM8K or HotpotQA data
is used: the hypothesis under test is whether generic format demonstrations
transfer.

Usage
-----
    python src/train_qlora.py --out checkpoints/lora-adapters --seed 42

To reproduce the seed-variance analysis in the paper, run this with several
seeds and evaluate each resulting adapter:

    for s in 0 1 2; do
        python src/train_qlora.py --out checkpoints/lora-seed$s --seed $s
        python src/run_eval.py --config int4_qlora --benchmark hotpotqa \
            --adapter-path checkpoints/lora-seed$s --outdir results/seed$s
    done
"""
import argparse
import json
import os
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer

from models import LORA_KWARGS, MODEL_ID, load_model
from prompts import CUSTOM_SYSTEM_PROMPT


def build_dataset(path="data/demonstrations.json"):
    demos = json.load(open(path))
    rows = [{
        "text": (f"<|system|>\n{CUSTOM_SYSTEM_PROMPT}<|end|>\n"
                 f"<|user|>\nTask: {d['task']}<|end|>\n"
                 f"<|assistant|>\n{d['trace']}<|end|>")
    } for d in demos]
    mean_actions = sum(d["trace"].count("Action:") for d in demos) / len(demos)
    print(f"{len(rows)} demonstrations, mean {mean_actions:.2f} tool calls each")
    return Dataset.from_list(rows), mean_actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/lora-adapters")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--demos", default="data/demonstrations.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    train_dataset, mean_actions = build_dataset(args.demos)

    model, tokenizer, _ = load_model("int4", seed=args.seed)
    model.config.use_cache = False
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(**LORA_KWARGS))
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out, "_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=1,
        save_strategy="no",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        dataset_text_field="text", max_seq_length=768, packing=False,
        tokenizer=tokenizer,
    )
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60

    os.makedirs(args.out, exist_ok=True)
    trainer.model.save_pretrained(args.out)
    json.dump({
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "n_demonstrations": len(train_dataset),
        "mean_actions_per_demo": round(mean_actions, 3),
        "train_minutes": round(minutes, 2),
        "base_model": MODEL_ID,
        "lora": LORA_KWARGS,
    }, open(os.path.join(args.out, "train_meta.json"), "w"), indent=2)
    print(f"Adapters saved to {args.out} ({minutes:.1f} min)")


if __name__ == "__main__":
    main()
