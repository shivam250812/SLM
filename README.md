# Quantized Small Language Models as Tool-Using Agents

Code and evaluation harness for *Quantized Small Language Models as Tool-Using
Agents: Memory Savings for Free, Fine-Tuning with Care*.

We evaluate Phi-3-mini (3.8B) end-to-end as a ReAct agent under FP16 and NF4
INT4 quantization, on a single NVIDIA T4, across three benchmarks: a 50-task
diagnostic suite, GSM8K with a calculator tool, and HotpotQA (distractor) with
a search tool. We additionally test whether QLoRA fine-tuning on 15 generic
ReAct demonstrations helps or hurts.

## Install

```bash
git clone <this repo> && cd <this repo>
pip install -r requirements.txt
export PYTHONPATH=src
```

Requires one GPU with at least 16 GB for the FP16 configuration; the INT4
configurations fit comfortably in 8 GB.

## Reproduce everything

```bash
bash scripts/run_all.sh
```

This runs all configurations, trains the adapters, runs the seed-variance
sweep, and writes `analysis/stats_output.md` with every statistic reported in
the paper.

## Run one configuration

```bash
python src/run_eval.py --config fp16       --benchmark all
python src/run_eval.py --config int4       --benchmark all
python src/train_qlora.py --out checkpoints/lora-adapters --seed 42
python src/run_eval.py --config int4_qlora --benchmark all \
    --adapter-path checkpoints/lora-adapters
```

Per-task results land in `results/<benchmark>_<config>.csv` with columns
`correct`, `steps`, `tool_calls`, `latency_s`, and the full agent `trace`.

## Analysis

```bash
python analysis/stats.py --results results --out analysis/stats_output.md
```

Because every configuration runs the same task list in the same order, the
results are **paired**. The analysis therefore uses McNemar's exact test for
accuracy and Wilcoxon signed-rank for tool-call depth, rather than comparing
independent proportions. It also runs a TOST equivalence test, which is what
licenses the claim that quantization is "essentially free" — a non-significant
difference test alone does not.

## Layout

```
src/agent.py         ReAct loop with real tool execution (shared by all configs)
src/tools.py         calculator, fact-base search, HotpotQA paragraph search
src/prompts.py       system prompts (identical across configs)
src/benchmarks.py    Custom-50, GSM8K, HotpotQA loaders and scorers
src/models.py        FP16 / INT4 / INT4+QLoRA / INT4+zero-adapter loading
src/run_eval.py      unified evaluation driver
src/train_qlora.py   QLoRA fine-tuning on the 15 demonstrations
analysis/stats.py    all statistics reported in the paper
data/demonstrations.json  the 15 training traces (mean 1.60 tool calls each)
notebooks/           original cloud notebook notebooks, outputs intact (see its README)
results/             measured per-task results from the HotpotQA replication
results_reconstructed/  original-round results recovered from notebook stdout
checkpoints/         trained LoRA adapters (seed 42)
```

## Data artifacts

`results/` holds the HotpotQA replication run: full per-task CSVs with
`tool_calls`, `steps` and complete agent traces, plus `environment.json` and
per-config metadata including peak GPU memory. This is the data behind the
trace-depth analysis.

`results_reconstructed/` holds the original evaluation round, recovered from
the notebooks' printed stdout after the cloud notebook session was lost. Accuracy and
latency are faithful; per-task `tool_calls` could not be recovered. Read
`results_reconstructed/PROVENANCE.md` before using it.

## Methodological notes

**Unified harness.** All configurations import the same `src/agent.py`,
`src/tools.py` and `src/benchmarks.py`. The model is the only variable.
This matters: an earlier evaluation round with a different harness and an
older bitsandbytes kernel scored the identical INT4 model 6 points lower on
the diagnostic suite. Small quantized models sit close to capability
thresholds on multi-step tasks, so harness and kernel differences move
headline agentic scores.

**Clean ablation.** `scripts/run_all.sh` runs each configuration in a separate
process. This is deliberate. `prepare_model_for_kbit_training()` upcasts layer
norms and the LM head to fp32, so evaluating INT4 and then fine-tuning the
same in-memory model would confound adapter effects with dtype changes. The
`int4_zero` configuration is the control for exactly this: prepared model,
freshly initialised adapter, no training.

**Concurrency.** Configurations should be run on the same physical GPU in
comparable conditions if latency is being compared. Latency measured across
different cloud sessions is not directly comparable.

**Memory.** `models.load_model` returns weight memory from
`get_memory_footprint()`; `run_eval.py` additionally records peak allocated
memory, which includes activations and the KV cache and is the number that
determines whether a model actually fits on a device.

## License

MIT. (Replace this section if you prefer a different license.)
