# Original experiment notebooks

The two cloud notebook notebooks that produced the results in Tables 1 and 2 of the
paper. They are included with **outputs intact**, deliberately.

| File | What it ran |
|---|---|
| `exp4_int4_and_qlora.ipynb` | INT4 and INT4+QLoRA across all three benchmarks; QLoRA training |
| `exp4b_fp16_baseline.ipynb` | FP16 baseline across all three benchmarks; final combined tables |

## Why the outputs are kept

`/workdir/` is wiped when a cloud notebook session ends, and the zip commands
in these notebooks were written inside triple-quoted strings, so they never
executed. The original per-task CSVs did not survive.

The printed stdout in these notebooks is therefore the **only surviving
record** of the original per-task results, and it is the source from which
`results_reconstructed/` was rebuilt. Stripping the outputs would delete the
evidence that makes that reconstruction auditable. Between the two notebooks,
773 per-task pass/fail markers and 9 run headers are preserved; anyone can
re-derive `results_reconstructed/` from them.

The later HotpotQA replication in `results/` does not have this problem: it
was run through `src/run_eval.py`, which writes per-task CSVs incrementally,
and the archive was downloaded before the session ended.

## Redaction

One Weights & Biases API key was echoed in plaintext in
`exp4b_fp16_baseline.ipynb` (it was pasted at wandb's "Enter your choice"
prompt rather than the masked key prompt, so it was not hidden). It has been
replaced with `<REDACTED-WANDB-KEY>` and the key itself has been revoked. No
other credentials are present; a scan for HuggingFace, OpenAI and AWS token
patterns came back clean.

Note that the 40-character hex string in each notebook's first-cell metadata
(`_uuid`) is cloud notebook boilerplate, not a secret.

## Known gaps in the recorded output

- `exp4_int4_and_qlora.ipynb` lost GSM8K/INT4 tasks 4–10 to cloud notebook's
  cell-output truncation. The run scored 68/100; the 93 visible tasks score
  64. See `results_reconstructed/PROVENANCE.md`.
- Per-task `tool_calls` was never printed, only summarised as a run-level
  mean. This is why the trace-depth analysis in the paper uses the
  replication run in `results/` rather than these notebooks.

## Relationship to `src/`

The harness in `src/` was extracted from these notebooks. The agent loop,
tools, prompts, benchmarks and scorers are byte-identical; `src/` adds a CLI,
per-task `tool_calls` and trace logging, peak-memory recording, environment
capture, and process isolation between configurations.
