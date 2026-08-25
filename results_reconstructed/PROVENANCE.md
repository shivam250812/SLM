# Reconstructed results — READ THIS BEFORE USING

These CSVs were **recovered from the printed stdout of the original cloud notebook
notebooks**, not from the harness's own CSV output. The original
`/workdir/exp4_*.csv` files were not preserved when the cloud notebook session
ended.

The source stdout is preserved in `notebooks/` with outputs intact, so this
reconstruction is independently auditable: 773 per-task pass/fail markers
across 9 run headers.

## What is real

- `correct` — the per-task pass/fail marker printed by the harness
- `latency_s` — the per-task latency printed by the harness
- `answer` — the answer string, **truncated to 55 characters** by the print
- `id`, `category`

Aggregates recomputed from these columns match the paper's reported numbers
exactly, so the recovery is faithful for accuracy and latency.

## What is missing and cannot be recovered

- **`tool_calls`** — never printed per task. Only the run-level means survive
  (in the notebook's summary tables). The Wilcoxon signed-rank test on
  tool-call depth, which the paper relies on as the primary evidence for
  search-horizon truncation, **cannot be computed from these files.**
- `steps`, `trace`, full untruncated `answer`, `gold`, `task`

## Known gap

`gsm8k_int4.csv` contains 93 of 100 tasks. Tasks 4–10 were lost to cloud notebook's
cell-output truncation. The full run scored 68/100; the 93 recovered tasks
score 64, so the 7 missing tasks contained 4 correct and 3 incorrect, but
which is unknown. All paired tests involving this run use n=93.

## Status

Use these to reproduce the accuracy and latency analysis. **Do not cite them
as the paper's data artifact** — re-run the harness and replace this
directory with genuine `results/` output before release.
