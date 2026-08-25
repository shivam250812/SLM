# Statistical analysis

## Accuracy with Wilson 95% CIs

| Benchmark | Config | Correct | Accuracy | 95% CI |
|---|---|---|---|---|
| hotpotqa | int4 | 43/100 | 43.0% | [33.7, 52.8] |
| hotpotqa | int4_qlora | 38/100 | 38.0% | [29.1, 47.8] |

## McNemar exact tests (paired)

| Benchmark | Comparison | Discordant | Diff (pts) | 95% CI | p |
|---|---|---|---|---|---|
| hotpotqa | int4 vs int4_qlora | 7/2 | +5.0 | [-0.8, +10.8] | 0.180 |

## Equivalence (TOST, margin +/-5.0 pts)


## Pooled fine-tuning effect (GSM8K + HotpotQA)

Discordant pairs: INT4 correct / QLoRA wrong = 7; reverse = 2. Exact McNemar **p = 0.180**.

## Tool-call depth (Wilcoxon signed-rank) -- primary Q3 evidence

| Benchmark | Comparison | mean A | mean B | p |
|---|---|---|---|---|
| hotpotqa | int4 vs int4_qlora | 1.48 | 1.13 | 0.0000 |

## Latency

| Benchmark | Config | Mean (s) | Median (s) |
|---|---|---|---|
| hotpotqa | int4 | 8.24 | 7.23 |
| hotpotqa | int4_qlora | 6.51 | 5.52 |
