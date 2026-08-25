# Statistical analysis

## Accuracy with Wilson 95% CIs

| Benchmark | Config | Correct | Accuracy | 95% CI |
|---|---|---|---|---|
| custom50 | fp16 | 49/50 | 98.0% | [89.5, 99.6] |
| custom50 | int4 | 49/50 | 98.0% | [89.5, 99.6] |
| custom50 | int4_qlora | 49/50 | 98.0% | [89.5, 99.6] |
| gsm8k | fp16 | 71/100 | 71.0% | [61.5, 79.0] |
| gsm8k | int4 | 64/93 | 68.8% | [58.8, 77.3] |
| gsm8k | int4_qlora | 65/100 | 65.0% | [55.3, 73.6] |
| hotpotqa | fp16 | 47/100 | 47.0% | [37.5, 56.7] |
| hotpotqa | int4 | 44/100 | 44.0% | [34.7, 53.8] |
| hotpotqa | int4_qlora | 38/100 | 38.0% | [29.1, 47.8] |

## McNemar exact tests (paired)

| Benchmark | Comparison | Discordant | Diff (pts) | 95% CI | p |
|---|---|---|---|---|---|
| custom50 | fp16 vs int4 | 0/0 | +0.0 | [+0.0, +0.0] | 1.000 |
| custom50 | fp16 vs int4_qlora | 1/1 | +0.0 | [-5.5, +5.5] | 1.000 |
| custom50 | int4 vs int4_qlora | 1/1 | +0.0 | [-5.5, +5.5] | 1.000 |
| gsm8k | fp16 vs int4 | 8/5 | +3.2 | [-4.3, +10.8] | 0.581 |
| gsm8k | fp16 vs int4_qlora | 12/6 | +6.0 | [-2.2, +14.2] | 0.238 |
| gsm8k | int4 vs int4_qlora | 8/4 | +4.3 | [-2.9, +11.5] | 0.388 |
| hotpotqa | fp16 vs int4 | 10/7 | +3.0 | [-5.1, +11.1] | 0.629 |
| hotpotqa | fp16 vs int4_qlora | 14/5 | +9.0 | [+0.6, +17.4] | 0.064 |
| hotpotqa | int4 vs int4_qlora | 7/1 | +6.0 | [+0.6, +11.4] | 0.070 |

## Equivalence (TOST, margin +/-5.0 pts)

- **custom50, FP16 vs INT4**: diff +0.0 pts, CI [+0.0, +0.0] -> EQUIVALENT within +/-5.0 pts
- **gsm8k, FP16 vs INT4**: diff +3.2 pts, CI [-4.3, +10.8] -> NOT equivalent within +/-5.0 pts (CI too wide)
- **hotpotqa, FP16 vs INT4**: diff +3.0 pts, CI [-5.1, +11.1] -> NOT equivalent within +/-5.0 pts (CI too wide)

## Pooled fine-tuning effect (GSM8K + HotpotQA)

Discordant pairs: INT4 correct / QLoRA wrong = 15; reverse = 5. Exact McNemar **p = 0.041**.

## Tool-call depth (Wilcoxon signed-rank) -- primary Q3 evidence

| Benchmark | Comparison | mean A | mean B | p |
|---|---|---|---|---|
| custom50 | fp16 vs int4 | n/a | n/a | **tool_calls not recorded** |
| custom50 | fp16 vs int4_qlora | n/a | n/a | **tool_calls not recorded** |
| custom50 | int4 vs int4_qlora | n/a | n/a | **tool_calls not recorded** |
| gsm8k | fp16 vs int4 | n/a | n/a | **tool_calls not recorded** |
| gsm8k | fp16 vs int4_qlora | n/a | n/a | **tool_calls not recorded** |
| gsm8k | int4 vs int4_qlora | n/a | n/a | **tool_calls not recorded** |
| hotpotqa | fp16 vs int4 | n/a | n/a | **tool_calls not recorded** |
| hotpotqa | fp16 vs int4_qlora | n/a | n/a | **tool_calls not recorded** |
| hotpotqa | int4 vs int4_qlora | n/a | n/a | **tool_calls not recorded** |

## Latency

| Benchmark | Config | Mean (s) | Median (s) |
|---|---|---|---|
| custom50 | fp16 | 3.58 | 3.25 |
| custom50 | int4 | 6.96 | 5.85 |
| custom50 | int4_qlora | 7.37 | 6.65 |
| gsm8k | fp16 | 7.32 | 6.20 |
| gsm8k | int4 | 12.19 | 10.90 |
| gsm8k | int4_qlora | 14.31 | 13.10 |
| hotpotqa | fp16 | 5.46 | 5.10 |
| hotpotqa | int4 | 9.03 | 8.20 |
| hotpotqa | int4_qlora | 7.30 | 6.25 |
