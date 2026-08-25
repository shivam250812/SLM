"""Statistical analysis for the paper.

Because every configuration runs the same task list in the same order, the
results are PAIRED. Paired tests (McNemar for accuracy, Wilcoxon signed-rank
for tool-call depth) are substantially more powerful than the independent
two-proportion comparison that a table of raw accuracies implies.

Produces every statistic reported in the Results section:
  * Wilson 95% confidence intervals per run
  * McNemar exact tests for every pairwise config comparison
  * paired difference CIs
  * a pooled McNemar test of the fine-tuning effect across both standard
    benchmarks
  * Wilcoxon signed-rank on per-task tool_calls -- the primary evidence for
    search-horizon truncation
  * TOST equivalence test for the "quantization is free" claim

Usage
-----
    python analysis/stats.py --results results --out analysis/stats_output.md
"""
import argparse
import itertools
import math
import os

import pandas as pd
from scipy import stats

CONFIGS = ["fp16", "int4", "int4_qlora", "int4_zero"]
BENCHMARKS = ["custom50", "gsm8k", "hotpotqa"]


def load(results_dir):
    runs = {}
    for bm in BENCHMARKS:
        for cfg in CONFIGS:
            path = os.path.join(results_dir, f"{bm}_{cfg}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path).set_index("id").sort_index()
                df["correct"] = df["correct"].astype(bool)
                runs[(bm, cfg)] = df
    return runs


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (centre - half), 100 * (centre + half)


def mcnemar(a, b):
    """Exact (binomial) McNemar test on paired boolean outcomes."""
    idx = a.index.intersection(b.index)
    b01 = int((a.loc[idx, "correct"] & ~b.loc[idx, "correct"]).sum())
    b10 = int((~a.loc[idx, "correct"] & b.loc[idx, "correct"]).sum())
    n = b01 + b10
    p = stats.binomtest(b01, n, 0.5).pvalue if n else 1.0
    diff = (b01 - b10) / len(idx)
    se = math.sqrt(max(n - (b01 - b10) ** 2 / len(idx), 0)) / len(idx)
    return dict(n=len(idx), b01=b01, b10=b10, p=p,
                diff=100 * diff, lo=100 * (diff - 1.96 * se),
                hi=100 * (diff + 1.96 * se))


def tost_equivalence(a, b, margin_pts=5.0):
    """Two one-sided tests. Rejects the null of a difference LARGER than
    `margin_pts` in either direction. This is the test that licenses an
    'essentially free' claim; a non-significant difference test does not."""
    res = mcnemar(a, b)
    if res["hi"] < margin_pts and res["lo"] > -margin_pts:
        verdict = f"EQUIVALENT within +/-{margin_pts} pts"
    else:
        verdict = f"NOT equivalent within +/-{margin_pts} pts (CI too wide)"
    return res, verdict


def wilcoxon_calls(a, b):
    """Paired Wilcoxon signed-rank on per-task tool-call counts.

    Returns None when tool_calls is absent or all-missing -- which is the case
    for the reconstructed results in results_reconstructed/, where per-task
    tool-call counts were never printed and cannot be recovered."""
    if "tool_calls" not in a.columns or "tool_calls" not in b.columns:
        return None
    idx = a.index.intersection(b.index)
    x = pd.to_numeric(a.loc[idx, "tool_calls"], errors="coerce")
    y = pd.to_numeric(b.loc[idx, "tool_calls"], errors="coerce")
    keep = x.notna() & y.notna()
    x, y = x[keep], y[keep]
    if len(x) == 0:
        return None
    if (x - y).abs().sum() == 0:
        return dict(n=len(x), mean_a=float(x.mean()), mean_b=float(y.mean()), p=1.0)
    res = stats.wilcoxon(x, y)
    return dict(n=len(x), mean_a=float(x.mean()), mean_b=float(y.mean()),
                stat=float(res.statistic), p=float(res.pvalue))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="analysis/stats_output.md")
    ap.add_argument("--margin", type=float, default=5.0,
                    help="equivalence margin in accuracy points for TOST")
    args = ap.parse_args()

    runs = load(args.results)
    if not runs:
        raise SystemExit(f"no result CSVs found in {args.results}/")

    L = ["# Statistical analysis", ""]

    L += ["## Accuracy with Wilson 95% CIs", "",
          "| Benchmark | Config | Correct | Accuracy | 95% CI |",
          "|---|---|---|---|---|"]
    for (bm, cfg), df in sorted(runs.items()):
        k, n = int(df["correct"].sum()), len(df)
        lo, hi = wilson(k, n)
        L.append(f"| {bm} | {cfg} | {k}/{n} | {100*k/n:.1f}% | [{lo:.1f}, {hi:.1f}] |")

    L += ["", "## McNemar exact tests (paired)", "",
          "| Benchmark | Comparison | Discordant | Diff (pts) | 95% CI | p |",
          "|---|---|---|---|---|---|"]
    for bm in BENCHMARKS:
        present = [c for c in CONFIGS if (bm, c) in runs]
        for a, b in itertools.combinations(present, 2):
            r = mcnemar(runs[(bm, a)], runs[(bm, b)])
            L.append(f"| {bm} | {a} vs {b} | {r['b01']}/{r['b10']} | "
                     f"{r['diff']:+.1f} | [{r['lo']:+.1f}, {r['hi']:+.1f}] | {r['p']:.3f} |")

    L += ["", f"## Equivalence (TOST, margin +/-{args.margin} pts)", ""]
    for bm in BENCHMARKS:
        if (bm, "fp16") in runs and (bm, "int4") in runs:
            r, verdict = tost_equivalence(runs[(bm, "fp16")], runs[(bm, "int4")],
                                          args.margin)
            L.append(f"- **{bm}, FP16 vs INT4**: diff {r['diff']:+.1f} pts, "
                     f"CI [{r['lo']:+.1f}, {r['hi']:+.1f}] -> {verdict}")

    L += ["", "## Pooled fine-tuning effect (GSM8K + HotpotQA)", ""]
    tot01 = tot10 = 0
    for bm in ("gsm8k", "hotpotqa"):
        if (bm, "int4") in runs and (bm, "int4_qlora") in runs:
            r = mcnemar(runs[(bm, "int4")], runs[(bm, "int4_qlora")])
            tot01 += r["b01"]
            tot10 += r["b10"]
    if tot01 + tot10:
        p = stats.binomtest(tot01, tot01 + tot10, 0.5).pvalue
        L.append(f"Discordant pairs: INT4 correct / QLoRA wrong = {tot01}; "
                 f"reverse = {tot10}. Exact McNemar **p = {p:.3f}**.")

    L += ["", "## Tool-call depth (Wilcoxon signed-rank) -- primary Q3 evidence", "",
          "| Benchmark | Comparison | mean A | mean B | p |", "|---|---|---|---|---|"]
    for bm in BENCHMARKS:
        present = [c for c in CONFIGS if (bm, c) in runs]
        for a, b in itertools.combinations(present, 2):
            r = wilcoxon_calls(runs[(bm, a)], runs[(bm, b)])
            if r is None:
                L.append(f"| {bm} | {a} vs {b} | n/a | n/a | "
                         "**tool_calls not recorded** |")
                continue
            L.append(f"| {bm} | {a} vs {b} | {r['mean_a']:.2f} | {r['mean_b']:.2f} "
                     f"| {r['p']:.4f} |")

    L += ["", "## Latency", "", "| Benchmark | Config | Mean (s) | Median (s) |",
          "|---|---|---|---|"]
    for (bm, cfg), df in sorted(runs.items()):
        L.append(f"| {bm} | {cfg} | {df['latency_s'].mean():.2f} "
                 f"| {df['latency_s'].median():.2f} |")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    text = "\n".join(L) + "\n"
    open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
