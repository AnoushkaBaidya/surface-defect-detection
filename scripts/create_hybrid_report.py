"""
Create the hybrid benchmark report.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "artifacts/benchmarks/hybrid"


def main() -> int:
    per_category = pd.read_csv(BASE_DIR / "per_category_results.csv")
    aggregate = pd.read_csv(BASE_DIR / "aggregate_results.csv")

    best_rows = aggregate.head(10)

    report = [
        "# Hybrid Benchmark Report",
        "",
        "## Purpose",
        "",
        "Evaluate whether combining classical anomaly scores with WinCLIP VLM scores improves classification robustness, recall, or false-negative reduction.",
        "",
        "## Fusion Methods",
        "",
        "- PatchCore + WinCLIP k=1",
        "- PatchCore + WinCLIP k=4",
        "- FastFlow + WinCLIP k=1",
        "- FastFlow + WinCLIP k=4",
        "",
        "Weights tested:",
        "",
        "- 0.7 anomaly / 0.3 VLM",
        "- 0.5 anomaly / 0.5 VLM",
        "- 0.3 anomaly / 0.7 VLM",
        "",
        "## Important Evaluation Note",
        "",
        "Thresholded F1/recall/FNR use diagnostic max-F1 thresholding. AUROC is the primary threshold-independent comparison.",
        "",
        "## Top Aggregate Hybrid Results",
        "",
        best_rows.to_markdown(index=False),
        "",
        "## Engineering Interpretation",
        "",
        "- Hybrid fusion is valuable only if it improves recall/FNR or robustness over PatchCore/FastFlow alone.",
        "- If fusion increases false positives or latency without improving missed defects, the production system should prefer the simpler model.",
        "- A negative result is still valuable because it proves that VLM fusion was evaluated rather than assumed beneficial.",
        "",
        "## Full Per-Category Result Count",
        "",
        f"Rows evaluated: {len(per_category)}",
        "",
    ]

    output_path = BASE_DIR / "hybrid_benchmark_report.md"
    output_path.write_text("\n".join(report), encoding="utf-8")

    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
