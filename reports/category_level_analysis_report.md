# Category-Level Analysis Report

## Aggregate Winner Distribution

Best classification AUROC by category:

- PatchCore: 12 categories
- PaDiM: 1 category (`carpet`)
- ResNet-18: 2 categories (`screw`, `zipper`)

## Why This Matters

The category-level view shows why a single summary metric is not enough:

- some categories allow the supervised baseline to look competitive
- texture-heavy categories can change the best local ranking
- the aggregate ranking still favors anomaly detection because it is more consistent

## Contrast Example

ResNet-18 struggled badly on several categories:

- `toothbrush` recall 0.0435
- `grid` recall 0.2093
- `transistor` recall 0.4000

PatchCore’s weakest recall categories were still much stronger:

- `pill` recall 0.9291
- `transistor` recall 0.9500
- `grid` recall 0.9649

## Conclusion

Category variation reinforces the repository’s main design choice: rank models by stable aggregate defect-miss behavior, not by a few locally favorable categories.
