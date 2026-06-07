# MVTec AD Dataset Overview

MVTec AD is a widely used anomaly detection benchmark for industrial inspection. It is suitable for this repository because it includes both object and texture categories, image-level anomaly labels, and pixel-level masks for localization analysis.

## Categories

- object-like: bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper
- texture-like: carpet, grid, leather, tile, wood

## Why It Fits This Repository

- defect examples are heterogeneous
- normal-only training is realistic for anomaly detection research
- localization masks enable segmentation evaluation
- the 15-category scope prevents overfitting the project story to one easy class

## Public Scope Boundary

This repository stops at MVTec AD. It does not make claims based on private validation sets or internal inspection imagery.
