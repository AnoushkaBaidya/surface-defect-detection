# Project Overview

This repository centers one practical industrial inspection question: what defect detection workflow is most reliable when normal images are plentiful but labeled defect examples are sparse, incomplete, or unrepresentative?

The first answer most teams try is a supervised classifier. That baseline matters, so this project kept ResNet-18 as the reference point. The benchmark showed why that was not enough. Aggregate AUROC looked acceptable in places, but recall and false-negative behavior were not strong enough for open-set industrial inspection.

That result shifted the focus toward normal-only anomaly detection. PaDiM, PatchCore, and FastFlow each improved robustness under the MVTec AD setting, with PatchCore emerging as the strongest pure anomaly model overall. The tradeoff was clear as well: the highest-quality anomaly model was also the heaviest and slowest.

Vision-language models were explored next, not as replacements for anomaly detection, but as an additional semantic signal. WinCLIP did contribute useful recall behavior, but its latency made it difficult to position as a standalone production choice in this benchmark.

That led naturally to hybrid fusion. The most effective setup did not put VLMs in charge. Instead, the best result came from letting PatchCore remain the dominant signal and giving WinCLIP a smaller auxiliary weight. That design improved aggregate recall and false-negative rate without pretending that semantic prompting alone solves industrial anomaly detection.

The core lesson is simple: in inspection systems, recall and false-negative rate are operationally more important than a single headline AUROC value. The benchmark therefore treats model selection as an engineering tradeoff problem, not just a ranking exercise.

The repository keeps that story intentionally narrow:

- MVTec AD only
- the benchmark outputs produced here
- honest discussion of latency, size, and runtime constraints
- no claims based on private or internal validation work

There are a few adjacent folders for deployment, API, and container scaffolding, but the completed work in this repo is the MVTec benchmark itself plus the measured runtime path needed to interpret serving tradeoffs.
