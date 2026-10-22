# Preliminary validation results

All values below are percentages, reported as mean ± sample standard deviation across seeds 42, 123, and 999 (n=3). Average precision is also multiplied by 100. These are validation results: checkpoints and tuned thresholds were selected on this same split, so these are not unbiased test estimates. AP is threshold-independent and is not trapezoidal PR-AUC.

## Default thresholds

| Model | Strategy | Macro-F1 | Hate precision | Hate recall | FPR | AP |
|---|---|---:|---:|---:|---:|---:|
| bert | unweighted | 77.55 ± 1.16 | 73.07 ± 3.69 | 77.28 ± 7.07 | 21.58 ± 6.11 | 82.35 ± 0.46 |
| bert | weighted | 76.98 ± 1.28 | 70.27 ± 3.72 | 81.89 ± 8.90 | 26.29 ± 7.58 | 83.08 ± 1.02 |
| bert | oversampling | 77.07 ± 0.60 | 71.52 ± 1.56 | 78.06 ± 1.96 | 23.21 ± 2.29 | 80.98 ± 0.58 |
| hatebert | unweighted | 77.14 ± 0.38 | 73.14 ± 3.55 | 76.03 ± 8.79 | 21.23 ± 6.55 | 82.73 ± 0.74 |
| hatebert | weighted | 77.20 ± 0.59 | 71.87 ± 3.15 | 78.69 ± 8.72 | 23.33 ± 6.43 | 82.71 ± 0.76 |
| hatebert | oversampling | 76.80 ± 1.27 | 70.10 ± 2.41 | 80.56 ± 1.86 | 25.71 ± 3.54 | 81.05 ± 1.22 |

## Tuned thresholds

| Model | Strategy | Macro-F1 | Hate precision | Hate recall | FPR | AP |
|---|---|---:|---:|---:|---:|---:|
| bert | unweighted | 78.11 ± 1.04 | 72.91 ± 1.95 | 78.84 ± 4.81 | 21.93 ± 3.34 | 82.35 ± 0.46 |
| bert | weighted | 78.72 ± 1.27 | 73.62 ± 1.34 | 79.23 ± 3.12 | 21.18 ± 1.62 | 83.08 ± 1.02 |
| bert | oversampling | 77.34 ± 0.46 | 72.11 ± 1.42 | 77.75 ± 2.31 | 22.45 ± 2.20 | 80.98 ± 0.58 |
| hatebert | unweighted | 78.13 ± 0.85 | 73.65 ± 1.78 | 77.60 ± 6.42 | 20.83 ± 3.73 | 82.73 ± 0.74 |
| hatebert | weighted | 78.05 ± 0.89 | 73.64 ± 0.90 | 76.97 ± 1.73 | 20.54 ± 0.88 | 82.71 ± 0.76 |
| hatebert | oversampling | 77.43 ± 1.22 | 70.42 ± 0.46 | 81.81 ± 3.38 | 25.60 ± 0.53 | 81.05 ± 1.22 |

## Tuned strategies minus tuned unweighted baseline

Differences are paired by model and seed; mean ± sample standard deviation, in percentage points. Positive FPR differences mean more false positives. Three seeds describe variability but do not establish statistical significance.

| Model | Strategy | Δ macro-F1 | Δ hate recall | Δ FPR | Δ AP |
|---|---|---:|---:|---:|---:|
| bert | weighted | +0.61 ± 1.54 | +0.39 ± 7.93 | -0.76 ± 3.73 | +0.74 ± 0.67 |
| bert | oversampling | -0.77 ± 1.36 | -1.09 ± 3.09 | +0.52 ± 3.71 | -1.37 ± 0.78 |
| hatebert | weighted | -0.09 ± 0.28 | -0.62 ± 4.74 | -0.29 ± 3.89 | -0.02 ± 0.09 |
| hatebert | oversampling | -0.71 ± 0.37 | +4.22 ± 3.04 | +4.77 ± 3.22 | -1.67 ± 0.69 |
