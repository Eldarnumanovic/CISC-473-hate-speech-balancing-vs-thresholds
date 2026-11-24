# Validation checkpoint-selection sensitivity

Original rule: maximum validation macro-F1 at threshold 0.5. Alternative: maximum validation average precision. Exact epoch ties favor the earlier epoch. Both rules tune thresholds on the selected epoch using the original grid and tie rules. Values are mean ± sample SD across three seeds, expressed in percent. This is a validation sensitivity analysis, not a test evaluation or a change to the frozen primary checkpoints.

AP selection changes the epoch in 9 of 18 runs.

| Model | Strategy | Selection | Tuned macro-F1 | AP | Δ tuned F1 vs matched unweighted |
|---|---|---|---:|---:|---:|
| bert | unweighted | f1_at_05 | 78.11 ± 1.04 | 82.35 ± 0.46 | +0.00 ± 0.00 |
| bert | unweighted | average_precision | 78.28 ± 0.53 | 82.98 ± 1.00 | +0.00 ± 0.00 |
| bert | weighted | f1_at_05 | 78.72 ± 1.27 | 83.08 ± 1.02 | +0.61 ± 1.54 |
| bert | weighted | average_precision | 78.96 ± 0.85 | 83.30 ± 0.66 | +0.68 ± 0.60 |
| bert | oversampling | f1_at_05 | 77.34 ± 0.46 | 80.98 ± 0.58 | -0.77 ± 1.36 |
| bert | oversampling | average_precision | 79.01 ± 0.69 | 82.85 ± 0.82 | +0.74 ± 0.48 |
| hatebert | unweighted | f1_at_05 | 78.13 ± 0.85 | 82.73 ± 0.74 | +0.00 ± 0.00 |
| hatebert | unweighted | average_precision | 78.22 ± 0.77 | 82.80 ± 0.69 | +0.00 ± 0.00 |
| hatebert | weighted | f1_at_05 | 78.05 ± 0.89 | 82.71 ± 0.76 | -0.09 ± 0.28 |
| hatebert | weighted | average_precision | 78.05 ± 0.89 | 82.71 ± 0.76 | -0.17 ± 0.41 |
| hatebert | oversampling | f1_at_05 | 77.43 ± 1.22 | 81.05 ± 1.22 | -0.71 ± 0.37 |
| hatebert | oversampling | average_precision | 78.12 ± 0.80 | 81.95 ± 0.45 | -0.10 ± 0.62 |

Increasing validation AP under AP selection is expected by construction. Neither selection rule provides unbiased estimates on the validation set used to select it. Saved predictions permit this comparison but cannot recreate unsaved epoch weights for test or perturbation evaluation.
