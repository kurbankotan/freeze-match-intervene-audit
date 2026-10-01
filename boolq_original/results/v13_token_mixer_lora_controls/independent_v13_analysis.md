# Independent V13 verification

Result files verified: 16/16.

## Held-out BoolQ

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Head only | 64.45 ± 0.17 | 60.47 ± 0.19 | 60.65 ± 0.19 |
| Token-wise MLP | 78.66 ± 0.28 | 76.77 ± 0.42 | 77.06 ± 0.36 |
| Token mixer | 83.08 ± 0.29 | 83.01 ± 0.22 | 82.34 ± 0.27 |
| LRAA | 84.10 ± 0.72 | 83.89 ± 0.53 | 83.34 ± 0.67 |
| LoRA r=11 | 88.12 ± 0.12 | 87.79 ± 0.10 | 87.48 ± 0.10 |

## Human-validated 58-pair contrast set

| Model | O accuracy | A accuracy | B accuracy | Contrast consistency | Flip rate |
|---|---:|---:|---:|---:|---:|
| Token-wise MLP | 84.48 | 93.10 | 77.01 | 70.11 | 70.11 |
| Token mixer | 91.38 | 92.53 | 84.48 | 77.01 | 77.01 |
| LRAA | 93.68 | 95.40 | 81.61 | 77.01 | 77.01 |
| LoRA r=11 | 91.95 | 98.85 | 82.18 | 81.03 | 81.03 |

## Prespecified token-mixing test

LRAA minus token mixer balanced accuracy: 0.88 pp (seed-level 95% t CI -0.83 to 2.60); decision: FAIL.
