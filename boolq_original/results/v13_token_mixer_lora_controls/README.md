# V13 token-mixer and LoRA controls

This directory contains the completed V13 evaluation of two frozen-backbone controls on the exact V9 BoolQ split and seeds.

- `token_mixer` is a non-attention dilated depthwise-convolution adapter with exactly 1,050,625 adapter parameters, equal to LRAA.
- `lora_r11` is PEFT LoRA with rank 11 on all 24 DeBERTa `query_proj` and 24 `value_proj` modules. It has 1,081,344 LoRA parameters, 2.9210% more trainable parameters than LRAA before adding the common 1,025-parameter scorer.

All six source checkpoints were evaluated on the 3,270-example held-out BoolQ split, with and without the trained adaptation path, and on the 58-pair human-validated polarity set. `result_file_manifest_sha256.csv` verifies every supplied result file. `checkpoint_manifest_sha256.csv` verifies the six checkpoint files stored under `checkpoints/v13_controls/`.

## Held-out results

Values are mean ± sample standard deviation across seeds 13, 42, and 71.

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Token mixer | 83.08 ± 0.29 | 83.01 ± 0.22 | 82.34 ± 0.27 |
| LRAA | 84.10 ± 0.72 | 83.89 ± 0.53 | 83.34 ± 0.67 |
| LoRA r=11 | 88.12 ± 0.12 | 87.79 ± 0.10 | 87.48 ± 0.10 |

The prespecified LRAA-versus-token-mixer decision is **FAIL**. LRAA is higher in all three seeds, but its mean balanced-accuracy advantage is only +0.88 percentage point and its paired seed-level 95% t interval is [−0.83, +2.60]. This result does not isolate attention topology from generic token mixing.

LoRA exceeds LRAA by 4.03 accuracy points, 3.89 balanced-accuracy points, and 4.13 macro-F1 points. The paired seed-level 95% intervals for LRAA minus LoRA are [−5.86, −2.19], [−5.06, −2.72], and [−5.82, −2.44] points, respectively. Exact per-seed McNemar results are stored in `heldout_mcnemar.csv`.

The locked protocol applied the same learning rate of `2e-4` to LRAA, the token mixer, and LoRA. No dedicated method-specific sweep was conducted for either LRAA or LoRA, so the comparison is symmetric but should not be interpreted as each method's tuned optimum.

## Human-validated polarity results

| Model | O accuracy | A accuracy | B accuracy | Contrast consistency | Flip rate |
|---|---:|---:|---:|---:|---:|
| Token mixer | 91.38 | 92.53 | 84.48 | 77.01 | 77.01 |
| LRAA | 93.68 | 95.40 | 81.61 | 77.01 | 77.01 |
| LoRA r=11 | 91.95 | 98.85 | 82.18 | 81.03 | 81.03 |

LRAA and the token mixer have identical mean contrast consistency on the 58-pair validated set. LoRA is descriptively 4.02 points higher than LRAA, but the 58-item clustered interval for LRAA minus LoRA is [−12.64, +4.02] points. These small-set comparisons are descriptive and do not establish topology superiority.

`independent_v13_analysis.json` and the associated CSV/Markdown files independently recompute the principal results from raw predictions. Seed-level intervals describe optimization-seed variation only; item-cluster bootstrap intervals resample held-out examples or contrast identifiers while retaining paired seed repeats.
