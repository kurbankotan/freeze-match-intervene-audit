# Full Fine-Tuning Can Obscure Adapter Contribution

This repository is the reproducibility package for **Full Fine-Tuning Can Obscure Adapter Contribution: A Frozen-Backbone, Parameter-Matched Study on BoolQ**.

The tested module is described conservatively as a **Low-Rank Attention Adapter (LRAA)**: a residual rank-256, eight-head self-attention branch applied to the complete final token sequence of a frozen DeBERTa-v3-large encoder. Attention, low-rank parameterization, adapters, and backbone freezing are established ideas. The principal contribution is methodological: demonstrating why context sensitivity under full fine-tuning does not establish adapter contribution, and how backbone freezing plus parameter matching makes that contribution testable.

## Main verified results

The frozen-backbone BoolQ experiment uses a passage-grouped split, seeds 13/42/71, three epochs, batch size 2, gradient accumulation 8, learning rate 2e-4, and identical trainable capacity for LRAA and the MLP adapter (1,050,625 adapter parameters plus a shared 1,025-parameter scorer).

| Quantity | Verified value |
|---|---:|
| LRAA held-out accuracy | 84.0979% |
| LRAA balanced accuracy | 83.8928% |
| LRAA macro-F1 | 83.3442% |
| LRAA minus MLP accuracy | +5.4332 points |
| LRAA minus MLP balanced accuracy | +7.1180 points |
| LRAA minus MLP macro-F1 | +6.2860 points |
| MLP adapter-on minus adapter-off balanced accuracy | +8.8496 points |
| LRAA adapter-on minus adapter-off balanced accuracy | +27.6156 points |
| LRAA minus MLP difference in balanced-accuracy ablation loss | +18.7659 points |
| Validated full-FT held-out balanced accuracy | 88.7490% |
| Validated full-FT adapter-on minus adapter-off balanced accuracy | +0.8068 points |
| Frozen minus validated full-FT balanced-accuracy ablation loss | +26.8088 points |
| Training-regime decision | PASS |
| Human-validated contrast pairs usable in all arms | 58/81 |
| LRAA polarity contrast consistency (validated subset) | 77.01% |
| LRAA polarity flip rate (validated subset) | 77.01% |
| MLP polarity contrast consistency (validated subset) | 70.12% |
| Validated-subset LRAA minus MLP contrast gain | +6.90 points |
| Token-mixer held-out balanced accuracy | 83.0097% |
| LRAA minus token-mixer balanced accuracy | +0.8831 points |
| LRAA minus token-mixer 95% seed interval | [−0.8339, +2.6001] points |
| Prespecified attention-topology decision | FAIL |
| LoRA r=11 held-out balanced accuracy | 87.7852% |
| LRAA minus LoRA r=11 balanced accuracy | −3.8924 points |

The CUDA rescore passed all 45 parity checks across nine model-seed runs. Two independent human reviewers then evaluated all 81 texts under blinded A/B presentation. After adjudication, 58 pairs were usable in all conditions; 18 excluded items can be regenerated and five are permanent drops. Filtering the unchanged raw predictions to those 58 pairs gives a primary 100,000-replicate item-cluster interval of **[-4.60, +18.97]**, with `Pr(gain <= 0) = .131`. The two-way item/seed sensitivity interval is **[-5.17, +18.97]**, with `Pr(gain <= 0) = .145`. Item-majority McNemar discordant counts are 9 versus 3, with exact two-sided `p = .146`. Machine-readable files retain the full computed precision. The original 81-item analysis is retained as an unfiltered audit.

The decision audit is explicit: **Q1 comparative advantage over the token-wise MLP = FAIL** because the validated-subset item-cluster interval includes zero; **Q2 absolute polarity behavior = PASS** because consistency exceeds 45% and flip rate exceeds 50%. With 58 validated item clusters, the approximate 80%-power minimum detectable effect is 16.68–16.73 points, versus the observed 6.90-point gain. The comparative result is therefore inconclusive rather than evidence of equality. The naive pooled interval and the unfiltered 81-item analysis are retained only as audit outputs.

V13 closes the two remaining architectural controls. The exactly parameter-matched non-attention token mixer reaches 83.01% balanced accuracy, while LRAA reaches 83.89%; the +0.88-point paired difference has a seed-level 95% t interval of [−0.83, +2.60]. The prespecified attention-topology rule therefore **fails**, so the data do not separate attention from generic token mixing. A standard LoRA control with rank 11 reaches 87.79% balanced accuracy and exceeds LRAA in every seed. These completed controls reinforce the paper's methodological contribution while ruling out a claim that LRAA is the best tested PEFT method.

All frozen-backbone methods used the same learning rate of `2e-4`; neither LRAA nor LoRA received a dedicated method-specific hyperparameter sweep. The comparison is symmetric under the locked protocol, but it is not evidence that either method has reached its individually tuned optimum.

## Repository layout

```text
checkpoints/                  Nine trained scorer/adapter state dictionaries and SHA-256 manifest
data/                         The 81-pair O/A/B polarity contrast set and dataset card
notebooks/                    Full V9 reproduction/training and CUDA clustered-bootstrap notebooks
results/v9_reproduction/      Machine-readable reproduction, evaluation, and audit outputs
results/cuda_clustered_bootstrap/  CUDA parity, raw predictions, and clustered analyses
results/mlp_adapter_off/      Held-out MLP/LRAA adapter-on/off predictions and paired audits
results/v11_equal_lr_optimization_failure/  Auditable invalid full-backbone run
results/v12_validated_full_finetuning/  Optimization-valid trainable-backbone control
results/v13_token_mixer_lora_controls/  Completed matched mixer, LoRA, adapter-off, and contrast audits
results/human_validation/     Independent first pass, adjudication, and regeneration list
results/human_validated_contrast/  Metrics and clustered inference on the 58 valid pairs
scripts/                      Unchanged metric code plus validated-subset rescore wrapper
```

The completed V13 notebooks add (i) a non-attention token mixer with exactly 1,050,625 adapter parameters and (ii) a standard PEFT LoRA control. `notebooks/lraa_v13_token_mixer_lora_controls_colab.ipynb` trains and checkpoints the six runs. `notebooks/lraa_v13_evaluate_all_checkpoints_colab.ipynb` evaluates already trained checkpoints without retraining and produced the archived V13 tables.

## Naming compatibility

The manuscript uses **LRAA** as a descriptive module label rather than a claim to a new attention family. The completed experiments use the legacy internal identifier `lcca`. To preserve exact checkpoint loading and hash verification, filenames, state-dictionary keys, notebook variables, and machine-readable fields retain `lcca`. In this package, those legacy identifiers refer to LRAA. Do not rename checkpoint keys or model identifiers unless the loaders are updated at the same time.

## Colab use

1. Download the official [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large) model and place its files in:
   `/content/drive/MyDrive/Colab Notebooks/NLP Modeller/deberta-v3-large`
2. Upload `data/contrast_set.jsonl` and `scripts/score_contrast_set.py` to the Colab session when prompted.
3. For the exact full rerun and checkpoint generation, open `notebooks/lraa_v9_reproduction_checkpoints_contrast_colab.ipynb` and run cells from top to bottom. This retrains nine model-seed conditions and can take many hours.
4. For the fast post-training CUDA parity and clustered-bootstrap analysis, open `notebooks/lraa_v9_clustered_bootstrap_cuda_parity_colab.ipynb`, upload the reproduction artifact ZIP when prompted, and run cells from top to bottom. This uses the archived checkpoints and does not retrain.
5. For the held-out MLP/LRAA adapter-on/off audit, open `notebooks/lraa_v9_mlp_adapter_off_audit_colab.ipynb`, upload the repository artifact ZIP when prompted, and run cells from top to bottom. This audit does not retrain. It first requires exact enabled-condition parity and then reports each architecture's ablation loss plus the paired difference between those losses.
6. `notebooks/lraa_v11_matched_full_finetuning_control_colab.ipynb` is retained with its outputs as an optimization failure: applying `2e-4` to the full encoder left the model near chance, so its nominal decision is invalid evidence.
7. Run `notebooks/lraa_v12_validated_full_finetuning_control_colab.ipynb` for the corrected trainable-backbone control. It keeps the V9 adapter/scorer rate at `2e-4`, uses the previously established encoder rate `8e-6`, and applies an optimization-validity gate before any attribution decision.
8. To reproduce the post-adjudication polarity table without a GPU, run `scripts/rescore_human_validated_subset.py` with the final human-validation CSV, the original contrast JSONL, and the archived raw CUDA predictions. It imports the unchanged metric functions and performs no inference or training.
9. Run `notebooks/lraa_v13_token_mixer_lora_controls_colab.ipynb` to retrain the completed parameter-matched token-mixer and standard LoRA controls. It mounts Drive first, clones this public repository automatically, resumes completed model×seed checkpoints from Drive, and downloads one result ZIP. It does not delete the runtime.
10. To evaluate the six archived V13 checkpoints without training, run `notebooks/lraa_v13_evaluate_all_checkpoints_colab.ipynb`. It reconstructs the held-out split, performs adapter-on/off and 58-pair contrast inference, verifies hashes, and downloads the compact evaluation artifact.

The notebooks mount Google Drive at the beginning because Colab requires interactive authorization. GPU selection can be requested by notebook metadata, but Colab itself controls the actual accelerator and High-RAM allocation.

## Environment recorded by the CUDA audit

- GPU: Tesla T4
- PyTorch: 2.11.0+cu128
- Transformers: 4.57.6
- Autocast dtype: `torch.bfloat16`

See `requirements.txt` and `results/cuda_clustered_bootstrap/cuda_run_environment.json` for the reproducibility record. The base encoder weights are not included in this repository.

## Integrity and provenance

- `checkpoints/checkpoint_manifest.csv` records the exact byte size and SHA-256 digest of every checkpoint.
- `results/v9_reproduction/contrast_metric_source_sha256.txt` records the digest of the unchanged metric implementation.
- `results/cuda_clustered_bootstrap/cuda_parity_audit.csv` contains all 45 archived-versus-GPU comparisons.
- Raw CUDA contrast predictions are included for independent reanalysis.
- `results/cuda_clustered_bootstrap/power_sensitivity_analysis.json` records the inputs and full-precision calculations behind the displayed 13.1–13.4-point sensitivity range.
- `results/mlp_adapter_off/enabled_condition_parity_audit.csv` records all 18 exact held-out parity checks before the adapter-off interpretation.
- `results/mlp_adapter_off/difference_of_adapter_gains.csv` reports whether LRAA's adapter-on/off reduction exceeds the corresponding MLP reduction in matched seeds.
- `results/v11_equal_lr_optimization_failure/corrected_interpretation.json` prevents the completed but non-learning V11 run from being mistaken for supporting evidence.
- `results/v12_validated_full_finetuning/optimization_validity.json` records the three prespecified learning checks, and `training_regime_contrast_summary.json` records the gated `PASS` decision.
- `results/human_validation/` preserves the independent normalized judgments, pre-adjudication reliability, final 81-item decisions, and the 18-item regeneration worklist.
- `results/human_validated_contrast/` records the model-independent 58-item filter, recalculated metrics, 100,000-replicate clustered analyses, and full-precision power sensitivity calculation.
- `results/v13_token_mixer_lora_controls/` records all six held-out and contrast evaluations, exact per-seed McNemar tests, the prespecified FAIL decision, environment metadata, source-checkpoint hashes, and an independent recomputation from raw predictions.

Run `python scripts/verify_v13_artifacts.py` from the repository root to verify the archived V13 result files and all six V13 checkpoints against their recorded byte sizes and SHA-256 digests.

## Licenses, data, and model terms

Repository software is distributed under the MIT License. The included polarity contrast data are adaptations of BoolQ and are distributed under CC BY-SA 3.0 with the attribution and change notice in `data/LICENSE.md`. DeBERTa-v3-large weights are not committed; they remain governed by Microsoft’s upstream MIT terms. Third-party libraries retain their own licenses.

## Citation

Citation metadata is provided in `CITATION.cff`. The final repository URL and publication metadata will be added when the release location is fixed. The unpublished manuscript is intentionally not included in this artifact package.
