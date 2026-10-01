# Polarity contrast set

`contrast_set.jsonl` contains 81 contrast identifiers. Each item has three passage arms with the same question:

- **O:** original passage.
- **A:** evidence sentence rewritten while preserving its meaning and gold answer.
- **B:** the same evidence negated so that the gold answer flips.

Both yes/no answer candidates are scored for all three arms. Joint contrast consistency is one only when a checkpoint answers both A and B correctly. The O arm is also required because the metric script uses it to measure edit cost and related diagnostics.

The set is label-imbalanced and was verified by one language-model annotator rather than blinded independent human judges. It is therefore an audit set, not a definitive benchmark. The comparative Q1 decision fails because the item-cluster interval includes zero; the absolute Q2 polarity criteria pass. The study is underpowered for the observed seven-point comparative gain.

The metric implementation in `scripts/score_contrast_set.py` is preserved unchanged. Its archived SHA-256 digest is recorded in `results/v9_reproduction/contrast_metric_source_sha256.txt`.

No redistribution license was specified in the supplied materials. See the repository-level `LICENSE_SELECTION_REQUIRED.md` before public archival release.
