# Human-validated polarity analysis

This directory re-scores the archived CUDA-parity raw predictions on the 58
contrast items that both survive the independent human review and are marked
usable in all O/A/B conditions after adjudication.

No model is retrained and no prediction is recomputed. The wrapper
`../../scripts/rescore_human_validated_subset.py` imports the unchanged metric
functions from `../../scripts/score_contrast_set.py`, filters by
`final_item_usable == yes`, and repeats the 100,000-replicate paired
item-cluster and two-way item/seed bootstraps.

Headline result: LRAA has 77.01% mean contrast consistency versus 70.12% for the
matched MLP, a +6.90-point difference. The item-cluster 95% interval is
[-4.60, +18.97], so the prespecified comparative Q1 decision remains FAIL.
LRAA's 77.01% consistency and 77.01% flip rate satisfy the absolute Q2 rule.

