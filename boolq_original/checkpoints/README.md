# Checkpoints

The repository contains the nine frozen-V9 scorer/adapter checkpoints in this directory and the six completed V13 control checkpoints under `v13_controls/`.

V13 includes `token_mixer_seed{13,42,71}.pt` and `lora_r11_seed{13,42,71}.pt`. Their byte sizes and SHA-256 digests are recorded in `../results/v13_token_mixer_lora_controls/checkpoint_manifest_sha256.csv`. The six files were checked against that manifest before publication.

This directory contains nine selected scorer/adapter state dictionaries:

- head-only: seeds 13, 42, and 71
- matched MLP: seeds 13, 42, and 71
- LRAA: seeds 13, 42, and 71

The LRAA files retain the legacy names `lcca_seed13.pt`, `lcca_seed42.pt`, and `lcca_seed71.pt` because the exact completed notebooks and state dictionaries use `lcca` as their internal identifier. Renaming them without modifying the loader would break exact compatibility.

Use `checkpoint_manifest.csv` to verify byte sizes and SHA-256 digests before evaluation.
