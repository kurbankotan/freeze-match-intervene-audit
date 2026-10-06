# freeze-match-intervene-audit

Code, manifests and raw results for the manuscript **"Full Fine-Tuning Can Obscure Adapter Contribution: A Freeze–Match–Intervene Audit of Attention and Bottleneck Adapters"** (Kurban Kotan, Manisa Celal Bayar University).

The audit asks three separate questions about an inserted module: does the fitted system depend on it (same-checkpoint intervention in three states: `on`, `norm_only`, `off`), does it beat exactly parameter-matched alternatives, and do the answers change when the backbone is frozen versus fully fine-tuned. Two adapter designs are audited on DeBERTa-v3-large: a Low-Rank Attention Adapter (LRAA; legacy identifier `lcca` in file names) and standard Houlsby bottleneck adapters, with BoolQ as the primary task, WinoGrande as a second task, and Qwen3-1.7B-Base as a second backbone.

## Contents of this repository

| Experiment | Notebook | Manifest | Results directory |
|---|---|---|---|
| D-NORM-1: three-state intervention on the nine frozen BoolQ checkpoints | `notebooks/fmi_nb1_dnorm1_frozen_decomposition_colab.ipynb` | `manifests/FMI_manifests_and_decision_rules_v1_2.md`, Part A | `results/fmi_dnorm1_frozen_results_20260924T181240Z/` |
| FT-RERUN-1 + D2: retrained full fine-tuning (three states) and no-adapter full fine-tuning | `notebooks/fmi_nb2_fullft_rerun_d2_colab.ipynb` | v1.2, Part A.8 | `results/fmi_nb2_fullft_results_20260925T081859Z/` |
| FT-TRAJ-1: two further full-fine-tuning trajectories with per-epoch evaluation | `notebooks/fmi_nb2b_ft_traj1_per_epoch_colab.ipynb` | `manifests/FMI_manifest_v1_3_amendment_FT-TRAJ-1.md` | `results/fmi_nb2b_ft_traj1_results_20260925T111023Z/` |
| Q-HELDOUT-1: Qwen3-1.7B-Base hash pre-check and single-look held-out evaluation | `notebooks/fmi_nb3_qwen_hash_precheck_cpu_colab.ipynb`, `notebooks/fmi_nb4_qwen_heldout_single_look_colab.ipynb` | v1.2, Part B | `results/fmi_nb4_qwen_heldout_results_20260925T164440Z/` |
| W-1: WinoGrande, five seeds × seven methods | `notebooks/fmi_nb5_winogrande_second_task_colab.ipynb` | `manifests/FMI_manifest_W1_second_task_WinoGrande.md` | `results/fmi_nb5_winogrande_results_20260930T222505Z/` |
| H-1: standard Houlsby adapters (48 bottleneck adapters), frozen vs full fine-tuning, BoolQ and WinoGrande | `notebooks/fmi_nb7_houlsby_adapters_colab.ipynb` | `manifests/FMI_manifest_H1_standard_adapters.md` | `results/fmi_nb7_houlsby_boolq_results_20261005T162343Z/`, `results/fmi_nb7_houlsby_winogrande_results_20261006T060824Z/` |

Each results directory holds per-row predictions with probabilities, aggregate tables, bootstrap and McNemar outputs, gate reports, `run_environment.json` and `FILE_MANIFEST_SHA256.csv`. `MANIFEST_LOCK.md` records the SHA-256 of each manifest and every annotation made during execution (including a STOP-gate event and a numerics override, both described in the paper).

## Original BoolQ study

`boolq_original/` holds the archived BoolQ study unchanged: training and evaluation notebooks, the fifteen frozen-backbone checkpoints (`boolq_original/checkpoints/`), raw held-out and contrast predictions, result tables and SHA-256 manifests (`boolq_original/results/`), and the polarity contrast set with its generation and screening records. The revision notebooks read these archives from this directory.

## Not in this repository

- Large trained states are not redistributed and are identified by SHA-256 in the result files: DeBERTa-v3-large full fine-tuning (about 1.7 GB each) and Qwen3-1.7B-Base (about 6.9 GB each).
- Datasets (public, not redistributed): BoolQ via the Hugging Face dataset `google/boolq` (revision `35b264d`); WinoGrande 1.1 from `https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip` (SHA-256 `3619ab10…`). Models: DeBERTa-v3-large and Qwen3-1.7B-Base (revision `ea980cb0…`) from the Hugging Face Hub, under their upstream terms.

## Reproducing

The notebooks run on Google Colab (A100 recommended for training; a T4 suffices for the frozen decomposition). Each notebook pins its dependencies, verifies checkpoint hashes before loading, writes per-run progress to Google Drive and resumes after interruption. Prespecified decision rules and the outcome maps are in the manifests; the paper reports every rule as met or not met.

## License and citation

Software: MIT License (see `LICENSE`). Derived polarity contrast texts: CC BY-SA 3.0 with BoolQ attribution and a change notice. See `CITATION.cff`.
