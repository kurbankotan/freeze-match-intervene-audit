# Revision 2026-09: freeze–match–intervene audit — new experiments

Artifacts behind the revised manuscript *Full Fine-Tuning Can Obscure Adapter Contribution: A Freeze–Match–Intervene
Audit of a Low-Rank Attention Adapter*. Nothing in the original `results/` and `checkpoints/` directories was modified;
every archived number is unchanged.

| Experiment | Notebook | Manifest | Results directory | Adds |
|---|---|---|---|---|
| D-NORM-1: three-state intervention (on / norm_only / off) on the nine frozen BoolQ checkpoints | `notebooks/fmi_nb1_dnorm1_frozen_decomposition_colab.ipynb` | v1.2, Part A | `results/fmi_dnorm1_frozen_results_20260924T181240Z/` | Boundary component B = 0.0 ± 0.2 pt; 36/36 parity checks against the archives |
| FT-RERUN-1 + D2: retrained full-FT control with three states; no-adapter full FT | `notebooks/fmi_nb2_fullft_rerun_d2_colab.ipynb` | v1.2, Part A.8 | `results/fmi_nb2_fullft_results_20260925T081859Z/` | A_FT per seed 0.68 / 0.12 / 30.57; no-adapter FT +0.01 [−1.29, +1.31]; STOP-gate event (seed 71) |
| FT-TRAJ-1: two further full-FT trajectories with per-epoch three-state evaluation | `notebooks/fmi_nb2b_ft_traj1_per_epoch_colab.ipynb` | v1.3 amendment | `results/fmi_nb2b_ft_traj1_results_20260925T111023Z/` | Eight trajectories in total: seven independent, one dependent |
| Q-HELDOUT-1: Qwen3-1.7B-Base single-look held-out evaluation of 15 checkpoints | `notebooks/fmi_nb3_qwen_hash_precheck_cpu_colab.ipynb`, `notebooks/fmi_nb4_qwen_heldout_single_look_colab.ipynb` | v1.2, Part B | `results/fmi_nb4_qwen_heldout_results_20260925T164440Z/` | Δ_regime +1.67 [+0.47, +2.86] (attention), +1.67 [+0.34, +2.95] (mixer) |
| W-1: second task, WinoGrande, 5 seeds × 7 methods | `notebooks/fmi_nb5_winogrande_second_task_colab.ipynb` | W-1 | `results/fmi_nb5_winogrande_results_20260930T222505Z/` | ΔA +26.65 [+16.45, +36.85]; all prespecified rules met |

Every results directory contains per-row predictions, aggregate tables, bootstrap and McNemar outputs, gate reports,
`run_environment.json` and `FILE_MANIFEST_SHA256.csv`. `MANIFEST_LOCK.md` records the manifest hashes and every
annotation made during execution. Large model states are not redistributed (DeBERTa full fine-tuning ≈ 1.7 GB each;
Qwen ≈ 6.9 GB each); their SHA-256 values are in the result files. The WinoGrande adapter/scorer states, the Qwen path
weights (six safetensors files) and the FPDA development-set archives can be added under `checkpoints/revision_2026-09/`
and `results/revision_2026-09/fpda_development/` if storage allows.

`lcca` in file and checkpoint names is the legacy identifier of the module called LRAA in the paper.
