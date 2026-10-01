# MANIFEST_LOCK.md — revision 2026-09

Each entry records a manifest version, the SHA-256 of its released file, and the dated annotations made during
execution. None of these manifests was committed to a public repository before execution; they were written and
exchanged before the corresponding runs and are released with the artifacts (the manuscript states this explicitly).

| Manifest | File | SHA-256 | Written | Governs |
|---|---|---|---|---|
| v1.2 | manifests/FMI_manifests_and_decision_rules_v1_2.md | 3250afde4a2b6be6e3642b846c14371b11d320bf439b0a1030ba880d6213a0dc | 2026-09-24 | D-NORM-1 (NB-1); FT-RERUN-1 + D2 (NB-2); Q-HELDOUT-1 (NB-3, NB-4) |
| v1.3 amendment | manifests/FMI_manifest_v1_3_amendment_FT-TRAJ-1.md | 40ab9bbc963dac998560e7ca7c4c08826637c5922baa127c7b9b9bf332254ae7 | 2026-09-25 | FT-TRAJ-1 (NB-2b) |
| W-1 | manifests/FMI_manifest_W1_second_task_WinoGrande.md | 412fe5383cc08bd7848ee1d250f94ec054f78b082fac4060bbb2c72a24b6b3e1 | 2026-09-30 | WinoGrande second task (NB-5) |

## Annotations

- 2026-09-24 — NB-1 executed under v1.2 before this lock file existed; the rules did not change between execution and release.
- 2026-09-25 — STOP gate A.7.4, NB-2 LRAA seed 71. Deviation: module-off −30.07 points vs the archived control (enabled −0.27).
  Investigation: seeds 13 and 42 reproduce the archived control (enabled ≤ 0.31, off ≤ 0.05; same selected epochs); seed 71 selected
  epoch 3 (archived: 2) after a divergent trajectory (internal-validation BA .883/.883/.890 vs .888/.901/.895); the selected state is
  adapter-dependent (norm_only 57.87, off 57.54, predicted-yes 91%); B_FT = 0.33. Cause: non-bitwise-reproducible full fine-tuning
  (A100-40GB vs A100-80GB), not a code defect. Decision: rerun retained; archived and rerun values reported side by side; manifest Part C.1 → case 2 for the rerun.
- 2026-09-25 — Gate B.7.2 deviation, NB-4. The frozen seed-42 archives (wave4b13) were produced on a Tesla T4 (emulated bf16); on an A100
  the identical base model differs on 6/925 development rows (max logit deviation 0.25, one bf16 unit at magnitude ≈ 30); with the path
  enabled, 11/925 (attention) and 15/925 (mixer) rows differ, max deviation 0.375. All A100-archived cells reproduce exactly (0 mismatches,
  0.0 deviation). Override limited to the wave4b13 cells (≤ 2% label mismatches, ≤ 0.5 logit deviation).
- 2026-09-30 — W-1 written before the WinoGrande run; a smoke test on a subsample preceded the full run; after a power outage the analysis
  cell was re-executed from the stored progress files and reproduced all 42 result files byte for byte.
