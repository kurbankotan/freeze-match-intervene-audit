# Prespecification manifests, decision rules and runbook — v1.2 (2026-09-24)

Status: FOR AUTHOR APPROVAL, THEN LOCK. The lock is a separate record
(Part E); this file is not edited after locking. A later change produces v1.3
and a new lock entry. Compute figures are estimates from archived runtimes.

## Changes from v1.1

1. Parity gates written per model and state according to what each archive
   actually contains (labels only for LRAA/MLP; labels and `prob_yes` for the
   token mixer `on`; aggregate metrics only for the token mixer `off`).
2. Decision map completed: added the case A_frozen > 0 with ΔA entirely
   negative; Qwen wording made conditional on the sign of the point estimate
   when the interval includes zero.
3. DeBERTa bootstrap unit changed from rows to passage clusters (rows sharing a
   passage are resampled together); archived example-level bootstraps stay as
   archived and passage-clustered versions are added from archived predictions
   without GPU.
4. Full-FT rerun: a stop gate replaces "report and use anyway" when the rerun
   deviates from the archived control by more than 1.0 BA point in any seed.
5. Commit hash and manifest SHA-256 recorded in a separate lock file, not in
   this document.
6. NB-3 (Qwen hash pre-check) noted as CPU-only but not instantaneous (about
   62 GB read from Drive).

Conventions: BA = balanced accuracy in percentage points; seed-level interval =
paired t interval over seeds; cluster bootstrap = resampling of passage clusters
with all rows of a passage and all seed repeats retained; an interval that
includes zero is "inconclusive", never "equivalent".

---

## Part A — D-NORM-1: DeBERTa intervention decomposition

### A.1 Purpose

The archived module-off intervention bypasses the whole inserted module,
including its output LayerNorm, so the trained scorer receives un-normalized
encoder states. D-NORM-1 separates the token-mixing component from the
normalization-boundary component on the same checkpoints, in both regimes.

### A.2 Frozen-backbone checkpoints (9; SHA-256 verified locally by Codex)

LRAA (legacy id `lcca`), token-wise MLP, token mixer × seeds 13, 42, 71.
Head-only has no module; LoRA has no external LayerNorm; neither enters
D-NORM-1.

### A.3 Intervention states

Module family H' = LN(H + g·Dropout(Z)), eval mode:

- `on`: H' = LN(H + g·Z)
- `norm_only`: H' = LN(H), Z set to zero, trained LN parameters kept
- `off`: H' = H, module bypassed (archived "adapter-off" state)

Frozen: H is the frozen DeBERTa-v3-large final-layer output. Full FT: H is the
fine-tuned encoder output at the selected epoch. Masks, serialization,
truncation (`only_second`), max length 512, bf16 autocast and evaluation batch
size follow the archived evaluation code; metric functions are imported
unchanged.

### A.4 Data

Official BoolQ validation split, 3,270 rows in 2,938 passage clusters (the
study's held-out set). No internal-validation runs for the frozen models.

### A.5 Quantities

Per model, seed and state: accuracy, BA, macro-F1, predicted-yes rate, log
loss; per-example `prediction` and `prob_yes` archived for every state
produced in D-NORM-1 (this fills the gaps noted in A.7).

Decomposition (BA):

- A = on − norm_only (token-mixing component)
- B = norm_only − off (normalization-boundary component)
- A + B must equal the archived on − off within 0.01 point where an archived
  on − off exists (gate A.7.3)
- share = A/(A+B) with its bootstrap interval when A + B > 0; otherwise
  "undefined"
- G = on − head_only (60.47), a system-level quantity reported beside A

Regime (LRAA): A_frozen, B_frozen, A_FT, B_FT; ΔA = A_frozen − A_FT;
ΔB = B_frozen − B_FT. The archived on − off regime contrast (26.81) is kept
as the pre-decomposition value.

### A.6 Uncertainty

Seed-level: mean, SD, all seed values, paired 95% t interval for A, B, ΔA.
Cluster bootstrap over the 2,938 passage clusters (all rows of a passage and
all seed repeats resampled together), 10,000 replicates, seed 20260924, for A,
B, share, ΔA and the D2 contrast. Exact two-sided McNemar per seed for on vs
norm_only and norm_only vs off. All intervals two-sided.

The archived held-out example-level bootstraps (for example the +1.02
[+0.23, +1.80] LRAA-minus-mixer accuracy interval) remain as archived;
passage-clustered versions of the same contrasts are computed from the archived
predictions on CPU and reported beside them in the revision.

### A.7 Gates (before interpretation)

1. Every checkpoint's SHA-256 matches its archived manifest entry (done for the
   nine frozen files; re-checked inside the notebook).
2. Parity against the archives, per model and state. Codex lists the exact
   archive paths and columns in the lock record before the run.
   - LRAA and MLP, frozen, `on` and `off`: archived held-out per-example
     predicted labels exist (adapter-on/off audit). Gate: label-exact match on
     all 3,270 rows and all seeds, zero mismatches. No probability comparison
     (probabilities not archived).
   - Token mixer, frozen, `on`: archived per-example labels and `prob_yes`
     exist (architecture-control predictions). Gate: label-exact match, zero
     mismatches; maximum absolute `prob_yes` deviation recorded and ≤ 1e-3.
   - Token mixer, frozen, `off`: only aggregate metrics archived
     (`adapter_off_accuracy`, `adapter_off_balanced_accuracy`,
     `adapter_off_macro_f1` per seed). Gate: aggregates reproduced within 0.01
     point; the per-example `off` predictions produced now are archived.
   - Aggregates for all `on` and `off` states: accuracy, BA, macro-F1 within
     0.01 point of the archived tables.
   A label mismatch, an aggregate deviation above 0.01 point, or a probability
   deviation above 1e-3 stops interpretation until a written explanation is
   added to the lock record.
3. A + B equals the archived on − off within 0.01 BA point for every frozen
   model and seed.
4. Full-FT rerun (A.8): (i) the original optimization-validity gate passes
   (every seed ≥ 60% internal-validation BA; every enabled held-out seed above
   chance; enabled held-out mean above 60.47); (ii) enabled and off held-out
   BA per seed are compared with the archived control (88.99/88.04,
   88.55/88.18, 88.70/87.61). STOP GATE: if any seed deviates by more than
   1.0 BA point in either state, the run is not interpreted; the deviation is
   investigated (environment, library versions, data order, class weights,
   selection rule) and the finding is written to the lock record before any
   further step. Only after that record exists is the rerun used for the
   decomposition, with both archived and rerun values reported.

### A.8 Full-fine-tuning rerun (FT-RERUN-1) and D2, one notebook

Settings identical to the validated control: passage-grouped split, held-out
set, seeds 13/42/71, three epochs, max length 512, batch 2 × accumulation 8,
weight decay 0.01, warm-up 0.06, class-weighted loss, selection key
(BA, macro-F1, −loss) on internal validation, encoder learning rate 8e-6,
adapter/scorer 2e-4, bf16 autocast. The archived validated-control
checkpoints hold only the adapter and scorer, which is why the encoder must
be retrained. ⟦Author: confirm no manual full-state copy exists elsewhere.⟧

Arm (a) LRAA full FT: at the selected epoch, evaluate on / norm_only / off on
the held-out set; save the full model state (encoder + module + scorer,
about 1.7 GB fp32 per seed) to Drive with SHA-256; record gate value and branch
norm (D4).

Arm (b) No-adapter full FT (D2): encoder + scorer only, identical settings;
evaluate on the held-out set; save the full state with SHA-256.

D2 quantities: no_adapter_FT BA per seed; paired difference LRAA_FT(on) −
no_adapter_FT with seed-level interval and cluster bootstrap; McNemar per seed.
Wording per Part C.2.

Estimate: six training runs at about 62 minutes each on an A100 (archived
185.3 min for three seeds) plus evaluation: 6–7 A100 hours; resumable per seed.

### A.9 D4 inside D-NORM-1 (descriptive)

For every module checkpoint in both regimes: learned gate value (sigmoid(α) for
LRAA and MLP; tanh(θ) for the mixer) and mean relative branch norm
‖g·Z[CLS]‖ / ‖H[CLS]‖ over the held-out set. No inference attached.

### A.10 Prohibited

No new hyperparameters, no checkpoint re-selection, no extra seeds inside
D-NORM-1, no change to metric code, no internal-validation evaluation for the
frozen models.

### A.11 Reporting template (manuscript Table 2; seed rows to Appendix A)

| Model | Regime | on | norm_only | off | A [seed CI] | B [seed CI] | share [cluster CI] | G |
|---|---|---:|---:|---:|---|---|---|---:|
| LRAA | frozen | 83.89 | ⟦D1⟧ | 56.28 | ⟦D1⟧ | ⟦D1⟧ | ⟦D1⟧ | 23.42 |
| MLP | frozen | 76.77 | ⟦D1⟧ | 67.93 | ⟦D1⟧ | ⟦D1⟧ | ⟦D1⟧ | 16.30 |
| Token mixer | frozen | 83.01 | ⟦D1⟧ | 64.07 | ⟦D1⟧ | ⟦D1⟧ | ⟦D1⟧ | 22.54 |
| LRAA | full FT (rerun) | ⟦D1⟧ | ⟦D1⟧ | ⟦D1⟧ | ⟦D1⟧ | ⟦D1⟧ | ⟦D1⟧ | vs no-adapter FT: ⟦D2⟧ |

---

## Part B — Q-HELDOUT-1: Qwen3-1.7B-Base single-look held-out evaluation

### B.1 Purpose

Convert the exploratory development-set factorial (2 topologies × 2 regimes ×
3 seeds, plus 3 no-path models) into a held-out result with prespecified
inference, using existing checkpoints only. The official BoolQ validation split
is opened once, after the lock and after gates B.7 pass.

### B.2 Fixed inputs

Backbone `Qwen/Qwen3-1.7B-Base` at revision
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`; data `google/boolq` at revision
`35b264d03638db9f4ce671b711558bf7ff0f80d5`, split `validation` (3,270 rows);
grouping unit `sha256(passage)`, group count recorded. Scoring: native
verbalizer at the last non-pad token through `lm_head`, ids ` yes` = 9834,
` no` = 902, two-way softmax; serialization, max length 512, truncation, bf16
autocast, eval batch size unchanged from the development-set notebooks. No
prompt or verbalizer change.

### B.3 Checkpoint inventory (15)

Existence and byte sizes confirmed by Codex from the local Drive mirror;
SHA-256 values verified in NB-3 (B.7.1).

Frozen paths (path weights; backbone = pinned base model):

| Cell | Seed | Epoch | File | Bytes | SHA-256 |
|---|---:|---:|---|---:|---|
| attention_frozen | 13 | 1 | wave4b10 …/attention_frozen_seed13_best.safetensors | 4,211,252 | d47383e873f7a830b816fdaba9a5731bb7c4638cbeb843ca635da50413c9558c |
| mixer_frozen | 13 | 3 | wave4b10 …/mixer_frozen_seed13_best.safetensors | 4,211,180 | 0b15dba7c9f75f0c200b4b60f8b57012ba0ea1940a1c078d852cc4e49fa434d7 |
| attention_frozen | 42 | 2 | wave4b13 …/attention_frozen_seed42_best.safetensors | 4,211,252 | 3b6516f329d27eadbd993c4f9563be303f5fc853293df87bf5c69fe97c6e27e1 |
| mixer_frozen | 42 | 3 | wave4b13 …/mixer_frozen_seed42_best.safetensors | 4,211,180 | 456fc913dfccdb637e00e3f48aa2d78a5e2d10654eb20ac4cb5d2d6a1ce02f11 |
| attention_frozen | 71 | 2 | wave4b14 …/attention_frozen_seed71_best.safetensors | 4,211,252 | 871ce186224563a598645906b08cf3ed9c849cdfb733f4ea353de7abcf676e9b |
| mixer_frozen | 71 | 3 | wave4b14 …/mixer_frozen_seed71_best.safetensors | 4,211,180 | 81d944c261bb5dcffe7cb13dd1f7d5152df87a021a1d483eb63d016f92c8d9f6 |

Trainable backbones (`best_model_path_weights.pt` in each wave's Drive progress
directory; loaded only after the hash matches):

| Cell | Seed | Epoch | Wave | Bytes | SHA-256 |
|---|---:|---:|---|---:|---|
| attention_trainable | 13 | 1 | wave4b11 | 6,886,644,540 | e1cbb96584eecdbcc4b9efe127bb01a0e1659118158c7ce298aef5d040f42029 |
| mixer_trainable | 13 | 1 | wave4b12 | 6,886,644,197 | ade9bde676492048168522a45dfa5ba1811b5d7485c71f947d404fcac072bee6 |
| attention_trainable | 42 | 3 | wave4b15 | 6,886,644,540 | 462b60feebdea8ec541b4e7a275b1e99a7c108bf90344e1e3cd0cd0819e73ac3 |
| mixer_trainable | 42 | 3 | wave4b16 | 6,886,644,197 | bcd3792d59db37f126e037353092e3142928d9f7b9f3711663ca1ae084fee613 |
| attention_trainable | 71 | 3 | wave4b17 | 6,886,644,540 | 1b9e597e67dc15f5622c90909944715b338b3b4fcb91c1ddc32c99a6fd7a8c15 |
| mixer_trainable | 71 | 3 | wave4b18 | 6,886,644,197 | 5e8b5317623f510c2dc4c4a80c742a9e12af079a447d76a669264bfc165d1d95 |
| no_path | 13 | 1 | wave4b8 | 6,882,431,387 | dc653687667dffd8be4b3ef6218c51298f2da46b390da22a64bcbb129b1e25c6 |
| no_path | 42 | 3 | wave4b19 | 6,882,431,387 | af198e20cd0a2bcb9e99a7d246f84b60b734e360870fa536641e90e9b53ac090 |
| no_path | 71 | 3 | wave4b20 | 6,882,431,387 | dfb538ba31aaa3c1e08eee432e95ba0c7f58787ea0ce8424e938fde905dd8b8e |

A missing or mismatched file is recorded and its cell reported as unavailable;
no retraining inside Q-HELDOUT-1.

### B.4 States per checkpoint

Frozen paths: on; off (= pristine base, evaluated once and shared). Trainable
paths: on; off (exact identity-preserving bypass). No-path: native. Pristine
base: native (zero-shot reference). Metrics: BA (primary), accuracy, macro-F1,
predicted-yes rate, log loss; per-row predictions with both candidate logits
archived for every state.

### B.5 Estimands

Per topology T ∈ {attention, mixer}, BA: Δ_path,frozen(T), Δ_path,trainable(T);
PRIMARY Δ_regime(T) = Δ_path,frozen(T) − Δ_path,trainable(T). Secondary:
enabled_trainable(T) − no_path; attention − mixer within regime; frozen on −
zero-shot; no_path − zero-shot; accuracy versions of all.

### B.6 Uncertainty

PRIMARY: passage-cluster bootstrap with one cluster resample per replicate
applied identically to every seed and condition (shared-passage scheme),
10,000 replicates, seed 20260925, percentile intervals. Sensitivity:
seed-then-cluster hierarchical bootstrap; seed-level descriptives; exact
McNemar on vs off per checkpoint. No scheme may be promoted to primary after
results are seen.

### B.7 Gates (before the held-out split is scored)

1. NB-3, CPU-only Colab: SHA-256 of all 15 files computed and compared to
   B.3; report archived. Reads about 62 GB from Drive, so it takes time even
   without a GPU; it is run once.
2. Development-set reproduction: every checkpoint re-scored on the 925
   development rows reproduces the archived best-epoch per-row labels with zero
   mismatches; the maximum absolute deviation of the archived per-row logits
   (`yes_logit`, `no_logit`) is recorded (≤ 1e-3 expected; larger deviations
   stop the run until explained in the lock record). Frozen `off` equals
   `native` exactly.
3. Split audit: the held-out loader reports 3,270 rows, zero passage-hash
   overlap with the 8,502 + 925 development partition, and the group count.

### B.8 Reporting rule

Whatever the outcome, the main text carries one short table (Table 5 of the
draft) and one paragraph of prespecified wording; full details go to
Appendix F.

Wording, exhaustive over the primary interval CI(T) and point estimate P(T):

- CI(T) entirely above zero: "the frozen-minus-trainable dependence difference
  is positive for T on the held-out set (P, CI)"; "replicates" may be used for
  direction and sign only, never for magnitude.
- CI(T) includes zero and P(T) > 0: "the point estimate is positive but the
  interval includes zero; inconclusive at this sample (P, CI)".
- CI(T) includes zero and P(T) ≤ 0: "the point estimate is zero or negative
  and the interval includes zero; inconclusive at this sample (P, CI)".
- CI(T) entirely below zero: "the difference is negative (P, CI)".
- In all cases: seed-level direction consistency reported as a descriptive
  fact; magnitude stated in points and compared with DeBERTa's as an
  observation; the headroom explanation stated as a hypothesis (B.9).

### B.9 Prespecified statements

Intervals including zero are inconclusive; equivalence is not claimed. The
DeBERTa–Qwen magnitude difference is an observation; the headroom explanation
is a hypothesis consistent with two data points, not a tested cause. No
attention-superiority claim in either regime.

### B.10 Prohibited

No checkpoint re-selection; no training triggered by the held-out result; no
prompt or verbalizer change; no second look with modified analysis code; no
pooling of DeBERTa and Qwen estimates.

### B.11 Compute (estimate)

Nine 6.9 GB checkpoint loads plus six path loads; about 3,270 × states forward
passes at up to 512 tokens; the development-set gate adds 925 × states per
checkpoint. Estimate 4–7 A100 hours. Recorded: GPU, torch/CUDA build, wall
time per checkpoint.

---

## Part C — Manuscript decision map

### C.1 D-NORM-1 (exhaustive; wording fixed in advance)

CI_A = seed-level 95% interval of A_frozen (LRAA); CI_ΔA = interval of
ΔA = A_frozen − A_FT.

1. CI_A above zero and CI_ΔA above zero → primary dependence measure becomes A;
   abstract and tables replace 27.62/26.81 with A_frozen/ΔA; B reported as a
   measured boundary component; main claim retained.
2. CI_A above zero and CI_ΔA includes zero → the module carries load when
   frozen; the regime claim is reported as inconclusive at three seeds after
   boundary correction; the archived on − off contrast shown as
   pre-decomposition only; abstract restated accordingly.
3. CI_A above zero and CI_ΔA entirely below zero → the token-mixing component
   is larger under full fine-tuning than under freezing; the training-regime
   claim is not supported for this module and its direction is reversed; title
   and abstract rewritten to report this; the archived on − off contrast shown
   as pre-decomposition only.
4. CI_A includes zero or lies below zero (any CI_ΔA) → the token-mixing
   component of the frozen dependence is not distinguishable from zero at
   three seeds; the archived dependence cannot be attributed to the module's
   computation; B is reported as the measured boundary component without
   claiming it causes the remainder; the primary interpretation is rewritten
   as a methodological result about intervention design; G is reported as
   system-level performance, not as dependence.

In every case all three states are reported for every seed and model; no
archived number is dropped.

### C.2 D2 (no-adapter full FT)

Report no_adapter_FT BA with seed values and the paired difference from
LRAA_FT(on) with seed-level and cluster intervals. Interval includes zero →
"adding LRAA to full fine-tuning produced no measurable change at three
seeds"; entirely above zero → report the gain and note that the intervention
concerns dependence, not gain; entirely below zero → report as such.

### C.3 D3 (deferred)

Decided after D-NORM-1 results within the fixed budget. If run: seeds drawn
from a fixed RNG and recorded in the lock record before training; reported as
a separate extended replication with t(4) intervals; never merged with the
three-seed prespecified decisions.

### C.4 D4

Descriptive gate values and branch norms; Discussion context only.

### C.5 What never changes

Data split, seeds 13/42/71, metric code, checkpoint selection rules, the
polarity contrast set and its validation, every archived number.

---

## Part D — Runbook (owners and order)

Owners: Author (A), Codex (C), Claude (K).

0. A — Three confirmations: TACL status with dates; whether JMLR Submit was
   pressed; whether official BoolQ validation results were consulted during
   V1–V8 development. These determine venue eligibility and how held-out
   results are labelled.
1. A — Approve this file. C — Commit it; create the lock record (Part E) with
   this file's SHA-256 and the commit hash; also list in the lock record the
   archive paths and columns used by gate A.7.2.
2. C — Prepare NB-1 (D-NORM frozen: 9 checkpoints × 3 states, gates A.7.1–3,
   D4, analysis per A.6) and NB-3 (Qwen hash pre-check, CPU). K and C review
   both before any execution. NB-1 estimate: under 1 GPU hour (T4 sufficient).
3. C — Run NB-3 (CPU), archive the report. Run NB-1 after review.
4. C — Prepare NB-2 (FT-RERUN-1 + D2 per A.8, stop gate A.7.4). K reviews.
   Run after review. Estimate 6–7 A100 hours, resumable per seed.
5. C — Prepare NB-4 (Q-HELDOUT-1: gates B.7.2–3, then single-look scoring and
   analysis per B.5–B.6). K reviews. Run after NB-3 passes and after review.
   Estimate 4–7 A100 hours.
6. K — Apply Part C to the manuscript draft when NB-1/NB-2 results arrive
   (⟦D1⟧, ⟦D2⟧, ⟦D4⟧); apply B.8 when NB-4 arrives (⟦Q1⟧); add
   passage-clustered versions of archived held-out contrasts (A.6, CPU).
7. A + K — In parallel: reference verification ([VERIFY] items), ⟦AUTHOR⟧
   items, language pass, central figure.
8. A + K — Decide D3 and any optional control within the fixed budget after
   NB-1/NB-2; decide venue after step 0 is answered and NB-1/NB-2/NB-4 are in.
9. K — Submission .docx (numbered references, figures, tables) and cover
   letter with the change log.

GPU order: NB-1 → NB-2 → NB-4 (NB-3 is CPU and precedes NB-4). Nothing else
runs. The archived Wave 4B-20 notebook is not rerun.

---

## Part E — Lock record template (separate file `MANIFEST_LOCK.md`)

```
manifest_file: FMI_manifests_and_decision_rules_v1_2.md
manifest_sha256: <computed at commit>
commit: <hash>
locked_on: <UTC timestamp>
approved_by: Kurban Kotan
gate_A7_2_archives:
  lraa_mlp_heldout_on_off_labels: <repo path>; columns: <...>
  token_mixer_heldout_on: <repo path>; columns: example_index, gold, prediction, prob_yes, model, seed
  token_mixer_off_aggregates: <repo path>; columns: adapter_off_accuracy, adapter_off_balanced_accuracy, adapter_off_macro_f1
qwen_dev_predictions: <per-wave paths>; columns: row_index, target, prediction, yes_logit, no_logit, state
annotations: []   # dated entries for stop-gate investigations; never edit the manifest itself
```

Any later change to the rules is a new manifest version with a new lock entry.
