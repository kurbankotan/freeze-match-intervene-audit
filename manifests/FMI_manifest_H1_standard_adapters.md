# Manifest H-1 — Does full fine-tuning also obscure the contribution of standard adapters? (2026-10-02)

Written before any run. Released with the artifacts; not committed in advance (stated as such in the paper).

## Question

The audited LRAA is a bespoke module. H-1 asks whether the central finding generalizes to the most widely used adapter
design: bottleneck adapters inserted in every Transformer layer (Houlsby et al., 2019). If standard adapters carry load
under a frozen backbone but become dispensable under matched full fine-tuning, the claim concerns adapters in general,
not one module.

## Fixed design

- Backbone: DeBERTa-v3-large. Tasks and splits identical to the paper: BoolQ (passage-grouped internal split 8,351 /
  1,076; official validation 3,270 as held-out; seeds 13, 42, 71) and WinoGrande 1.1 size l (twin-cluster internal split;
  official dev 1,267; seeds 13, 42, 71, 7595, 1676).
- Adapter: Houlsby bottleneck adapter after the output projection of every attention sublayer and every feed-forward
  sublayer (48 adapters), before the residual sum and LayerNorm: y ↦ y + W_up GELU(W_down y), bottleneck 10, biases,
  W_up initialized to zero (identity at initialization). Trainable adapter parameters: 48 × 21,514 = 1,032,672
  (1.7% fewer than LRAA, 4.5% fewer than LoRA r = 11), plus the common 1,025-parameter scorer on [CLS].
- Regimes: (a) frozen backbone, adapters and scorer trained (lr 2 × 10⁻⁴, the locked rate of every frozen method);
  (b) full fine-tuning with adapters (encoder 8 × 10⁻⁶; adapters and scorer 2 × 10⁻⁴), identical to the LRAA control.
- Everything else locked from the paper: three epochs, batch 2 × accumulation 8, AdamW (weight decay 0.01), 6% warm-up,
  bf16 autocast, class-weighted cross-entropy over the candidate pair, selection per epoch on internal validation by
  (BA, macro-F1, −loss), gradient checkpointing. No learning-rate sweep.
- Intervention: `on` (all adapters active) and `off` (all 48 adapters bypassed exactly: y ↦ y). There is no adapter-owned
  normalization, so the bypass has no boundary component by construction.
- Baselines reused without retraining: the archived no-adapter full-fine-tuning runs (BoolQ, three seeds; WinoGrande, five
  seeds) and the frozen head-only means (BoolQ 60.47; WinoGrande 74.86).
- Single look: each selected checkpoint is scored once on the evaluation set.

## Estimands and rules

- Dependence: Δ_path = BA(on) − BA(off), per regime and seed.
- Regime rule (prespecified): every full-fine-tuning run passes the optimization gate (internal-validation BA ≥ 60%,
  held-out BA above chance, mean above the frozen head-only mean) and the frozen-minus-full-FT dependence difference is
  positive in every seed with its paired 95% seed interval above zero. Met / not met is reported as such.
- Frozen validity: the frozen-adapter mean must exceed the frozen head-only mean; otherwise the frozen arm is reported
  as an optimization failure and the regime rule is not evaluated.
- Contribution (D2): full fine-tuning with adapters minus full fine-tuning without, with seed interval and cluster
  bootstrap; wording as in manifest v1.2, Part C.2.
- Trajectories: every full-fine-tuning seed's on − off is listed; no threshold defines "dependent".
- Uncertainty: paired t intervals over seeds (t(2) BoolQ, t(4) WinoGrande); cluster bootstrap (passage clusters for
  BoolQ, twin clusters for WinoGrande), 10,000 replicates, seed 20261002, one shared resample for all seeds and
  conditions; exact McNemar on vs off per seed.
- Gate before analysis: the evaluation-set gold labels and order must match the archived no-adapter predictions exactly.

## Outcome map (wording fixed)

1. Regime rule met on both tasks → "the regime contrast generalizes to standard bottleneck adapters on both tasks";
   the abstract, introduction and conclusion state the finding for adapters in general.
2. Met on one task → reported per task; the generalization claim is restricted to that task.
3. Met on neither (frozen dependence small, or full-FT dependence large) → reported as a limit: the obscuring effect is
   specific to final-layer branches; the paper's claim stays restricted to the audited module type.
In every case all numbers are reported; nothing is dropped.

## Prohibited

No hyperparameter change after any result; no extra seeds; no re-selection; no second look at an evaluation set.

## Compute (estimate)

Frozen-adapter training back-propagates through the encoder (as LoRA does). Roughly 45–60 minutes per run on BoolQ and
35–50 minutes on WinoGrande on an A100: about 5–6 hours for BoolQ (6 runs) and 6–7 hours for WinoGrande (10 runs).
Restart-safe per (task, regime, seed).
