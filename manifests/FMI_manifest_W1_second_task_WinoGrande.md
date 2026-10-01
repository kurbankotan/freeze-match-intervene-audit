# Manifest W-1 — Second task under the same protocol: WinoGrande (2026-09-30)

Written before any run. Not committed in advance (stated as such in the paper); released with the artifacts.

## Why WinoGrande rather than PIQA

Two-candidate scoring fits the framework unchanged; the training set is small and sequences are short, so five
seeds fit in about four A100 hours; the frozen [CLS] probe is expected near chance (large headroom, the opposite
end from Qwen/BoolQ); full fine-tuning is expected far above it; twin sentences give a natural cluster unit; the
licence was already audited. PIQA remains a possible third task, not part of this manifest.

## Fixed design (everything locked from the BoolQ study unless stated)

- Backbone: DeBERTa-v3-large, frozen for all frozen-regime methods.
- Data: WinoGrande 1.1, size `l` (10,234 training items), official `dev` (1,267 items) as the evaluation set.
  Evaluation-set exposure: none before this study. Internal split of the training set by twin cluster
  (qID prefix before the final "-"): 90% internal training / 10% internal validation, seed 2026.
  Cluster unit for bootstrap: twin cluster of the dev set (qID prefix).
- Serialization: candidate i is the sentence with the blank replaced by option i, single segment
  ([CLS] filled sentence [SEP]); the two candidates form a pair; order randomized during training and reversed at
  evaluation. Max length 512 (dynamic padding). Scoring, loss (class-weighted cross-entropy over the pair; labels
  are balanced so weights ≈ 1) and the linear [CLS] scorer are unchanged.
- Methods, frozen regime (all trained from the same frozen forward pass within a seed, independent optimizers):
  head-only (1,025), token-wise MLP (1,050,625), LRAA (1,050,625), token mixer (1,050,625; same code as V13,
  including its tanh gate), and LoRA r = 11 on query/value of all 24 layers (1,081,344), trained separately.
- Methods, trainable regime: LRAA full fine-tuning (encoder 8e-6; adapter/scorer 2e-4) and no-adapter full
  fine-tuning (encoder 8e-6; scorer 2e-4).
- Optimization: AdamW, lr 2e-4 for frozen-regime methods, weight decay 0.01, 6% warm-up, three epochs, batch 2 ×
  accumulation 8, bf16 autocast; checkpoint per epoch selected on internal validation by (balanced accuracy,
  macro-F1, −loss). No learning-rate sweep for any method.
- Seeds: 13, 42, 71, 7595, 1676 (five from the start; t(4) intervals). All matched across methods.
- Intervention states at the selected checkpoint: on / norm_only / off for MLP, LRAA, mixer and full-FT LRAA;
  on / off for LoRA (off = LoRA update disabled with the trained scorer); on for head-only and no-adapter FT.
  D4: gate value and relative branch norm at [CLS].
- Single look: the dev set is scored once, after all training, for the selected checkpoints only.

## Estimands and decision rules (identical in form to the BoolQ study)

- Confirmation (LRAA vs MLP): mean balanced-accuracy gain ≥ 0.5, LRAA higher in every seed, paired 95% seed
  interval above zero.
- Attention topology (LRAA vs mixer): the same three conditions.
- Training regime: optimization gate (every FT seed ≥ 60% internal-validation BA; every enabled eval seed above
  chance; enabled mean above the frozen head-only mean); then ΔA = A_frozen − A_FT positive in all seeds with the
  paired 95% seed interval above zero. Reported alongside: on − off in both regimes, B in both regimes.
- D2: LRAA-FT(on) − no-adapter-FT with seed interval and cluster bootstrap; wording as Part C.2 of manifest v1.2.
- LoRA: benchmark without threshold.
- Uncertainty: seed-level paired t(4); twin-cluster bootstrap 10,000 replicates, seed 20260930; exact McNemar.
- Wording for intervals including zero: inconclusive; equivalence never claimed.

## Outcome map (exhaustive; wording fixed)

For each rule: met / not met, reported as such. The manuscript gains Section 6.7 "Second task" with one table
(five frozen methods + two FT arms; on/norm_only/off; A, B; ΔA; D2) and a per-seed appendix table. The abstract
gains one sentence stating the regime contrast on WinoGrande with its interval, whatever its sign. No pooling with
BoolQ or Qwen estimates.

## Prohibited

No hyperparameter changes after any result is seen; no re-selection; no extra seeds; no second look at the dev set;
no dropping of a method or seed.

## Compute (estimate)

Per seed: frozen quartet ≈ 5–8 min, LoRA ≈ 10–15 min, two full-FT arms ≈ 12–15 min each, evaluation ≈ 3 min.
Five seeds ≈ 3.5–5 A100 hours. Restart-safe per (seed, method).
