# Manifest amendment v1.3 — FT-TRAJ-1 (2026-09-25)

Applies on top of `FMI_manifests_and_decision_rules_v1_2.md`; nothing in v1.2 is changed. New lock entry required.

## Trigger

NB-2 STOP gate A.7.4 fired for LRAA seed 71: the rerun reproduced the archived validated control for seeds 13 and 42
(on ≤ 0.31 pt, off ≤ 0.05 pt, same selected epochs) but seed 71 followed a divergent trajectory (selected epoch 3 vs
archived 2) and reached an adapter-dependent solution (on 88.44, norm_only 57.87, off 57.54, predicted-yes 91%),
with B_FT = 0.33. Cause: non-bitwise-reproducible full fine-tuning, not a code defect. Decision recorded in
MANIFEST_LOCK.md annotations: rerun retained; archived and rerun values reported side by side.

## New question (descriptive, not a confirmatory rule)

How often does matched full fine-tuning arrive at an adapter-dependent solution, and when in training does the
dependence appear? Six trajectories exist (three archived, three rerun): five with A_FT ≤ 1.1 pt, one with 30.6 pt.

## A.12 FT-TRAJ-1: two additional full-fine-tuning trajectories with per-epoch three-state evaluation

- Arm: LRAA full fine-tuning only, settings identical to A.8 (encoder 8e-6, adapter/scorer 2e-4, three epochs,
  batch 2 × 8, wd 0.01, warm-up 0.06, class-weighted loss, selection by (BA, macro-F1, −loss) on internal validation).
- Seeds: drawn once from `numpy.random.default_rng(20260925).integers(100, 10000, size=2)` → **7595, 1676**.
  Recorded here before training. (Optional third seed only by a new amendment.)
- Per-epoch measurement (descriptive): after every epoch, the held-out set is scored in the three states
  (`on`, `norm_only`, `off`) with predictions, probabilities, gate value and relative branch norm archived per epoch.
  Checkpoint selection remains on internal validation; per-epoch held-out numbers never influence selection.
  The selected epoch's full state is saved to Drive with SHA-256.
- Quantities: A_FT(seed, epoch), B_FT(seed, epoch), on−off(seed, epoch); selected-epoch values join the
  extended-replication block (C.3): reported separately from the three-seed prespecified decisions, with t(4)
  intervals over the five rerun seeds (13, 42, 71, 7595, 1676), never merged with the archived control.
- Reporting: the complete list of A_FT values over all trajectories (archived 3 + rerun 3 + new 2), the per-epoch
  A_FT curves, and the enabled BA of every trajectory. No threshold defines "dependent"; the distribution is shown
  and described. Wording: "in k of n trajectories the token-mixing component exceeded ⟨value⟩ points" with the
  values listed, plus the observation that enabled balanced accuracy does not separate them.
- Gates: optimization-validity gate as A.7.4(i); no STOP comparison with the archive (new seeds have no archived
  counterpart); hash verification of the saved full state.
- Prohibited: no selection on held-out; no extra seeds beyond the two recorded; no change to any hyperparameter.
- Compute (estimate, from NB-2 timings of ~42 min per run on an A100): two runs plus six held-out evaluations,
  about 1.7 hours.

## Manuscript consequence (prespecified wording)

Section 6.1 gains a paragraph and a figure ("Dependence by trajectory and epoch"); Table 3 gains the rerun rows;
the abstract's regime sentence is rewritten to report both the archived prespecified contrast and the rerun,
stating that enabled accuracy does not reveal which solution the optimizer found. Title unchanged.
