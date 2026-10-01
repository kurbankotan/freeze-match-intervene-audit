# Human validation of the polarity contrast set

Two independent human reviewers evaluated all 81 BoolQ contrast items under
blinded, independently randomized A/B presentation. The files in this folder
preserve the normalized independent first pass and the post-adjudication
decisions. Reviewer identities are not included in the public artifact.

## Independent first pass

- Original-answer agreement: 93.8%, Cohen's kappa 0.842, Gwet's AC1 0.923.
- A-answer agreement: 98.8%, Cohen's kappa 0.966, Gwet's AC1 0.985.
- B-answer agreement: 93.8%, Cohen's kappa 0.842, Gwet's AC1 0.923.
- Item-acceptability agreement: 74.1%, Cohen's kappa 0.335, Gwet's AC1 0.678.

Relation-to-original judgments are retained in the item-level file but are not
treated as an independent reliability outcome because they are determined by
the answer judgments and have extreme category prevalence.

## Adjudication outcome

- 47 items were valid for both reviewers in the first pass.
- 34 items required a recorded decision: 24 contained a field-level
  disagreement and 10 were agreed rejections.
- 58/81 items are usable in all O/A/B conditions after adjudication.
- 23 items are excluded: 18 can be repaired by regenerating one or both
  rewritten conditions, and 5 are permanent drops because the original
  passage does not determine an answer.

The unchanged model predictions are re-analysed on the 58-item validated set in
`../human_validated_contrast/`. The filtering rule is purely textual and was
applied without retraining or changing the contrast metric implementation.

