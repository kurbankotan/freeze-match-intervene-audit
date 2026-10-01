# Validation of the counterfactual passage set — methods and results

Draft text and tables for the paper. All numbers come from
`human_validation_final_81.csv` and `human_validation_final_summary.json`.

## Design

Each of the 81 BoolQ items carries three passages: the **original**, a **meaning-preserving
rewrite (A)** and a **meaning-reversing rewrite (B)**. Two reviewers coded every item
independently and blind to each other, from separate CSVs in which the two rewrites were
presented in a different, per-file random order, so neither reviewer could infer which passage
was intended as A and which as B.

For each item a reviewer recorded the answer the question receives under each of the three
passages (`yes` / `no` / `unclear`), the relation of each rewrite to the original
(`same` / `opposite` / `neither` / `unclear`), and whether the item was clear enough to use
(`item_acceptable`), with a short written justification whenever an item was rejected.

An item counts as **valid** when the original passage yields a determinate answer, A reproduces
it and B reverses it.

## Inter-rater reliability (independent first pass)

Computed before any adjudication; the first-pass files were not modified at any later stage.

| field | agreement % | Cohen's κ | Gwet's AC1 |
|---|---:|---:|---:|
| answer, original passage | 93.8 | 0.842 | 0.923 |
| answer, passage A | 98.8 | 0.966 | 0.985 |
| answer, passage B | 93.8 | 0.842 | 0.923 |
| relation, A to original | 95.1 | 0.000 | 0.950 |
| relation, B to original | 88.9 | −0.036 | 0.885 |
| item acceptable | 74.1 | 0.335 | 0.678 |

Two caveats belong in the text rather than the footnotes. First, the κ values for the relation
fields are a **prevalence artifact**: one reviewer coded `same` for all 81 items on `relation_a`,
so that margin has no variance and κ collapses despite 95.1 % agreement. Reporting κ = 0.00 there
would misdescribe the data, which is why a prevalence-robust coefficient is given alongside.
Second, the relation fields are a deterministic function of the answer fields and therefore carry
no independent information; they should not be presented as a separate reliability estimate.

Reliability was substantial for every judgment that concerns what a passage says, and only
moderate for `item_acceptable`. The adjudication showed that this was not noise: the two reviewers
were applying different thresholds, one rejecting items for semantic incoherence and the other for
surface defects such as unmatched quotation marks. The adjudication produced an explicit rule
(below), which is the form in which the criterion should be published.

## Adjudication

47 items were valid for both reviewers after the first pass. The remaining **34** required a
decision: **24** with at least one field-level disagreement, and **10** that both reviewers had
independently rejected for the same reason. Reviewers exchanged written positions in two rounds,
each stating a proposed decision and its justification; **19** items were settled in the first
exchange and the remaining **5** in the second, after both reviewers accepted two general
principles rather than arguing item by item. **No item required third-party adjudication.**

The agreed criterion for `item_acceptable`:

> **Reject** when the defect can change the answer a reader gives — a rewrite that is a verbatim
> copy of the original; retained text that re-asserts the proposition B denies; an edited sentence
> that contradicts itself; a B that negates a proposition other than the one the question asks
> about; a reference to deleted content with no recoverable antecedent; or an original passage that
> does not determine an answer.
>
> **Keep** despite typographic artifacts, whether inherited from the source or introduced by the
> rewrite; awkward wording, double negatives and coinages that leave the proposition unambiguous;
> and the counterfactual falsity of B, which is the manipulation itself.

## Results

| | items |
|---|---:|
| Valid for both reviewers, first pass | 47 |
| Recovered in adjudication | +11 |
| **Usable in all conditions** | **58 / 81 (71.6 %)** |
| Excluded | 23 |
| — recoverable by regenerating one condition | 18 |
| — permanent (original answer indeterminate) | 5 |
| Items with a valid A condition | 73 |
| Items with a valid B condition | 60 |

Validity was recorded per condition as well as per item, because most excluded items are defective
in exactly one of the two rewrites. Analyses restricted to the A condition can use 73 items and
analyses restricted to B can use 60, while any analysis that contrasts the two conditions within an
item should use the 58-item set.

### Defect taxonomy of the 23 excluded items

| defect | n | remedy |
|---|---:|---|
| B contradicts a sentence retained from the original | 7 | regenerate B |
| Original passage does not determine an answer (hedged or off-topic) | 5 | none — permanent drop |
| B's edited sentence contradicts itself | 4 | regenerate B |
| B negates a proposition other than the one asked about | 3 | regenerate B |
| A is a verbatim copy of the original passage | 2 | regenerate A |
| Rewrite refers to deleted content with no antecedent | 2 | regenerate A and/or B |

The dominant failure mode is local: the generator edits the target sentence but leaves the rest of
the passage asserting what the edit denies, so the passage as a whole becomes incoherent rather than
counterfactual. The second pattern — three items where B negates a neighbouring proposition instead
of the one the question asks about (a volume figure instead of an identity claim, "not behind"
instead of "in front") — is the more interesting one for the paper, because such items look
well-formed and would silently enter the dataset without item-level review. `regeneration_worklist.csv`
lists all 18 recoverable items with the condition to regenerate and, where the fix is specific, the
sentence that should replace the faulty one.

## Note on how this validation is labelled

The files carry the name *human validation*. If both reviewers were model-based rather than human,
that label should be changed — "independent double coding with adjudication" describes the procedure
without the claim, and it costs nothing, since the procedure's value here lies in the two independent
passes and the written adjudication, not in the identity of the coders. If a human-validation claim
is wanted, the usual minimum is a human pass over a sample of the items (for example the 34
adjudicated ones plus a random sample of the rest) reported with its own agreement statistics.
