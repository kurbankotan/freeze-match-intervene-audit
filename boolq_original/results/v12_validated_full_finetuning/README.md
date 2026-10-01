# V12 optimization-validated trainable-backbone control

This control retains the frozen-V9 data split, seeds, LRAA architecture,
adapter/scorer learning rate (`2e-4`), three epochs, batching, warm-up, and
checkpoint-selection rule. The DeBERTa-v3-large encoder is trainable at the
pre-existing encoder learning rate `8e-6`.

All three prespecified optimization checks passed. Mean enabled held-out
accuracy, balanced accuracy, and macro-F1 are 89.39%, 88.75%, and 88.73%.
Disabling LRAA changes balanced accuracy by +0.81 point, with a 95% paired-seed
interval of [-0.14, +1.75]. The corresponding frozen-V9 loss is +27.62 points.
The frozen-minus-trainable difference is +26.81 points, with a 95% interval of
[+19.90, +33.72], and is positive in all three matched seeds. The prespecified
training-regime decision is `PASS`.

The ZIP omits full encoder checkpoints because three copies would be several
gigabytes. It contains complete held-out predictions, histories, configuration,
environment, small adapter/scorer states, and a SHA-256 result manifest.
