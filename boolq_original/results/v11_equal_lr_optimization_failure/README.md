# V11 equal-learning-rate optimization failure

V11 applied the V9 adapter learning rate (`2e-4`) to the entire trainable
DeBERTa-v3-large model. The run completed and its integrity manifest passes, but
the enabled model did not learn the task: training loss remained near
`log(2)`, training accuracy remained near 50%, and mean held-out balanced
accuracy was 50.74%.

The nominal training-regime `PASS` recorded by the original V11 analysis must
therefore **not** be interpreted as evidence that full fine-tuning hides adapter
contribution. The small adapter ablation effect is confounded by failed
optimization. These files are retained as an auditable negative run rather than
silently discarded.

V12 corrects the analysis procedure by using a separately prespecified encoder
learning rate and by requiring an optimization-validity gate before the
adapter-attribution decision can be evaluated.
