# A/B Experiment Summary: Scratch vs Transfer

Generated: `2026-03-12T09:44:55.600096`

## Winner
- Decision: **tie**
- Reason: Metrics are effectively tied under current protocol.

## Scratch Metrics
- Keyword recall: `100.00%`
- False trigger rate: `0.00%`
- Weak-label accuracy: `100.00%`
- Avg positive score on positives: `99.9942`
- Avg positive score on non-positives: `0.7257`
- Quantized model size (MB): `0.0328`

## Transfer Metrics
- Keyword recall: `100.00%`
- False trigger rate: `0.00%`
- Weak-label accuracy: `100.00%`
- Avg positive score on positives: `99.9021`
- Avg positive score on non-positives: `0.1417`
- Quantized model size (MB): `4.9450`

## Protocol Notes
- Both methods use the same external `test/` folder outputs.
- For flat test files, labels are inferred with filename heuristic:
  - `positive*` => `heytm`
  - names containing `background|silence|noise` => `background`
  - otherwise => `unknown`
- Prefer replacing heuristic labels with a manually labeled test set for final model selection.
